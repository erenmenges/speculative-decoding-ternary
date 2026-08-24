from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForCausalLM

import config


def load_data():
    rows = []
    for target_dir in config.ALPHA_DIR.iterdir():  ### iter thru alpha target folders
        for csv_file in target_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            df["target"] = target_dir.name
            df["draft"] = csv_file.stem
            rows.append(df)

    data = pd.concat(rows, ignore_index=True)

    def family(draft):
        return "PTQ" if draft.startswith("PTQ") else "TriLM" if draft.startswith("TriLM") else "FloatLM"

    data["family"] = data["draft"].map(family)  ### map() takes function and applies it 
    data["size_m"] = data["draft"].map(config.SIZE_M)
    return data

def calculate_confidence_interval(values, n_subset=2000, seed=config.PLOTTING_SEED):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_subset)])  ### sample with replacement - fake population sampling
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def print_cell_confidence_intervals(data):
    for key, group in data.groupby(["target", "family", "size_m", "domain"]):
        values = group["alpha"].to_numpy()
        low, high = calculate_confidence_interval(values, 2000, config.PLOTTING_SEED)
        target, family, size_m, domain = key
        print(f"{target:20s} {family:8s} {size_m:5.0f} {domain:5s} "
              f"{values.mean():.4f} [{low:.4f}, {high:.4f}]  n={len(values)}")  ### 20s pads the string to 20 chars always, 5.0f pads the number to a 5 digit int


def paired_gap_ci(data, target, draft_a, draft_b, domain):
    sub = data[(data["target"] == target) & (data["domain"] == domain)]  ### get our domain and our target model's data only
    a = sub[sub["draft"] == draft_a].set_index("prompt_id")["alpha"]  ### get draft a's alphas
    b = sub[sub["draft"] == draft_b].set_index("prompt_id")["alpha"]

    assert len(a) > 0, "draft table returned 0 rows"
    assert len(a) == len(b) == a.index.isin(b.index).sum()

    diff = (a - b).to_numpy()

    low, high = calculate_confidence_interval(diff, 2000, config.PLOTTING_SEED)
    return diff.mean(), low, high

def plot_paired_gaps(data, domain, sizes=("99M", "190M", "390M", "560M", "830M", "1.1B", "1.5B")):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for target_id in config.TARGET_IDS:
        target = config.model_name(target_id)
        xs, means, lows, highs = [], [], [], []
        for size in sizes:
            mean, low, high = paired_gap_ci(data, target, f"TriLM_{size}_Unpacked", f"FloatLM_{size}", domain)
            xs.append(config.SIZE_M[f"FloatLM_{size}"])
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        label = target.replace("_Unpacked", "").replace("_", " ")
        ax.errorbar(xs, means, yerr=[lows, highs], fmt="o-", capsize=3, label=f"target: {label}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter(sizes))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("draft model params (M)")
    ax.set_ylabel(r"$\alpha$(TriLM draft) $-$ $\alpha$(FloatLM draft)")
    ax.set_title(f"paired gap at matched params ({domain})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DATA_DIR / f"paired_gap_{domain}.png", dpi=200)
    plt.close(fig)

def plot_domain_alphas(data: pd.DataFrame, domain: str, families, ymin, tag, x_col="size_m", xlabel="draft model params (M)", xticks=None, xticklabels=None):  ## by default this gives us a model n_params vs alpha plot
    sub = data[(data["domain"] == domain) & (data["family"].isin(families))]  ### extract our domain and model families
    for target, tgroup in sub.groupby("target"):
        fig, ax = plt.subplots(figsize=(5, 3.5) if len(families) > 1 else (3,3))  ### PTQ-only graph is smaller
        for family, fgroup in tgroup.groupby("family"):
            sizes, means, lows, highs = [], [], [], []
            for size, sgroup in fgroup.groupby(x_col):
                values = sgroup["alpha"].to_numpy()
                low, high = calculate_confidence_interval(values, 2000, config.PLOTTING_SEED)
                sizes.append(size)
                means.append(values.mean())
                lows.append(values.mean() - low)
                highs.append(high - values.mean())
            ax.errorbar(sizes, means, yerr=[lows, highs], fmt="o-", markersize=3, capsize=2, elinewidth=1.2, label=family)
        ax.set_xscale("log")
        if xticks is not None:
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.xaxis.set_major_formatter(FixedFormatter(xticklabels))
            ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(ymin, None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean alpha")

        ax.set_title(f"target: {target} ({domain})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.DATA_DIR / f"alpha_vs_{x_col}_{tag}_{target}_{domain}.png", dpi=200)
        plt.close(fig)

def load_empirical():
    rows = []
    for csv_file in config.EMPIRICAL_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df["draft"] = csv_file.stem
        rows.append(df)

    emp = pd.concat(rows, ignore_index=True)
    emp["measured_tokens_per_round"] = (emp["accepted_total"] + emp["rounds"] ) / emp["rounds"]
    emp["alpha_hat"] = emp["accepted_total"] / (emp["accepted_total"] + emp["rejected_rounds"])
    counts = emp.groupby("draft").size()
    assert counts.nunique() == 1, "empirical cells have different counts per draft"
    return emp

def predicted_tokens_per_round(alpha, k: int = config.K):
    return sum(alpha ** i for i in range(k + 1))  ### (E[n_accepted] = a + a^2 + a^3 + ... + a^k) + 1

def make_validation_table(data, empirical_prompts):
    analytical = data[(data["target"] == "FloatLM_3.9B") & (data["draft"].isin(empirical_prompts["draft"].unique()))]  ### only get the analytical data that we have an empirical counterpart for
    merged = empirical_prompts.merge(analytical[["prompt_id", "draft", "domain", "alpha"]], on=["prompt_id", "draft"])
    merged["predicted_tokens_per_round"] = predicted_tokens_per_round(merged["alpha"])
    assert len(merged) == len(empirical_prompts), "analytical empirical merge screwed up"
    return merged

def plot_validation(merged):
    cell = merged.groupby(["draft", "domain"])[["predicted_tokens_per_round", "measured_tokens_per_round"]].mean()
    fig, ax = plt.subplots()
    for domain, group in cell.groupby("domain"):
        ax.scatter(group["predicted_tokens_per_round"], group["measured_tokens_per_round"], label=domain)
    lims = [cell.min().min() - 0.1, cell.max().max() + 0.1]  ### the 0.1
    ax.plot(lims, lims, linestyle="--", linewidth=1, color="gray")  ### empirical should ideally match analytical, so we draw a slope-1 line to see if dots sit on it
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted tokens/round from analytical alpha")
    ax.set_ylabel("measured tokens/round, from real time speculative decoding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DATA_DIR / "validation_scatter.png", dpi=200)
    plt.close(fig)

def parity_ratios(data, target, domain, x_col="size_m", unit="M"):
    sub = data[(data["target"] == target) & (data["domain"] == domain) & (data["size_m"] < 3900)]
    float_lm = sub[sub["family"] == "FloatLM"].groupby(x_col)["alpha"].mean()
    tri_lm = sub[sub["family"] == "TriLM"].groupby(x_col)["alpha"].mean()
    ratios = []

    log_x = np.log(float_lm.index.to_numpy())

    alphas = float_lm.to_numpy()
    assert (np.diff(alphas) > 0).all(), "alpha not increasing across sizes/memory footprint, the np.interp will return bullshit"

    for size, alpha in tri_lm.items():
        if not (alphas[0] <= alpha <= alphas[-1]):  ### it would be stupid to ask what floatLM matches ternary-99M, because we don't have data for it
            print(f"TriLM {size:5.0f}{unit} outside the grid, skipped")
            continue

        interpreted_size_log = np.interp(alpha, alphas, log_x)  ### finds what size would we have if we asked "what size floatlm do we need to match ternary's alpha?"
        equivalent_size = np.exp(interpreted_size_log)
        ratio = equivalent_size / size
        print(f"TriLM {size:5.0f}{unit} ≈ FloatLM {equivalent_size:6.0f}{unit}  →  ratio {ratio:.2f}x")
        ratios.append(ratio)
    return ratios
    
TERNARY_BITS = 2
FLOAT_BITS = 16

def split_params(model_id):
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    quantizable, rest, n_proj = 0, 0, 0

    for name, p in lm.named_parameters():
        if name.startswith("model.layers.") and name.endswith("_proj.weight"):
            quantizable += p.numel()
            n_proj += 1
        else:
            rest += p.numel()
    assert n_proj == 7 * lm.config.num_hidden_layers, f"something went wrong and not all stuff got properly counted. {model_id}"
    return quantizable, rest, lm.config.tie_word_embeddings

def megabytes(quantizable, rest, w_bits):
    return (quantizable * w_bits + rest * FLOAT_BITS) / 8 / 1e6  ### divide to 8 because we convert it to bytes

def bits_for(model_id):
    return FLOAT_BITS if config.model_name(model_id).startswith("FloatLM") else TERNARY_BITS

def load_footprints():
    footprints = {}
    for model_id in config.DRAFT_IDS:
        quantizable, rest, tied = split_params(model_id)
        name = config.model_name(model_id)
        footprints[name] = megabytes(quantizable, rest, bits_for(model_id))
        print(f"{name:22s}----{quantizable/1e6:7.1f}M quant----{rest/1e6:6.1f}M rest----tied={tied}----{footprints[name]:7.1f} MB")
    return footprints

def plot_speedup(data, domain, target, footprints, families=("FloatLM", "TriLM", "PTQ"), k=config.K):
    target_mb = footprints[target]
    sub = data[(data["target"] == target) & (data["domain"] == domain) & (data["family"].isin(families))]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    print(f"--- speedup: {target} / {domain} (k={k}) ---")
    for family, fgroup in sub.groupby("family"):
        cell = fgroup.groupby("mb")["alpha"].mean()
        mbs = cell.index.to_numpy()
        speedup = predicted_tokens_per_round(cell.to_numpy(),  k) / (1 + k * mbs / target_mb)
        ax.plot(mbs, speedup, "o-", label=family)
        for mb, s, a in zip(mbs, speedup, cell.to_numpy()):
            print(f"{family:8s} {mb:7.1f}MB  alpha {a:.4f}  speedup {s:.2f}x")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("draft weight memory (MB)")
    ax.set_ylabel("predicted speedup")
    ax.set_title(f"target: {target} ({domain})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DATA_DIR / f"speedup_{target}_{domain}.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    data = load_data()

    # make sure 3.9B models have a ~0.99 alpha
    print("-" * 16, "Raw Analytical Results", "-" * 16)
    pivot = data.groupby(["target", "family", "size_m", "domain"])[["alpha", "agree"]].mean().round(4)
    print(pivot.to_string())

    data = data[data["size_m"] < 3900]

    footprints = load_footprints()
    data["mb"] = data["draft"].map(footprints)  ## dict lookup
    assert data["mb"].notna().all(), "a model has no footprint entry"

    SIZES = ["99M", "190M", "390M", "560M", "830M", "1.1B", "1.5B"]
    XTICKS = [config.SIZE_M[f"FloatLM_{s}"] for s in SIZES]

    # main
    plot_domain_alphas(data, "web", ["FloatLM", "TriLM", "PTQ"], ymin=0.0, tag="all", x_col="mb", xlabel="draft weight memory (MB)")
    plot_domain_alphas(data, "web", ["FloatLM", "TriLM"], ymin=0.5, tag="main", x_col="mb", xlabel="draft weight memory (MB)")

    # supplementary, maybe used
    plot_domain_alphas(data, "web", ["FloatLM", "TriLM"], ymin=0.5, tag="main", xticks=XTICKS, xticklabels=SIZES)

    print("-" * 16, "Analytical vs Empirical Results", "-" * 16)
    emp = load_empirical()
    merged = make_validation_table(data, emp)
    plot_validation(merged)
    val = merged.groupby(["draft", "domain"])[["alpha", "alpha_hat", "predicted_tokens_per_round", "measured_tokens_per_round"]].mean()
    val["alpha_gap"] = val["alpha_hat"] - val["alpha"]
    val["tpr_gap"] = val["measured_tokens_per_round"] - val["predicted_tokens_per_round"]
    print(val.round(4).to_string())
    print(f"alpha gap      mean {val['alpha_gap'].mean():+.4f}  max|.| {val['alpha_gap'].abs().max():.4f}")
    print(f"tokens/rnd gap mean {val['tpr_gap'].mean():+.4f}  max|.| {val['tpr_gap'].abs().max():.4f}")

    plot_speedup(data, "web", "FloatLM_3.9B", footprints)
    plot_speedup(data, "web", "TriLM_3.9B_Unpacked", footprints)

    print("-" * 16, "Cell CIs", "-" * 16)
    print_cell_confidence_intervals(data)

    print("-" * 16, "Paired Gap CIs", "-" * 16)
    plot_paired_gaps(data, "web")
    for target in config.TARGET_IDS:
        target = config.model_name(target)
        for domain in ["web", "chat", "code"]:
            for size in ["99M", "190M", "390M", "560M", "830M", "1.1B", "1.5B"]:
                mean, low, high = paired_gap_ci(data, target, f"TriLM_{size}_Unpacked", f"FloatLM_{size}", domain)
                print(f"{target:20s} {domain:5s} {size:6s} {mean:+.4f} [{low:+.4f}, {high:+.4f}]")

    print("-" * 16, "Parity ratios", "-" * 16)
    for target in config.TARGET_IDS:
        target = config.model_name(target)
        for domain in ["web", "chat", "code"]:
            print(f"--- {target} / {domain} (params) ---")
            parity_ratios(data, target, domain)
            print(f"--- {target} / {domain} (memory) ---")
            parity_ratios(data, target, domain, x_col="mb", unit="MB")

    print("-" * 16, "USEFUL SUMMARY FOR ME", "-" * 16)

    for target in [config.model_name(t) for t in config.TARGET_IDS]:
        print(f"\n=== {target} / web ===")
        for size in ["99M", "1.5B"]:
            mean, low, high = paired_gap_ci(data, target, f"TriLM_{size}_Unpacked", f"FloatLM_{size}", "web")
            print(f"  paired gap {size:5s} {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
        print("  memory parity:")
        parity_ratios(data, target, "web", x_col="mb", unit="MB")

    print("\n=== PTQ vs TriLM, matched footprint (web, FloatLM_3.9B) ===")
    sub = data[(data["target"] == "FloatLM_3.9B") & (data["domain"] == "web")]
    for size in ["99M", "1.5B"]:
        tri = sub[sub["draft"] == f"TriLM_{size}_Unpacked"]["alpha"].mean()
        ptq = sub[sub["draft"] == f"PTQ_{size}"]["alpha"].mean()
        print(f"  {size:5s} {footprints[f'PTQ_{size}']:6.1f} MB   TriLM {tri:.4f}   PTQ {ptq:.4f}")

    print("\n=== validation ===")
    print(f"  alpha gap      mean {val['alpha_gap'].mean():+.4f}  max|.| {val['alpha_gap'].abs().max():.4f}")
    print(f"  tokens/rnd gap mean {val['tpr_gap'].mean():+.4f}  max|.| {val['tpr_gap'].abs().max():.4f}")

    print("\n=== Fig 1: web, FloatLM_3.9B target ===")
    sub = data[(data["target"] == "FloatLM_3.9B") & (data["domain"] == "web")]
    for size in ["99M", "190M", "390M", "560M", "830M", "1.1B", "1.5B"]:
        f, t = f"FloatLM_{size}", f"TriLM_{size}_Unpacked"
        fa, ta = sub[sub["draft"] == f]["alpha"].mean(), sub[sub["draft"] == t]["alpha"].mean()
        print(f"  {size:5s}  FloatLM {footprints[f]:7.1f}MB {fa:.4f}   TriLM {footprints[t]:6.1f}MB {ta:.4f}   delta {ta - fa:+.4f}")
