import csv
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import config
from generate import adjust_logits, load_prompts

if torch.mps.is_available():
    device = "mps"
    torch.mps.set_per_process_memory_fraction(0.95)
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def spec_decode_one(prompt_tokens, target, draft, V, eos_id, seed, k=config.K):
    torch.manual_seed(seed)
    seq = torch.tensor(prompt_tokens, device=device).unsqueeze(dim=0)  ### batch one

    accepted_per_round = []  ### how many tokens are accepted per round, one round might accept 1, other might accept 3
    with torch.inference_mode():
        while seq.shape[1] < len(prompt_tokens) + config.GENERATION_LEN:  ### we don't know how many rounds it will take, so while loop
            draft_tokens, draft_qs, draft_distributions = [], [], []
            for _ in range(k):
                draft_input = torch.cat([seq, torch.tensor([draft_tokens], device=device)], dim=1) if draft_tokens else seq

                draft_logits = draft(input_ids=draft_input).logits[:, -1, :]
                q = torch.softmax(adjust_logits(draft_logits, eos_id, V), dim=-1)
                selected_token = torch.multinomial(q, num_samples=1)
                draft_tokens.append(int(selected_token))
                draft_qs.append(float(q[0, selected_token]))
                draft_distributions.append(q[0])

            target_input = torch.cat([seq, torch.tensor([draft_tokens], device=device)], dim=1)
            target_logits = target(input_ids=target_input).logits[:, -(k + 1):, :]  ### -(k + 1) because we want the last k tokens plus the target's own prediction before all that
            p_distributions = torch.softmax(adjust_logits(target_logits, eos_id, V), dim=-1)
            n_accepted = 0
            for i in range(k):
                p_i = float(p_distributions[0, i, draft_tokens[i]])  ### x is the token proposed by the draft. we check x's probability in target's dist.
                if torch.rand(()) < min(1.0, p_i / draft_qs[i]):  ### min(1, p(x)/q(x)), the formula
                    n_accepted += 1
                else:
                    break
            if n_accepted < k:
                p_rejected = p_distributions[0, n_accepted]  ### if 2 tokens were accepted, first rejection occurred at pos 2 (0->1->2)
                q_rejected = draft_distributions[n_accepted]

                residual = torch.clamp(p_rejected - q_rejected, min=0)

                total = residual.sum()
                assert total > 1e-9, "residual can't be 0, otherwise rejection wouldn't have happened"
                residual = residual / total

                extra_token = int(torch.multinomial(residual.unsqueeze(0), num_samples=1))  ### sample from the residual, get correct token
            else:
                extra_token = int(torch.multinomial(p_distributions[0, k], num_samples=1))  ### get the bonus token

            new_tokens = draft_tokens[:n_accepted] + [extra_token]
            seq = torch.cat([seq, torch.tensor([new_tokens], device=device)], dim=1)
            accepted_per_round.append(n_accepted)

    seq = seq[:, :len(prompt_tokens) + config.GENERATION_LEN]  ### last round can overshoot, trim it
    return accepted_per_round, seq[0].tolist()

def cell_csv(draft_id):
    return config.EMPIRICAL_DIR / f"{config.model_name(draft_id)}.csv"


if __name__ == "__main__":
    CELLS = [
        "SpectraSuite/FloatLM_99M", "SpectraSuite/FloatLM_560M", "SpectraSuite/FloatLM_1.5B",
        "SpectraSuite/TriLM_99M_Unpacked", "SpectraSuite/TriLM_560M_Unpacked", "SpectraSuite/TriLM_1.5B_Unpacked",
        str(config.PTQ_DIR / "PTQ_99M"), str(config.PTQ_DIR / "PTQ_560M"), str(config.PTQ_DIR / "PTQ_1.5B"),
    ]

    config.EMPIRICAL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
    V, eos_id = len(tokenizer), tokenizer.eos_token_id
    prompts = load_prompts(only_empirical=True)
    print(f"{len(prompts)} prompts loaded.")

    target_model = AutoModelForCausalLM.from_pretrained(config.TARGET_IDS[0], dtype=torch.bfloat16).to(device).eval()

    for draft_id in CELLS:
        if cell_csv(draft_id).exists():
            print(f"skip {config.model_name(draft_id)}")
            continue
        draft_model = AutoModelForCausalLM.from_pretrained(draft_id, dtype=torch.bfloat16).to(device).eval()
        start = time.perf_counter()
        rows = []
        for p in tqdm(prompts, desc=config.model_name(draft_id)):
            accepted, _ = spec_decode_one(p["tokens"], target_model, draft_model, V, eos_id, config.EMPIRICAL_SEED + p["id"], config.K)
            rejected_rounds = sum(1 for a in accepted if a < config.K)
            rows.append((p["id"], len(accepted), sum(accepted), rejected_rounds))  ### prompt id, how many rounds, how many tokens accepted total, how many rounds ended with rejection

        with open(cell_csv(draft_id).with_suffix(".tmp"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "rounds", "accepted_total", "rejected_rounds"])
            w.writerows(rows)
        os.rename(cell_csv(draft_id).with_suffix(".tmp"), cell_csv(draft_id))

        del draft_model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
        print(f"{config.model_name(draft_id)} finished in {time.perf_counter() - start:.0f}s")