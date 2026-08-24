import csv
import time
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from generate import adjust_logits, load_prompts

if torch.mps.is_available():
        device = "mps"
        torch.mps.set_per_process_memory_fraction(0.95)
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def calculate_alpha(draft_model, batch_file, V, eos_id): 
    out = torch.load(batch_file, map_location="cpu")
    seq = torch.cat([out["prefix_ids"], out["generated_ids"]], dim=1).to(device)
    with torch.inference_mode():
        draft_logits = draft_model(input_ids=seq).logits[:, config.PREFIX_LEN - 1 : -1, :]

    assert draft_logits.shape[1] == config.GENERATION_LEN

    q = torch.softmax(adjust_logits(draft_logits, eos_id, V), dim=-1)
    p = torch.softmax(out["generated_logits"].float().to(device), dim=-1)  ### already adjusted during generation to mask EOS
    alpha = torch.minimum(p, q).sum(dim=-1)  ### the acceptance probability per token
    agree = (q.argmax(-1) == p.argmax(dim=-1))  ### does greedy agree
    return out["prompt_ids"], alpha.mean(dim=1).cpu(), agree.float().mean(dim=1).cpu()

def csv_path(target_id, draft_id):
    return config.ALPHA_DIR / config.model_name(target_id) / f"{config.model_name(draft_id)}.csv"

def draft_done(target_id, draft_id):
    return csv_path(target_id, draft_id).exists()

def write_csv(target_id, draft_id, prompt_ids, prompt_info, alphas, agrees):
    final = csv_path(target_id, draft_id)
    tmp = final.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "domain", "empirical", "alpha", "agree"])
        for prompt_id, alpha, agree in zip(prompt_ids.tolist(), alphas.tolist(), agrees.tolist()):
            domain, empirical = prompt_info[prompt_id]
            w.writerow([prompt_id, domain, empirical, f"{alpha:.6f}", f"{agree:.6f}"])
    os.rename(tmp, final)

if __name__ == "__main__":
    prompts = load_prompts()
    prompt_info = {p["id"]: (p["domain"], p["empirical"]) for p in prompts}  ### dict comprehension - cool stuff
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)
    for target_id in config.TARGET_IDS:
        (config.ALPHA_DIR / config.model_name(target_id)).mkdir(parents=True, exist_ok=True)
        for draft_id in config.DRAFT_IDS:

            if draft_done(target_id, draft_id):  ### resume
                print(f"skipping {config.model_name(draft_id)} (csv exists)")
                continue
            print(f"scoring {config.model_name(draft_id)}")

            start = time.perf_counter()

            draft_model = AutoModelForCausalLM.from_pretrained(draft_id, dtype=torch.bfloat16).to(device).eval()

            prompt_id_batches, alpha_batches, agree_batches = [], [], []  ### process in batches and acculumate
            for draft_batch_file in sorted((config.TARGETS_DIR / config.model_name(target_id)).glob("batch_*.pt")):  ### glob comes from unix, filename pattern matching
                prompt_ids, alphas, agrees = calculate_alpha(draft_model, draft_batch_file, len(tokenizer), tokenizer.eos_token_id)
                prompt_id_batches.append(prompt_ids)
                alpha_batches.append(alphas)
                agree_batches.append(agrees)

            prompt_id_batches, alpha_batches, agree_batches = torch.cat(prompt_id_batches), torch.cat(alpha_batches), torch.cat(agree_batches)

            assert len(prompt_id_batches) == len(prompts), f"{config.model_name(draft_id)} didn't score all rows"

            write_csv(target_id, draft_id, prompt_id_batches, prompt_info, alpha_batches, agree_batches)

            del draft_model
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            print(f"Analyzing alphas - {config.model_name(target_id)} - {config.model_name(draft_id)} finished in {time.perf_counter() - start:,.0f}s")
