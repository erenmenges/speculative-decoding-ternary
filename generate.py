import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_prompts(only_empirical: bool = False):
    raw = json.loads((config.DATA_DIR / "prompts.json").read_text())
    meta, prompts = raw["meta"], raw["prompts"]
    for i, p in enumerate(prompts):
        p["id"] = i
    if only_empirical:
        prompts = [p for p in prompts if p["empirical"]]
    assert meta["data_seed"] == config.DATA_SEED
    return prompts

def make_batches(prompts):
    for batch_idx, start in enumerate(range(0, len(prompts), config.BATCH_SIZE)):  ### so we get (0, 0), (1, 64), (2, 128), ..
        yield batch_idx, prompts[start : start + config.BATCH_SIZE]

def adjust_logits(logits: torch.Tensor, eos_id: int, vocab_size: int):
    logits = logits[..., :vocab_size].float() / 1.0  ### temp 1.0
    logits[..., eos_id] = float("-inf")
    return logits

def generate_logits(lm, batch: list, batch_idx: int, V: int, eos_id: int, device: str):
    torch.manual_seed(config.GEN_SEED + batch_idx)


    context_prefix = torch.tensor([prompt["tokens"] for prompt in batch])
    prompt_ids = torch.tensor([prompt["id"] for prompt in batch])

    current_input = context_prefix.to(device) ### the whole prefix at first
    assert current_input.shape[1] == config.PREFIX_LEN

    generated_logits = []
    generated_ids = []
    past = None

    with torch.inference_mode():
        for _ in range(config.GENERATION_LEN):
            lm_out = lm(input_ids=current_input, past_key_values=past, use_cache=True)
            last_logits = adjust_logits(lm_out.logits[:, -1, :], eos_id, V)
            last_logits_fp16 = last_logits.to(dtype=torch.float16)  ### we save fp16
            p = torch.softmax(last_logits, dim=-1)  ### we use fp32 for softmax
            current_input = torch.multinomial(p, num_samples=1)  ### make it such that the only input is the last token, since we have KV cache

            generated_logits.append(last_logits_fp16.to(device="cpu"))
            generated_ids.append(current_input.to(device="cpu"))

            past = lm_out.past_key_values

    saved_logits = torch.stack(generated_logits, dim=1)
    saved_ids = torch.cat(generated_ids, dim=1)

    assert saved_logits.shape == (config.BATCH_SIZE, config.GENERATION_LEN, V)
    assert saved_ids.shape == (config.BATCH_SIZE, config.GENERATION_LEN)

    return {
        "prompt_ids": prompt_ids,
        "prefix_ids": context_prefix.cpu(),
        "generated_ids": saved_ids,
        "generated_logits": saved_logits
        }

def batch_path(target_id, batch_idx):
    return config.TARGETS_DIR / config.model_name(target_id) / f"batch_{batch_idx:04d}.pt"

def batch_done(target_id, batch_idx):
    return batch_path(target_id, batch_idx).exists()

def save_batch(out, target_id, batch_idx):
    final = batch_path(target_id, batch_idx)
    tmp = final.with_suffix(".tmp")
    torch.save(out, tmp)
    os.rename(tmp, final)

if __name__ == "__main__":
    config.TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    if torch.mps.is_available():
        device = "mps"
        torch.mps.set_per_process_memory_fraction(0.9)
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)

    assert tokenizer.eos_token_id is not None

    prompts = load_prompts()
    for target_id in config.TARGET_IDS:
        (config.TARGETS_DIR / config.model_name(target_id)).mkdir(parents=True, exist_ok=True)

        target_model = AutoModelForCausalLM.from_pretrained(target_id, dtype=torch.bfloat16).to(device).eval()

        for batch_idx, batch in make_batches(prompts):
            start = time.perf_counter()
            if batch_done(target_id, batch_idx):  ### resume functionality
                continue

            out = generate_logits(target_model, batch, batch_idx, len(tokenizer), tokenizer.eos_token_id, device)
            save_batch(out, target_id, batch_idx)

            if device == "mps":
                torch.mps.empty_cache()
            print(f"Target {target_id} - Batch {batch_idx} is done in {time.perf_counter() - start:,.0f}s")

        del target_model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    
    print(f"Finished.")