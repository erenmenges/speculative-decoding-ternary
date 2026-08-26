import json
import random
from collections.abc import Callable

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import config

ALLOWED_CODING_LANGS = ["Python", "C++", "JavaScript", "Java"]


def download_models():
    for id in config.ALL_IDS:
        snapshot_download(repo_id=id)


def extract_from_web(example):
    return example["text"]


def extract_from_chat(example):
    if example["language"] != "English":
        return None
    lines = []
    for turn in example["conversation"]:
        if turn["content"] is None:
            return None
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n\n".join(lines)


def extract_from_code(example):
    for f in example["files"]:
        if (
            f["language"] in ALLOWED_CODING_LANGS
            and f["license_type"] == "permissive"
            and 2_000 < f["size_bytes"] < 200_000
            and not f["is_vendor"]
            and f["file_timestamp"] > config.CUTOFF_UNIX
        ):
            return f["content"]
    return None


def make_prompt(example: str, tokenizer):
    ids = tokenizer(example, add_special_tokens=False)["input_ids"]
    if len(ids) < config.MIN_TOKENS_IN_EXAMPLE:
        return None
    start = 500
    assert len(ids[start:start + config.PREFIX_LEN]) == config.PREFIX_LEN, "prompt not 256 tokens"
    return ids[start : start + config.PREFIX_LEN]


def collect(stream, extract: Callable, domain: str, quota: int, tokenizer):
    assert config.MIN_TOKENS_IN_EXAMPLE >= 500 + config.PREFIX_LEN + config.GENERATION_LEN

    prompts = []
    seen = set()
    for i, example in enumerate(stream):
        text = extract(example)
        if text is None:
            continue
        ids = make_prompt(text, tokenizer)
        if ids is None:
            continue
        ids_tuple = tuple(ids)
        if ids_tuple in seen:
            continue
        seen.add(ids_tuple)
        prompts.append({"domain": domain, "source_index": i, "tokens": ids})
        if len(prompts) % 64 == 0:
            print(f"Collected {len(prompts)} prompts in {domain}")
        if len(prompts) == quota:
            break
    assert len(prompts) == quota
    return prompts


def create_empirical_subset(prompts: list, subset_size: int):
    rng = random.Random(config.DATA_SEED)
    for p in prompts:
        p["empirical"] = False
    for p in rng.sample(prompts, subset_size):
        p["empirical"] = True


def save_prompts(prompts: list[dict]):
    out = {
        "meta": {
            "data_seed": config.DATA_SEED,
            "datasets": {
                "web": "allenai/c4 (en, validation)",
                "code": "HuggingFaceCode/stack-v3-train",
                "chat": "allenai/WildChat-4.8M",
            },
        },
        "prompts": prompts,
    }
    path = config.DATA_DIR / "prompts.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(prompts)} prompts to {path}")


if __name__ == "__main__":
    download_models()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ID)

    web = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    web = web.shuffle(seed=config.DATA_SEED, buffer_size=1000)
    web_prompts = collect(web, extract_from_web, "web", 512, tokenizer)

    chat = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
    chat = chat.shuffle(seed=config.DATA_SEED, buffer_size=1000)
    chat_prompts = collect(chat, extract_from_chat, "chat", 256, tokenizer)

    code = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True)
    code = code.shuffle(seed=config.DATA_SEED, buffer_size=500)
    code_prompts = collect(code, extract_from_code, "code", 256, tokenizer)

    create_empirical_subset(web_prompts, 128)
    create_empirical_subset(chat_prompts, 64)
    create_empirical_subset(code_prompts, 64)

    save_prompts(web_prompts + chat_prompts + code_prompts)
