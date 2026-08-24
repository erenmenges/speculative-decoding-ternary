from transformers import AutoTokenizer

import config

tokenizers = [AutoTokenizer.from_pretrained(model_id) for model_id in config.ALL_IDS]

for i, t1 in enumerate(tokenizers):
    for k, t2 in enumerate(tokenizers):
        assert len(t1) == len(t2), f"tokenizer length mismatch at {config.ALL_IDS[i]} and {config.ALL_IDS[k]}"
        t1_vocab = t1.get_vocab()
        t2_vocab = t2.get_vocab()
        assert t1_vocab == t2_vocab, f"tokenizer vocab mismatch at {config.ALL_IDS[i]} and {config.ALL_IDS[k]}"
        assert t1.special_tokens_map == t2.special_tokens_map, f"tokenizer special tokens mismatch at {config.ALL_IDS[i]} and {config.ALL_IDS[k]}"
        assert t1.all_special_ids == t2.all_special_ids, f"tokenizer special tokens id's mismatch at {config.ALL_IDS[i]} and {config.ALL_IDS[k]}"
        text = "The quick red fox jumped over the lazy dog. Why?"
        a = t1(text, add_special_tokens=False)["input_ids"]
        b = t2(text, add_special_tokens=False)["input_ids"]
        assert a == b, f"tokenizer example mismatch at {config.ALL_IDS[i]} and {config.ALL_IDS[k]}"

print("Vocabulary matches.")

