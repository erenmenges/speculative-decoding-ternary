import os

import torch
from transformers import AutoModelForCausalLM

import config


# i took this from my other project "Avenue" and adopted it to per-matrix
def quantize_weights(W: torch.Tensor):
    quantized_W = W.float()  ### cast to fp32
    abs_mu = quantized_W.abs().mean() + 1e-5  ### shape: scalar
    quantized_W = quantized_W / abs_mu  ### drop the scale. each param is now "how many avg weights is this?"
    quantized_W = torch.round(quantized_W)
    quantized_W = torch.clamp(quantized_W, min=-1, max=1)
    return quantized_W.to(W.dtype) * abs_mu


def ternarize(model_id):
    assert config.model_name(model_id).startswith("FloatLM")
    lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    dead_zone_ratio = []
    with torch.no_grad():
        for block in lm.model.layers:
            for layer in (
                block.self_attn.q_proj,
                block.self_attn.k_proj,
                block.self_attn.v_proj,
                block.self_attn.o_proj,
                block.mlp.gate_proj,
                block.mlp.up_proj,
                block.mlp.down_proj,
            ):
                w_q = quantize_weights(layer.weight)
                layer.weight.copy_(w_q)
                dead_zone_ratio.append((w_q == 0).float().mean())
                assert layer.weight.unique().numel() <= 3, "ternarization screwed up. more than {-1,0,1} values"

    print(f"Avg dead zone ratio: {sum(dead_zone_ratio) / len(dead_zone_ratio):.2f}")
    return lm.to(dtype=torch.bfloat16)

if __name__ == "__main__":
    config.PTQ_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in config.TO_BE_TERNARIZED_IDS:
        model_name = config.model_name(model_id)
        final = config.PTQ_DIR / f"PTQ_{model_name.split("_")[-1]}"
        if final.exists():
            continue

        ternarized_lm = ternarize(model_id)

        tmp = final.parent / ("tmp_" + final.name)

        ternarized_lm.save_pretrained(tmp)
        os.rename(tmp, final)