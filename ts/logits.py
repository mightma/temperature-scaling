"""Single forward pass per question; read out option-letter logits at the last position."""

from __future__ import annotations

import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import LETTERS
from .prompts import build_prompt


def load_model(name: str, dtype: str = "bfloat16", attn_impl: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(dtype=getattr(torch, dtype), device_map="auto")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except ValueError:
        # Multimodal checkpoints (e.g. Qwen3.5) may only register an image-text-to-text class.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


def letter_token_ids(tokenizer, k: int, variants: str) -> list[list[int]]:
    """Token ids for each option letter. ``"A"`` and ``" A"`` are distinct tokens; with
    ``variants="both"`` their logits are merged with logsumexp (i.e. probabilities summed)."""
    forms = {"space": [" {}"], "nospace": ["{}"], "both": [" {}", "{}"]}[variants]
    ids = []
    for letter in LETTERS[:k]:
        group = []
        for f in forms:
            toks = tokenizer.encode(f.format(letter), add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(f"{f.format(letter)!r} is not a single token for this tokenizer: {toks}")
            if toks[0] not in group:
                group.append(toks[0])
        ids.append(group)
    return ids


@torch.no_grad()
def extract_logits(
    model,
    tokenizer,
    items: list[dict],
    prompt_format: str = "plain",
    variants: str = "both",
    max_batch_tokens: int = 16384,
    max_batch_size: int = 32,
    desc: str = "forward",
) -> np.ndarray:
    """Returns float32 array [N, K_max] of option logits; missing options are -inf."""
    k_max = max(len(it["choices"]) for it in items)
    groups = letter_token_ids(tokenizer, k_max, variants)
    width = max(len(g) for g in groups)
    # [K_max, width] gather index, padded by repeating the group's first id (masked out below)
    gather = torch.tensor([g + [g[0]] * (width - len(g)) for g in groups])
    valid = torch.tensor([[True] * len(g) + [False] * (width - len(g)) for g in groups])

    prompts = [build_prompt(it, tokenizer, prompt_format) for it in items]
    encoded = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    order = sorted(range(len(items)), key=lambda i: -len(encoded[i]))

    # bf16 logits of magnitude ~16-32 are quantised to steps of 0.125, which is coarse for
    # calibration. Instead take the final hidden state and apply only the letter rows of the
    # LM head in fp32.
    decoder = model.get_decoder()
    head = model.get_output_embeddings()
    flat_ids = gather.flatten().to(head.weight.device)
    w = head.weight[flat_ids].float()  # [K_max*width, H]
    b = head.bias[flat_ids].float() if getattr(head, "bias", None) is not None else None
    cfg = getattr(model.config, "text_config", model.config)
    softcap = getattr(cfg, "final_logit_softcapping", None)
    scale = getattr(cfg, "logit_scale", None)  # e.g. Cohere

    out = np.full((len(items), k_max), -np.inf, dtype=np.float32)
    device = model.get_input_embeddings().weight.device
    pbar = tqdm(total=len(items), desc=desc)
    pos = 0
    while pos < len(order):
        longest = len(encoded[order[pos]])
        bs = max(1, min(max_batch_size, max_batch_tokens // longest))
        idx = order[pos : pos + bs]
        batch = tokenizer.pad({"input_ids": [encoded[i] for i in idx]}, return_tensors="pt", padding_side="left")
        h = decoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
        ).last_hidden_state[:, -1, :].to(w.device, torch.float32)
        logits = h @ w.T
        if b is not None:
            logits = logits + b
        if scale is not None:
            logits = logits * scale
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        sel = logits.view(len(idx), k_max, width)
        sel = sel.masked_fill(~valid.to(logits.device), -math.inf)
        z = torch.logsumexp(sel, dim=-1).cpu().numpy()
        for row, i in enumerate(idx):
            k = len(items[i]["choices"])
            out[i, :k] = z[row, :k]
        pos += bs
        pbar.update(len(idx))
    pbar.close()
    return out
