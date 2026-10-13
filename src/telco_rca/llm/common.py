"""Model / tokenizer loading shared by training, inference and export.

Model YAML keys (see configs/models/*.yaml):
  hf_id               Hugging Face repo id
  loader              auto | causal_lm | image_text_to_text
  compute_dtype       float16 | float32   (T4: no bf16; float32 only as an overflow fallback)
  lora                {r, alpha, dropout, target_modules (regex)}
  chat_template_kwargs, system_in_user, end_of_turn
"""
from __future__ import annotations

import logging
import math
import os

log = logging.getLogger(__name__)


def torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32,
            "bfloat16": torch.bfloat16}[name]


_TINY: dict = {}

TINY_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _tiny_tokenizer():
    """Byte-level BPE trained on a fixed corpus: deterministic, offline."""
    if "tok" not in _TINY:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast

        tk = Tokenizer(models.BPE())
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        corpus = ['{"root_cause": "Interference", "affected_kpis": ["SINR"], "explanation": "x", "resolution": "y"}',
                  "EVIDENCE DEVIATING KPIs z= up down step ramp baseline window segment steps"] * 50
        trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<eos>", "<|system|>", "<|user|>",
                                                                      "<|assistant|>", "<|end|>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tk.train_from_iterator(corpus, trainer)
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>")
        tok.chat_template = TINY_TEMPLATE
        _TINY["tok"] = tok
    return _TINY["tok"]


def _tiny_model(tok):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    conf = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return LlamaForCausalLM(conf)


def load_tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    if cfg["hf_id"] == "__tiny__":
        return _tiny_tokenizer()
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer_id", cfg["hf_id"]), trust_remote_code=False,
                                        **(cfg.get("tokenizer_kwargs") or {}))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _auto_class(cfg: dict):
    import transformers

    loader = cfg.get("loader", "auto")
    if loader == "causal_lm":
        return transformers.AutoModelForCausalLM
    if loader == "image_text_to_text":
        return transformers.AutoModelForImageTextToText
    return transformers.AutoModelForCausalLM


def load_model(cfg: dict, quant: str = "4bit", compute_dtype: str | None = None, device_map=None):
    """quant: '4bit' (QLoRA / NF4) or 'none' (plain fp16/fp32 weights)."""
    import torch

    if cfg["hf_id"] == "__tiny__":
        return _tiny_model(_tiny_tokenizer())
    dt = torch_dtype(compute_dtype or cfg.get("compute_dtype", "float16"))
    kwargs = {"dtype": dt}
    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    kwargs["device_map"] = device_map
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dt,
            llm_int8_skip_modules=cfg.get("skip_quant_modules"),
        )
    if cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = cfg["attn_implementation"]
    cls = _auto_class(cfg)
    log.info("loading %s with %s quant=%s dtype=%s", cfg["hf_id"], cls.__name__, quant, dt)
    model = cls.from_pretrained(cfg["hf_id"], **kwargs)
    model.config.use_cache = False
    return model


def offload_to_cpu(model, names: list[str]) -> list[str]:
    """Move frozen lookup modules (e.g. Gemma 4's 2.8B-parameter per-layer embedding table) to host
    RAM after loading, with hooks that move their inputs to CPU and outputs back to the GPU.

    Done after loading rather than through ``device_map`` because accelerate refuses to train a
    4-bit model whose device map contains "cpu"."""
    import torch

    moved = []
    for name in names:
        try:
            mod = model.get_submodule(name)
        except AttributeError:
            log.warning("cpu_modules: %s not found", name)
            continue
        target = next(p.device for n, p in model.named_parameters() if not n.startswith(name))
        mod.to("cpu")
        for p in mod.parameters():
            p.requires_grad_(False)

        def pre(m, args, kwargs):
            mv = lambda t: t.to("cpu") if torch.is_tensor(t) else t  # noqa: E731
            return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}

        def post(m, args, out, _t=target):
            return out.to(_t) if torch.is_tensor(out) else out

        mod.register_forward_pre_hook(pre, with_kwargs=True)
        mod.register_forward_hook(post)
        moved.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("offloaded to CPU: %s", moved)
    return moved


def finite_probe(model, tokenizer, texts: list[str], max_len: int = 1024) -> dict:
    """Forward a few real examples and check the logits/loss are finite (fp16 overflow check)."""
    import torch

    model.eval()
    res = {"finite": True, "losses": []}
    dev = next(model.parameters()).device
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False)
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            loss = float(out.loss)
            amax = float(out.logits.float().abs().max())
            res["losses"].append(loss)
            if not math.isfinite(loss) or not math.isfinite(amax):
                res["finite"] = False
    return res


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        return {"cuda": True, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def peak_gpu_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**20
    except Exception:
        pass
    return None
