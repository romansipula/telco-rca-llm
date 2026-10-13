"""Chat-format examples for SFT and inference."""
from __future__ import annotations

from ..prompts import user_prompt_evidence, user_prompt_hybrid, user_prompt_raw
from ..schema import dump_target


def build_examples(rows: list[dict], meta: dict, mode: str, split: str, gb_probs: list[dict] | None = None
                   ) -> list[dict]:
    probs = {p["case_id"]: p for p in gb_probs or []}
    if mode == "hybrid" and not probs:
        raise ValueError("hybrid mode needs --gb-probs (from `telco-rca baseline`)")
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        if mode == "evidence":
            user = user_prompt_evidence(r["evidence_text"])
        elif mode == "raw":
            user = user_prompt_raw(r["raw_text"])
        elif mode == "hybrid":
            p = probs[r["case_id"]]
            if split == "train" and p["prob_source"] != "oof":
                raise ValueError("training rows must use out-of-fold probabilities")
            user = user_prompt_hybrid(r["evidence_text"], [tuple(x) for x in p["top3"]])
        else:
            raise ValueError(mode)
        msgs = [{"role": "system", "content": meta["system_prompt"]}, {"role": "user", "content": user}]
        ex = {"case_id": r["case_id"], "mode": mode, "messages": msgs}
        if r.get("target"):
            ex["target"] = dump_target(r["target"])
        out.append(ex)
    return out


def render_prompt(tokenizer, messages: list[dict], cfg: dict) -> str:
    """Apply the model's chat template; fold the system prompt into the user turn if needed."""
    kwargs = dict(cfg.get("chat_template_kwargs") or {})
    msgs = messages
    if cfg.get("system_in_user"):
        msgs = _fold_system(messages)
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:  # template without system role support
        return tokenizer.apply_chat_template(_fold_system(messages), tokenize=False, add_generation_prompt=True,
                                             **kwargs)


def render_full(tokenizer, messages: list[dict], target: str, cfg: dict) -> tuple[str, str]:
    """(prompt, prompt+target+end-of-turn) as text, guaranteeing the prompt is a prefix."""
    prompt = render_prompt(tokenizer, messages, cfg)
    end = cfg.get("end_of_turn") or end_of_turn(tokenizer, cfg)
    return prompt, prompt + target + end


def end_of_turn(tokenizer, cfg: dict) -> str:
    """Text the chat template appends after an assistant message (e.g. '<turn|>' or '</s>')."""
    probe = "XQZPROBE"
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": probe}]
    kwargs = dict(cfg.get("chat_template_kwargs") or {})
    try:
        full = tokenizer.apply_chat_template(msgs, tokenize=False, **kwargs)
        tail = full[full.rindex(probe) + len(probe):]
        tail = tail.rstrip("\n")
        if tail:
            return tail
    except Exception:
        pass
    return tokenizer.eos_token or ""


def _fold_system(messages: list[dict]) -> list[dict]:
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_txt = messages[0]["content"]
    rest = [dict(m) for m in messages[1:]]
    rest[0]["content"] = sys_txt + "\n\n" + rest[0]["content"]
    return rest
