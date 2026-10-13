"""Output contract: strict JSON {root_cause, affected_kpis, explanation, resolution}."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import jsonschema

OUTPUT_KEYS = ("root_cause", "affected_kpis", "explanation", "resolution")


def build_schema(labels: list[str], kpis: list[str]) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(OUTPUT_KEYS),
        "properties": {
            "root_cause": {"type": "string", "enum": list(labels)},
            "affected_kpis": {
                "type": "array",
                "items": {"type": "string", "enum": list(kpis)},
                "uniqueItems": True,
                "minItems": 1,
            },
            "explanation": {"type": "string", "minLength": 20},
            "resolution": {"type": "string", "minLength": 10},
        },
    }


@dataclass
class ParsedOutput:
    raw: str
    obj: dict | None
    json_ok: bool  # a JSON object could be parsed at all
    schema_ok: bool  # it also satisfies the strict schema
    errors: list[str] = field(default_factory=list)
    root_cause: str | None = None  # normalised label (case-insensitive exact match) or None
    affected_kpis: list[str] = field(default_factory=list)
    explanation: str = ""
    resolution: str = ""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _find_json(text: str) -> dict | None:
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    dec = json.JSONDecoder()
    for cand in candidates:
        i = cand.find("{")
        while i != -1:
            try:
                obj, _ = dec.raw_decode(cand[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            i = cand.find("{", i + 1)
    return None


def parse_output(text: str, schema: dict) -> ParsedOutput:
    obj = _find_json(text or "")
    if obj is None:
        return ParsedOutput(text, None, False, False, ["no JSON object found"])
    validator = jsonschema.Draft202012Validator(schema)
    errors = [e.message for e in validator.iter_errors(obj)]
    labels = schema["properties"]["root_cause"]["enum"]
    kpis = schema["properties"]["affected_kpis"]["items"]["enum"]
    lab_map = {lab.lower(): lab for lab in labels}
    kpi_map = {k.lower(): k for k in kpis}
    rc = obj.get("root_cause")
    rc_norm = lab_map.get(str(rc).strip().lower()) if rc is not None else None
    aff = obj.get("affected_kpis") or []
    if isinstance(aff, str):
        aff = [a.strip() for a in aff.split(",")]
    aff_norm = []
    for a in aff if isinstance(aff, list) else []:
        k = kpi_map.get(str(a).strip().lower())
        if k and k not in aff_norm:
            aff_norm.append(k)
    return ParsedOutput(
        raw=text, obj=obj, json_ok=True, schema_ok=not errors, errors=errors, root_cause=rc_norm,
        affected_kpis=aff_norm, explanation=str(obj.get("explanation", "") or ""),
        resolution=str(obj.get("resolution", "") or ""),
    )


def dump_target(obj: dict) -> str:
    """Canonical serialisation used as the SFT target (stable key order, no indentation)."""
    return json.dumps({k: obj[k] for k in OUTPUT_KEYS}, ensure_ascii=False)
