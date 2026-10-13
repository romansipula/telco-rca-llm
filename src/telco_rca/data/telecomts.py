"""Adapter for TelecomTS (huggingface.co/datasets/AliMaatouk/TelecomTS, MIT licence).

Only the anomalous windows are loaded (the anomaly is assumed to be detected upstream). Each of
the 12 anomalous recordings (``anomalous/jammer/<app>`` and ``anomalous/synthetic/<zone>/<app>``)
is one group: windows are cut with stride 32 and length 128, so neighbouring windows of a
recording overlap by 75% and must never be split across train/test.

Reference profile (baseline when a window has too little normal data of its own), in order:

1. ``history``: robust statistics of the same recording's *preceding* ``history_steps`` steps
   (default 3000 = 5 min) that the upstream detector did not flag (flags = union of all labelled
   anomaly segments of that recording). This is the operational analogue of "the cell's recent
   normal behaviour" and uses no root-cause information. Causal: nothing after the window start.
2. ``normal_recording``: the dataset's ``normal/stationary/Zone_X/no_congestion/<app>`` recording
   for the same zone and application (used for the jammer recordings, which are jammed end to end). The dataset's ``description``, ``statistics`` and ``QnA`` fields are NOT used as
model input: the descriptions were written with the anomaly symptoms in the prompt and the
reasoning traces were produced by Qwen3-32B (see docs/DATA.md).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from .base import Case, DatasetAdapter, KPISpec, ReferenceProfile, Ticket, robust_profile

log = logging.getLogger(__name__)

HF_REPO = "AliMaatouk/TelecomTS"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"

RECORDINGS = [
    "anomalous/jammer/File",
    "anomalous/jammer/Twitch",
    "anomalous/jammer/YouTube",
    *[f"anomalous/synthetic/Zone_{z}/{a}" for z in "ABC" for a in ("File", "Twitch", "YouTube")],
]

LABELS = [
    "Antenna Failure",
    "Buffer Overflow (Gradual Buildup)",
    "Co-Channel Interference (Mild)",
    "Co-Channel Interference (Severe)",
    "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)",
    "Faulty RF Filters (Temporal)",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
    "Jamming",
    "Resource Allocation Bugs",
]

KPI_SPECS = [
    KPISpec("RSRP", "dBm", "reference signal received power (downlink)"),
    KPISpec("DL_BLER", "ratio", "downlink block error rate"),
    KPISpec("DL_MCS", "index", "downlink modulation and coding scheme"),
    KPISpec("UL_BLER", "ratio", "uplink block error rate"),
    KPISpec("UL_MCS", "index", "uplink modulation and coding scheme"),
    KPISpec("UL_NPRB", "PRB", "uplink noise/interference per PRB indicator"),
    KPISpec("UL_SNR", "dB", "uplink signal-to-noise ratio"),
    KPISpec("TX_Bytes", "B/100ms", "uplink bytes transmitted"),
    KPISpec("RX_Bytes", "B/100ms", "downlink bytes received"),
    KPISpec("Estimated_UL_Buffer", "B", "estimated uplink buffer occupancy"),
    KPISpec("PRBs_DL_Current", "PRB", "downlink PRBs allocated"),
    KPISpec("PRBs_UL_Current", "PRB", "uplink PRBs allocated"),
    KPISpec("PRB_Utilization_DL", "%", "downlink PRB utilisation"),
    KPISpec("PRB_Utilization_UL", "%", "uplink PRB utilisation"),
    KPISpec("UL_NumberOfPackets", "pkt", "uplink packet count"),
    KPISpec("DL_NumberOfPackets", "pkt", "downlink packet count"),
    KPISpec("UL_Protocol", "", "dominant uplink transport protocol", kind="categorical"),
    KPISpec("DL_Protocol", "", "dominant downlink transport protocol", kind="categorical"),
]

_TICKET_FIELDS = {
    "issue": r"\*\*Issue:\*\*",
    "symptoms": r"\*\*Symptoms:\*\*",
    "root_cause": r"\*\*Root Cause:\*\*",
    "resolution": r"\*\*Resolution:\*\*",
}


def parse_ticket(text: str) -> Ticket:
    """Split a TelecomTS markdown ticket into its four fields."""
    if not text:
        return Ticket()
    positions = []
    for key, pat in _TICKET_FIELDS.items():
        m = re.search(pat, text)
        if m:
            positions.append((m.start(), m.end(), key))
    positions.sort()
    out = {}
    for i, (_, end, key) in enumerate(positions):
        stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        val = text[end:stop]
        val = re.sub(r"\s*-\s*$", "", val.strip())
        out[key] = " ".join(val.replace("\n", " ").split()).strip(" -")
    return Ticket(**out)


def _download(rel_path: str, cache_dir: Path, retries: int = 4) -> Path:
    """Download one dataset file, verifying its length (a silently truncated CSV would change the
    reference profiles and therefore the prompts)."""
    dst = cache_dir / rel_path
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = HF_BASE + rel_path
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        log.info("downloading %s (attempt %d)", url, attempt)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "telco-rca"})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                expected = int(r.headers.get("Content-Length") or -1)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = tmp.stat().st_size
            if expected >= 0 and got != expected:
                raise OSError(f"truncated download: {got} of {expected} bytes")
            tmp.rename(dst)
            return dst
        except OSError as e:
            log.warning("download failed: %s", e)
            time.sleep(2 * attempt)
    raise OSError(f"could not download {url}")


def _cat(v) -> str:
    """Protocol channels: TCP / UDP / none (no packets in that 100 ms step)."""
    if v is None or str(v).strip() in ("", "None", "nan", "null"):
        return "none"
    return str(v)


def _app_folder(app: str) -> str:
    return {"youtube": "YouTube", "twitch": "Twitch", "file": "File"}[app.lower()]


class TelecomTSAdapter(DatasetAdapter):
    name = "telecomts"

    def __init__(
        self,
        cache_dir: str | os.PathLike | None = None,
        use_reference: bool = True,
        history_steps: int = 3000,
        min_history: int = 300,
    ):
        self.cache_dir = Path(cache_dir or os.environ.get("TELCO_RCA_CACHE", "data_cache")) / "telecomts"
        self.use_reference = use_reference
        self.history_steps = history_steps
        self.min_history = min_history
        self._ref_cache: dict[tuple[str, str], ReferenceProfile] = {}
        self._rec_cache: dict[str, tuple[dict, dict, dict[str, int], np.ndarray]] = {}
        self._case_ref: dict[str, ReferenceProfile | None] = {}
        self._windows: dict[str, list[tuple[str, tuple[int, int] | None]]] = {}

    def kpis(self) -> list[KPISpec]:
        return list(KPI_SPECS)

    def label_set(self) -> list[str]:
        return list(LABELS)

    def load_cases(self) -> list[Case]:
        cases: list[Case] = []
        num_names = self.numeric_kpis()
        cat_names = self.categorical_kpis()
        for rec in RECORDINGS:
            path = _download(f"{rec}/processed/chunked.jsonl", self.cache_dir)
            group = rec.removeprefix("anomalous/")
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a = r["anomalies"]
                    if not a.get("exists"):
                        continue
                    k = r["KPIs"]
                    numeric = {n: np.asarray(k[n], dtype=float) for n in num_names}
                    categorical = {n: [_cat(v) for v in k[n]] for n in cat_names}
                    dur = a.get("anomaly_duration") or {}
                    seg = None
                    if dur.get("start") is not None and dur.get("end") is not None:
                        seg = (int(dur["start"]), int(dur["end"]))
                    self._windows.setdefault(group, []).append((r["start_time"], seg))
                    labels = r.get("labels", {})
                    cases.append(
                        Case(
                            case_id=f"{group}@{r['start_time']}",
                            group_id=group,
                            numeric=numeric,
                            categorical=categorical,
                            sampling_seconds=1.0 / float(r.get("sampling_rate", 10)),
                            segment=seg,
                            context={
                                "zone": labels.get("zone"),
                                "application": labels.get("application"),
                                "mobility": labels.get("mobility"),
                                "start_time": r["start_time"],
                            },
                            root_cause=a["type"],
                            affected_kpis=[x for x in a.get("affected_kpis", []) if x],
                            ticket=parse_ticket(a.get("troubleshooting_tickets", "")),
                            source=self.name,
                        )
                    )
        cases.sort(key=lambda c: c.case_id)
        return cases

    def _read_csv(self, rel: str) -> tuple[dict, dict, list[str]]:
        path = _download(rel, self.cache_dir)
        num_names, cat_names = self.numeric_kpis(), self.categorical_kpis()
        cols: dict[str, list] = {n: [] for n in num_names + cat_names}
        stamps: list[str] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                stamps.append(row.get("timestamp", ""))
                for n in num_names:
                    v = row.get(n, "")
                    cols[n].append(float(v) if v not in ("", None) else np.nan)
                for n in cat_names:
                    cols[n].append(_cat(row.get(n)))
        num = {n: np.asarray(cols[n], dtype=float) for n in num_names}
        cat = {n: cols[n] for n in cat_names}
        return num, cat, stamps

    def _recording(self, group: str):
        if group not in self._rec_cache:
            num, cat, stamps = self._read_csv(f"anomalous/{group}/raw/metrics.csv")
            index = {t: i for i, t in enumerate(stamps)}
            flagged = np.zeros(len(stamps), dtype=bool)
            if group not in self._windows:
                self.load_cases()
            for start, seg in self._windows.get(group, []):
                i = index.get(start)
                if i is None:
                    continue
                if seg is None:
                    flagged[i : i + 128] = True
                else:
                    flagged[i + seg[0] : i + seg[1] + 1] = True
            self._rec_cache[group] = (num, cat, index, flagged)
        return self._rec_cache[group]

    def history_profile(self, case: Case) -> ReferenceProfile | None:
        num, cat, index, flagged = self._recording(case.group_id)
        i = index.get(case.context.get("start_time"))
        if i is None:
            return None
        lo = max(0, i - self.history_steps)
        keep = ~flagged[lo:i]
        if keep.sum() < self.min_history:
            return None
        return robust_profile(
            {k: v[lo:i][keep] for k, v in num.items()},
            {k: [x for x, m in zip(v[lo:i], keep) if m] for k, v in cat.items()},
        )

    def reference_profile(self, case: Case) -> ReferenceProfile | None:
        if not self.use_reference:
            return None
        if case.case_id not in self._case_ref:
            prof = self.history_profile(case)
            if prof is not None:
                prof.source = "history"
            else:
                prof = self.normal_recording_profile(case)
            self._case_ref[case.case_id] = prof
        return self._case_ref[case.case_id]

    def normal_recording_profile(self, case: Case) -> ReferenceProfile | None:
        zone, app = case.context.get("zone"), case.context.get("application")
        if not zone or not app:
            return None
        key = (zone, app)
        if key not in self._ref_cache:
            rel = f"normal/stationary/Zone_{zone}/no_congestion/{_app_folder(app)}/raw/metrics.csv"
            num, cat, _ = self._read_csv(rel)
            prof = robust_profile(num, cat)
            prof.source = "normal_recording"
            self._ref_cache[key] = prof
        return self._ref_cache[key]
