from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


TIME_RE = re.compile(r"^(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)\.(?P<ms>\d+)$")


def time_to_seconds(t: str) -> float:
    if not isinstance(t, str):
        t = str(t)
    t = t.strip()
    m = TIME_RE.match(t)
    if not m:
        raise ValueError(f"Unexpected time format: {t!r}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    ms = int(m.group("ms"))
    return h * 3600 + mi * 60 + s + ms / 1000.0


def mouth_open_to_int(x) -> int:
    if not isinstance(x, str):
        return 0
    x = x.strip().lower()
    return 1 if x == "open" else 0


def pick_col(columns: list[str], candidates: list[str]) -> str:
    colset = {c.strip(): c for c in columns}
    for cand in candidates:
        if cand in colset:
            return colset[cand]
    # fallback: case-insensitive contains
    lower = [(c, c.lower()) for c in columns]
    for cand in candidates:
        cand_l = cand.lower()
        for orig, low in lower:
            if cand_l == low:
                return orig
    raise KeyError(f"Could not find any of these columns: {candidates}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True, help="FaceReader detailed export .xlsx path")
    ap.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first sheet)")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--quality-threshold", type=float, default=0.7, help="Drop frames below this Quality")
    ap.add_argument("--keep-all", action="store_true", help="If set, do not filter by Quality")
    ap.add_argument("--min-block-rows", type=int, default=200, help="Drop blocks smaller than this")
    return ap.parse_args()


def split_by_time_resets(df: pd.DataFrame, time_col: str = "time_sec") -> list[pd.DataFrame]:
    t = df[time_col].to_numpy()
    reset_idx = np.where(t[1:] < t[:-1])[0] + 1
    starts = [0] + reset_idx.tolist()
    ends = reset_idx.tolist() + [len(df)]
    return [df.iloc[s:e].copy() for s, e in zip(starts, ends)]


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.xlsx).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(xlsx_path)
    sheet = args.sheet or xl.sheet_names[0]

    # Read first to inspect columns
    preview = pd.read_excel(xlsx_path, sheet_name=sheet, nrows=1)
    cols = list(preview.columns)

    time_col = pick_col(cols, ["Video Time", "Time"])
    quality_col = pick_col(cols, ["Quality"])
    mouth_col = pick_col(cols, ["Mouth"])
    au10_col = pick_col(cols, ["Action Unit 10 - Upper Lip Raiser", "Action Unit 10", "AU10", "Upper Lip Raiser"])
    au12_col = pick_col(cols, ["Action Unit 12 - Lip Corner Puller", "Action Unit 12", "AU12", "Lip Corner Puller"])
    au25_col = pick_col(cols, ["Action Unit 25 - Lips Part", "Action Unit 25", "AU25", "Lips Part"])
    au26_col = pick_col(cols, ["Action Unit 26 - Jaw Drop", "Action Unit 26", "AU26", "Jaw Drop"])

    usecols = [time_col, quality_col, mouth_col, au10_col, au12_col, au25_col, au26_col]
    df = pd.read_excel(xlsx_path, sheet_name=sheet, usecols=usecols)

    df = df.dropna(subset=[time_col])
    df["time_sec"] = df[time_col].astype(str).map(time_to_seconds)

    df["mouth_open"] = df[mouth_col].map(mouth_open_to_int)
    df["au10"] = pd.to_numeric(df[au10_col], errors="coerce").fillna(0.0)
    df["au12"] = pd.to_numeric(df[au12_col], errors="coerce").fillna(0.0)
    df["au25"] = pd.to_numeric(df[au25_col], errors="coerce").fillna(0.0)
    df["au26"] = pd.to_numeric(df[au26_col], errors="coerce").fillna(0.0)
    df["quality"] = pd.to_numeric(df[quality_col], errors="coerce").fillna(0.0)

    if not args.keep_all:
        df = df[df["quality"] >= args.quality_threshold].copy()

    df["mouth_score_raw"] = 0.45 * df["au25"] + 0.35 * df["au26"] + 0.15 * df["au10"] + 0.05 * df["au12"]

    blocks = split_by_time_resets(df, "time_sec")

    # Drop tiny blocks
    blocks = [b for b in blocks if len(b) >= args.min_block_rows]

    summary = []
    for i, b in enumerate(blocks, start=1):
        pid = f"participant_{i:02d}"
        out_path = out_dir / f"{pid}.csv"

        out_b = b[["time_sec", "quality", "mouth_open", "au10", "au12", "au25", "au26", "mouth_score_raw"]].copy()
        out_b.to_csv(out_path, index=False)

        summary.append(
            {
                "participant_block": pid,
                "n_rows": int(len(out_b)),
                "t_min": float(out_b["time_sec"].min()),
                "t_max": float(out_b["time_sec"].max()),
                "mean_quality": float(out_b["quality"].mean()),
            }
        )

    pd.DataFrame(summary).to_csv(out_dir / "participants_summary.csv", index=False)

    print(f"Sheet: {sheet}")
    print(f"Detected participant blocks: {len(blocks)}")
    print("Wrote:", out_dir / "participants_summary.csv")
    print("Wrote participant CSVs to:", out_dir)
