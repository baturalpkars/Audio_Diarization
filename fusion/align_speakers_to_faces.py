from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype


def load_segments(segments_json: Path) -> List[dict]:
    data = json.loads(segments_json.read_text())
    # expected: [{"start":..., "end":..., "speaker":...}, ...]
    return data


def load_participants(participants_dir: Path) -> Dict[str, pd.DataFrame]:
    files = sorted(participants_dir.glob("participant_*.csv"))
    if not files:
        raise FileNotFoundError(f"No participant_*.csv found in {participants_dir}")

    participants = {}
    for f in files:
        df = pd.read_csv(f)
        required = {"time_sec", "mouth_score_raw", "mouth_open"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f.name} missing columns: {missing}")

        df = df.sort_values("time_sec").reset_index(drop=True)
        participants[f.stem] = df
    return participants


def smooth_series(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, w, mode="same")


def build_mouth_lookup(df: pd.DataFrame, smooth_ms: float = 240.0) -> tuple[
    Any, ndarray, ndarray[tuple[int, ...], dtype[Any]], dict[str, float | int]]:
    """
    Build a smoothed + normalized mouth activity signal from FaceReader.
    Returns (t, mouth_norm, debug_info)
    """

    t = df["time_sec"].to_numpy(dtype=np.float32)
    mouth = df["mouth_score_raw"].to_numpy(dtype=np.float32)

    dt = float(np.median(np.diff(t))) if len(t) >= 2 else 0.04
    if dt <= 0:
        dt = 0.04

    window = max(1, int(round((smooth_ms / 1000.0) / dt)))
    mouth_s = smooth_series(mouth, window=window)

    # robust per-participant scaling
    p10 = float(np.percentile(mouth_s, 10))
    p90 = float(np.percentile(mouth_s, 90))
    denom = (p90 - p10) if (p90 - p10) > 1e-6 else 1.0
    mouth_norm = np.clip((mouth_s - p10) / denom, 0.0, 1.0)

    dbg = {
        "dt": dt,
        "window": window,
        "p10": p10,
        "p90": p90,
        "mean_raw": float(mouth.mean()),
        "mean_smooth": float(mouth_s.mean()),
        "mean_norm": float(mouth_norm.mean()),
    }
    return t, mouth_s, mouth_norm, dbg


def mouth_speed(mouth_s: np.ndarray, dt: float) -> np.ndarray:
    d = np.abs(np.diff(mouth_s, prepend=mouth_s[0])) / max(dt, 1e-6)
    return d.astype(np.float32)


def build_open_lookup(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df["time_sec"].to_numpy(dtype=np.float32)
    o = df["mouth_open"].to_numpy(dtype=np.float32)
    return t, o


def oscillation_rate(t: np.ndarray, o: np.ndarray, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    if i1 - i0 <= 2:
        return 0.0
    seg = o[i0:i1]
    flips = np.abs(np.diff(seg)).sum()  # since seg is 0/1, diff is -1/0/1
    dur = float(end - start)
    return float(flips / max(dur, 1e-6))


def mean_std_in_interval(t: np.ndarray, x: np.ndarray, start: float, end: float) -> tuple[float, float]:
    if end <= start:
        return 0.0, 0.0
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    seg = x[i0:i1]
    if len(seg) < 3:
        return 0.0, 0.0
    return float(seg.mean()), float(seg.std())


def softmax_probs(vals: np.ndarray, temp: float) -> np.ndarray:
    # stable softmax
    v = vals.astype(np.float64)
    v = v - np.max(v)
    x = np.exp(v / max(temp, 1e-6))
    return (x / (np.sum(x) + 1e-12)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=str, required=True)
    ap.add_argument("--participants-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--min-seg-sec", type=float, default=0.4)
    ap.add_argument("--av-offset-sec", type=float, default=0.0)
    ap.add_argument("--smooth-ms", type=float, default=240.0)

    # === NEW: tuning for normalized scores ===
    ap.add_argument("--margin-ratio", type=float, default=1.12,
                    help="Ambiguous if best < margin_ratio * top2. For normalized scores use ~1.08–1.15.")
    ap.add_argument("--best-min", type=float, default=0.06,
                    help="Ambiguous if best score < best_min (prevents confident decisions on tiny scores).")
    ap.add_argument("--stick-ratio", type=float, default=0.92,
                    help="If ambiguous, keep previous only if prev_val >= stick_ratio * best_val.")
    ap.add_argument("--use-global-map", action="store_true",
                    help="If set, final assignment uses global speaker->participant mapping, with per-seg override.")
    ap.add_argument("--ambiguous-global-max-sec", type=float, default=-1.0,
                    help="If >0, use fixed threshold. If <=0, use ratio-based threshold.")
    ap.add_argument("--ambiguous-global-max-ratio", type=float, default=0.05,
                    help="Ratio of total meeting duration used when max-sec <=0 (clamped to 5–30s).")
    ap.add_argument("--ambiguous-local-margin", type=float, default=1.01,
                    help="If ambiguous and best/top2 ratio <= this, force local best (super-ambiguous).")
    ap.add_argument("--softmax-temp", type=float, default=0.08,
                    help="Softmax temperature for per-segment normalization. Smaller => sharper winners (0.05–0.15).")

    args = ap.parse_args()

    segments = load_segments(Path(args.segments).resolve())
    participants = load_participants(Path(args.participants_dir).resolve())

    # Adaptive ambiguous threshold (useful for varying meeting lengths).
    if args.ambiguous_global_max_sec <= 0:
        if segments:
            t_min = min(float(s["start"]) for s in segments)
            t_max = max(float(s["end"]) for s in segments)
            total_dur = max(0.0, t_max - t_min)
        else:
            total_dur = 0.0
        adaptive = total_dur * args.ambiguous_global_max_ratio
        args.ambiguous_global_max_sec = max(5.0, min(30.0, adaptive))

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    speakers = sorted({s["speaker"] for s in segments})
    part_ids = sorted(participants.keys())
    speaker_to_i = {s: i for i, s in enumerate(speakers)}
    part_to_j = {p: j for j, p in enumerate(part_ids)}

    # Build lookups
    score_lookup = {}
    open_lookup = {}
    debug_rows = []

    for pid in part_ids:
        t, mouth_s, mouth_norm, dbg = build_mouth_lookup(participants[pid], smooth_ms=args.smooth_ms)

        spd = mouth_speed(mouth_norm, dbg["dt"])
        p95 = float(np.percentile(spd, 95))
        spd = spd / (p95 + 1e-6)
        spd = np.clip(spd, 0.0, 1.0)

        t_o, o_sig = build_open_lookup(participants[pid])

        score_lookup[pid] = (t, spd)  # use speed only (your current approach)
        open_lookup[pid] = (t_o, o_sig)

        debug_rows.append({"participant": pid, **dbg, "spd_p95": p95, "spd_mean": float(spd.mean())})

    pd.DataFrame(debug_rows).to_csv(out_dir / "participant_debug.csv", index=False)

    # ========= PASS 1: build per-segment scores + GLOBAL aggregation =========
    global_score = np.zeros((len(speakers), len(part_ids)), dtype=np.float32)
    segment_debug_rows = []

    # We’ll also store per-segment best/top2 AND best/top2 values for later decisions.
    # key = (start,end,speaker) -> dict with debug
    seg_key_to_debug = {}

    for seg in segments:
        start = float(seg["start"]) + args.av_offset_sec
        end = float(seg["end"]) + args.av_offset_sec
        spk = seg["speaker"]
        dur = end - start
        if dur < args.min_seg_sec:
            continue

        si = speaker_to_i[spk]

        per_p = {}
        for pid in part_ids:
            t_spd, spd = score_lookup[pid]
            mu_spd, sd_spd = mean_std_in_interval(t_spd, spd, start, end)
            S = mu_spd + 0.5 * sd_spd

            t_o, o_sig = open_lookup[pid]
            O, _ = mean_std_in_interval(t_o, o_sig, start, end)
            F = oscillation_rate(t_o, o_sig, start, end)

            combined = S * (0.85 + 0.15 * O) + 0.02 * min(F, 2.0)
            per_p[pid] = float(combined)

        sorted_p = sorted(per_p.items(), key=lambda kv: kv[1], reverse=True)
        best_pid, best_val = sorted_p[0]
        top2_pid, top2_val = sorted_p[1] if len(sorted_p) > 1 else (None, None)
        top3_pid, top3_val = sorted_p[2] if len(sorted_p) > 2 else (None, None)

        best_over_top2 = (best_val / (top2_val + 1e-9)) if top2_val is not None else None

        # NEW ambiguity: ratio + absolute min
        ambiguous = False
        if best_val < args.best_min:
            ambiguous = True
        elif top2_val is not None and best_val < args.margin_ratio * top2_val:
            ambiguous = True

        # accumulate to global matrix
        vals = np.array([per_p[pid] for pid in part_ids], dtype=np.float32)
        probs = softmax_probs(vals, args.softmax_temp)

        # Option A (recommended): accumulate log-probs (MAP-like)
        global_score[si] += dur * np.log(probs + 1e-12)

        # Option B: accumulate probs (simpler)
        # global_score[si] += dur * probs

        row = {
            "start": start,
            "end": end,
            "dur": dur,
            "speaker": spk,
            "best_participant": best_pid,
            "best_combined": best_val,
            "top2_participant": top2_pid,
            "top2_combined": top2_val,
            "top3_participant": top3_pid,
            "top3_combined": top3_val,
            "ambiguous": ambiguous,
            "best_over_top2": best_over_top2,
        }
        segment_debug_rows.append(row)
        seg_key_to_debug[(start, end, spk)] = row

    # Save debug + global score matrix
    pd.DataFrame(segment_debug_rows).to_csv(out_dir / "segment_vvad_debug.csv", index=False)
    pd.DataFrame(global_score, index=speakers, columns=part_ids).to_csv(out_dir / "score_matrix.csv")

    # ========= PASS 2: GLOBAL speaker->participant mapping =========
    global_map = {}
    for si, spk in enumerate(speakers):
        pj = int(np.argmax(global_score[si]))
        global_map[spk] = part_ids[pj]

    (out_dir / "speaker_to_participant.json").write_text(json.dumps(global_map, indent=2))

    # ========= PASS 3: FINAL alignment =========
    # default policy:
    # - if --use-global-map: start from global_map[spk] and override only if segment is confident for another pid
    # - else: use best_pid from segment
    #
    # ambiguity handling:
    # - if ambiguous: keep prev only if prev is still close to best (prev_val >= stick_ratio * best_val)

    prev_pid = None
    aligned = []

    for seg in segments:
        start = float(seg["start"]) + args.av_offset_sec
        end = float(seg["end"]) + args.av_offset_sec
        spk = seg["speaker"]
        dur = end - start
        if dur < args.min_seg_sec:
            continue

        row = seg_key_to_debug.get((start, end, spk))
        if row is None:
            aligned.append({
                "start": start,
                "end": end,
                "speaker": spk,
                "participant": prev_pid,
                "ambiguous": True,
                "reason": "missing_debug_row",
            })
            continue

        best_pid = row["best_participant"]
        best_val = float(row["best_combined"])
        top2_pid = row["top2_participant"]
        top2_val = float(row["top2_combined"]) if row["top2_combined"] is not None else None
        ambiguous = bool(row["ambiguous"])
        best_over_top2 = row.get("best_over_top2")
        if best_over_top2 is not None:
            try:
                best_over_top2 = float(best_over_top2)
            except Exception:
                best_over_top2 = None

        # choose base candidate
        if args.use_global_map:
            # If ambiguous and super-close, always use local best.
            super_ambiguous = (
                ambiguous
                and (best_over_top2 is not None)
                and (best_over_top2 <= args.ambiguous_local_margin)
            )

            # If ambiguous and long, prefer local best to avoid global lock-in.
            if super_ambiguous or (ambiguous and dur > args.ambiguous_global_max_sec):
                chosen = best_pid
            else:
                chosen = global_map.get(spk, best_pid)

                # override only if REALLY strong
                override_ratio = 1.6  # start 1.4–2.0
                if (top2_val is not None) and (best_val >= override_ratio * top2_val):
                    chosen = best_pid
        else:
            chosen = best_pid

        # sticky ambiguity fix: only keep prev if prev still plausible in THIS segment
        if ambiguous and prev_pid is not None:
            # we need prev_val; compute quickly by re-using top info:
            # if prev == best: ok; if prev == top2: use top2_val; else treat as 0
            if prev_pid == best_pid:
                prev_val = best_val
            elif prev_pid == top2_pid and top2_val is not None:
                prev_val = top2_val
            else:
                prev_val = 0.0

            if prev_val >= args.stick_ratio * best_val:
                chosen = prev_pid

        # update prev
        prev_pid = chosen

        aligned.append({
            "start": start,
            "end": end,
            "speaker": spk,
            "participant": chosen,
            "ambiguous": ambiguous,
            "best_pid": best_pid,
            "best_val": best_val,
            "top2_pid": top2_pid,
            "top2_val": top2_val,
            "global_pid": global_map.get(spk),
        })

    (out_dir / "aligned_segments.json").write_text(json.dumps(aligned, indent=2))

    print("Saved:")
    print(" - participant_debug.csv")
    print(" - segment_vvad_debug.csv")
    print(" - score_matrix.csv  (GLOBAL speaker x participant)")
    print(" - speaker_to_participant.json  (GLOBAL mapping)")
    print(" - aligned_segments.json")


if __name__ == "__main__":
    main()
