from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


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


def build_mouth_lookup(df: pd.DataFrame, smooth_ms: float = 240.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a smoothed + normalized mouth activity signal from FaceReader.
    Returns (t, mouth_norm, debug_info)
    """

    t = df["time_sec"].to_numpy(dtype=np.float32)
    mouth = df["mouth_score_raw"].to_numpy(dtype=np.float32)

    # Estimate dt
    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            dt = 0.04
    else:
        dt = 0.04

    window = max(1, int(round((smooth_ms / 1000.0) / dt)))
    mouth_s = smooth_series(mouth, window=window)

    dbg = {
        "dt": dt,
        "window": window,
        "mean_raw": float(mouth.mean()),
    }
    return t, mouth_s, dbg


def build_open_lookup(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = df["time_sec"].to_numpy(dtype=np.float32)
    o = df["mouth_open"].to_numpy(dtype=np.float32)
    return t, o


def oscillation_rate(t: np.ndarray, o: np.ndarray, start: float, end: float) -> float:
    """
    Measures how often mouth_open flips within the interval.
    Returns flips per second (approx).
    """
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


def mean_mouth_in_interval(t: np.ndarray, mouth: np.ndarray, start: float, end: float) -> float:
    # select indices in [start, end]
    if end <= start:
        return 0.0
    i0 = np.searchsorted(t, start, side="left")
    i1 = np.searchsorted(t, end, side="right")
    if i1 <= i0:
        return 0.0
    return float(mouth[i0:i1].mean())


def solve_assignment(score: np.ndarray, speakers: List[str], participants: List[str]) -> Dict[str, str]:
    """
    score[speaker_i, participant_j] higher is better.
    Returns mapping speaker -> participant
    """
    if HAS_SCIPY:
        # Hungarian solves a MIN cost assignment; we maximize score -> minimize negative score
        cost = -score
        r, c = linear_sum_assignment(cost)
        return {speakers[i]: participants[j] for i, j in zip(r, c)}

    # Greedy fallback
    mapping = {}
    used_p = set()
    for si in np.argsort(score.max(axis=1))[::-1]:
        pj = int(np.argmax(score[si]))
        while pj in used_p:
            # if taken, pick next best
            order = np.argsort(score[si])[::-1]
            pj = next((int(k) for k in order if int(k) not in used_p), None)
            if pj is None:
                break
        if pj is not None:
            mapping[speakers[si]] = participants[pj]
            used_p.add(pj)
    return mapping


def solve_many_to_one(score: np.ndarray, speakers: list[str], participants: list[str]) -> dict[str, str]:
    """
    Many-to-one assignment: each speaker gets assigned to the participant with highest score.
    :param score: scores of the mouth activity and duration
    :param speakers: speaker list
    :param participants: participant list (faces)
    :return: A mapping speaker -> participant
    """
    mapping = {}
    for si, spk in enumerate(speakers):
        pj = int(np.argmax(score[si]))
        mapping[spk] = participants[pj]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=str, required=True, help="NeMo segments.json (start/end/speaker)")
    ap.add_argument("--participants-dir", type=str, required=True, help="Folder containing participant_*.csv")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--min-seg-sec", type=float, default=0.4, help="Ignore diarization segments shorter than this")
    ap.add_argument("--av-offset-sec", type=float, default=0.0,
                    help="Shift diarization times by this offset (audio->video). "
                         "If audio is delayed vs video, try +0.1 to +0.3")
    ap.add_argument("--smooth-ms", type=float, default=240.0, help="Smoothing window for mouth activity")
    args = ap.parse_args()

    segments_path = Path(args.segments).resolve()
    participants_dir = Path(args.participants_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = load_segments(segments_path)
    participants = load_participants(participants_dir)

    # Unique speaker IDs
    speakers = sorted({s["speaker"] for s in segments})
    part_ids = sorted(participants.keys())
    speaker_to_i = {s: i for i, s in enumerate(speakers)}

    # Build mouth lookups
    score_lookup = {}
    open_lookup = {}
    debug_rows = []

    for pid in part_ids:
        t_s, s_sig, dbg = build_mouth_lookup(participants[pid], smooth_ms=args.smooth_ms)
        t_o, o_sig = build_open_lookup(participants[pid])

        score_lookup[pid] = (t_s, s_sig)
        open_lookup[pid] = (t_o, o_sig)

        debug_rows.append({"participant": pid, **dbg})

    # Write debug CSV
    pd.DataFrame(debug_rows).to_csv(out_dir / "participant_debug.csv", index=False)
    print("Wrote:", out_dir / "participant_debug.csv")

    score = np.zeros((len(speakers), len(part_ids)), dtype=np.float32)

    # Accumulate mouth-energy per segment
    for seg in segments:
        spk = seg["speaker"]
        start = float(seg["start"]) + args.av_offset_sec
        end = float(seg["end"]) + args.av_offset_sec
        dur = end - start
        if dur < args.min_seg_sec:
            continue

        si = speaker_to_i[spk]

        # “winner-takes-most” style: add duration * mean mouth activity
        for pj, pid in enumerate(part_ids):
            t_s, s_sig = score_lookup[pid]
            S = mean_mouth_in_interval(t_s, s_sig, start, end)

            t_o, o_sig = open_lookup[pid]
            O = mean_mouth_in_interval(t_o, o_sig, start, end)  # fraction of frames open
            F = oscillation_rate(t_o, o_sig, start, end)  # flips per second

            # Soft consistency multiplier: never kills score
            # (tune weights later)
            combined = S * (0.7 + 0.3 * O) + 0.05 * min(F, 2.0)  # cap flips contribution

            score[si, pj] += float(dur * combined)

    # mapping = solve_assignment(score, speakers, part_ids)
    mapping = solve_many_to_one(score, speakers, part_ids)

    # Save mapping
    (out_dir / "speaker_to_participant.json").write_text(json.dumps(mapping, indent=2))

    # For Debugging:
    segment_debug = []

    for seg in segments:
        start = float(seg["start"]) + args.av_offset_sec
        end = float(seg["end"]) + args.av_offset_sec
        dur = end - start
        if dur < args.min_seg_sec:
            continue

        per_p = {}
        for pid in part_ids:
            t_s, s_sig = score_lookup[pid]
            S = mean_mouth_in_interval(t_s, s_sig, start, end)

            t_o, o_sig = open_lookup[pid]
            O = mean_mouth_in_interval(t_o, o_sig, start, end)
            F = oscillation_rate(t_o, o_sig, start, end)

            combined = S * (0.7 + 0.3 * O) + 0.05 * min(F, 2.0)
            per_p[pid] = combined

        sorted_p = sorted(per_p.items(), key=lambda kv: kv[1], reverse=True)
        top3 = sorted_p[:3]

        best_pid, best_val = top3[0]
        top2_pid, top2_val = top3[1] if len(top3) > 1 else (None, None)
        top3_pid, top3_val = top3[2] if len(top3) > 2 else (None, None)

        segment_debug.append({
            "start": start,
            "end": end,
            "dur": dur,
            "speaker": seg["speaker"],
            "best_participant": best_pid,
            "best_combined": best_val,
            "top2_participant": top2_pid,
            "top2_combined": top2_val,
            "top3_participant": top3_pid,
            "top3_combined": top3_val,
        })

    pd.DataFrame(segment_debug).to_csv(out_dir / "segment_vvad_debug.csv", index=False)
    print("Wrote:", out_dir / "segment_vvad_debug.csv")

    # Save aligned segments
    aligned = []
    for seg in segments:
        spk = seg["speaker"]
        aligned.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": spk,
            "participant": mapping.get(spk, None)
        })
    (out_dir / "aligned_segments.json").write_text(json.dumps(aligned, indent=2))

    # Save score matrix
    score_df = pd.DataFrame(score, index=speakers, columns=part_ids)
    score_df.to_csv(out_dir / "score_matrix.csv")

    print("Saved:")
    print(" -", out_dir / "speaker_to_participant.json")
    print(" -", out_dir / "aligned_segments.json")
    print(" -", out_dir / "score_matrix.csv")
    if not HAS_SCIPY:
        print("NOTE: scipy not found -> used greedy assignment. Install scipy for optimal assignment.")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--segments", type=str, required=True)
#     ap.add_argument("--participants-dir", type=str, required=True)
#     ap.add_argument("--out", type=str, required=True)
#     ap.add_argument("--min-seg-sec", type=float, default=0.4)
#     ap.add_argument("--av-offset-sec", type=float, default=0.0)
#     ap.add_argument("--smooth-ms", type=float, default=240.0)
#     args = ap.parse_args()
#
#     segments = load_segments(Path(args.segments))
#     participants = load_participants(Path(args.participants_dir))
#
#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # Build mouth lookups
#     lookups = {}
#     debug_rows = []
#
#     for pid, df in participants.items():
#         t, mouth, dbg = build_mouth_lookup(df, smooth_ms=args.smooth_ms)
#         lookups[pid] = (t, mouth)
#         debug_rows.append({"participant": pid, **dbg})
#
#     # Save participant debug info
#     pd.DataFrame(debug_rows).to_csv(out_dir / "participant_debug.csv", index=False)
#
#     aligned = []
#
#     for seg in segments:
#         start = float(seg["start"]) + args.av_offset_sec
#         end = float(seg["end"]) + args.av_offset_sec
#         dur = end - start
#
#         if dur < args.min_seg_sec:
#             continue
#
#         best_pid = None
#         best_score = -1.0
#
#         for pid, (t, mouth) in lookups.items():
#             s = mean_mouth_in_interval(t, mouth, start, end)
#             if s > best_score:
#                 best_score = s
#                 best_pid = pid
#
#         aligned.append({
#             "start": start,
#             "end": end,
#             "speaker": seg["speaker"],
#             "participant": best_pid,
#             "mean_mouth_score": round(best_score, 4),
#         })
#
#     (out_dir / "aligned_segments.json").write_text(
#         json.dumps(aligned, indent=2)
#     )
#
#     print("Saved:")
#     print(" - aligned_segments.json")
#     print(" - participant_debug.csv")

if __name__ == "__main__":
    main()
