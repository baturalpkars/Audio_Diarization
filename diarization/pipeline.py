import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict

import yaml

from .audio_prep import prepare_audio
from .manifest import write_manifest
from .nemo_runner import run_nemo_offline_diarization


def _parse_rttm(rttm_path: Path) -> List[Dict[str, Any]]:
    segments = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if parts[0].upper() != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append({"start": start, "end": start + dur, "speaker": speaker})
    return segments


def summarize_segments(segments):
    talk_time = defaultdict(float)
    turns = defaultdict(int)

    for s in segments:
        spk = s["speaker"]
        dur = float(s["end"]) - float(s["start"])
        talk_time[spk] += dur
        turns[spk] += 1

    summary = []
    for spk in sorted(talk_time.keys()):
        summary.append({
            "speaker": spk,
            "total_s": round(talk_time[spk], 2),
            "turns": int(turns[spk]),
            "avg_turn_s": round(talk_time[spk] / turns[spk], 2),
        })
    return summary


def diarize(
        input_path: str | Path,
        out_dir: str | Path,
        profile: str = "meeting",
        num_speakers: Optional[int] = None,
        config_path: str | Path = "diarization/configs/diarization.yaml",
) -> Dict[str, Path]:
    """
    End-to-end diarization entrypoint (reusable for FaceReader integration).
    Returns key artifact paths: prepared wav, manifest, RTTM, segments.json.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load project config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sample_rate = int(cfg["audio"]["sample_rate"])
    nemo_repo_dir = Path(cfg["nemo"]["nemo_repo_dir"])
    nemo_infer_script_rel = cfg["nemo"]["nemo_infer_script_rel"]
    nemo_infer_config_dir_rel = cfg["nemo"]["nemo_infer_config_dir_rel"]

    if profile not in cfg["profiles"]:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(cfg['profiles'].keys())}")

    nemo_config_name = cfg["profiles"][profile]["nemo_config_name"]

    # 1) Prepare audio
    recording_id = input_path.stem.split(".")[0]
    prepared_wav = out_dir / f"{recording_id}.wav"

    prepare_audio(input_path, prepared_wav, sample_rate=sample_rate)

    # 2) Manifest
    manifest_path = out_dir / "diar_manifest.json"
    write_manifest(prepared_wav, manifest_path, num_speakers=num_speakers)

    # 3) Run NeMo
    nemo_out = out_dir / "nemo_out"
    run_nemo_offline_diarization(
        nemo_repo_dir=nemo_repo_dir,
        nemo_infer_script_rel=nemo_infer_script_rel,
        nemo_infer_config_dir_rel=nemo_infer_config_dir_rel,
        nemo_config_name=nemo_config_name,
        manifest_path=manifest_path,
        out_dir=nemo_out,
    )

    # 4) Collect RTTM
    pred_rttm_dir = nemo_out / "pred_rttms"
    rttms = sorted(pred_rttm_dir.glob("*.rttm"))
    if not rttms:
        raise FileNotFoundError(f"No RTTM produced. Expected in: {pred_rttm_dir}")
    rttm_path = rttms[0]

    # 5) Export JSON segments (easier for integration)
    segments = _parse_rttm(rttm_path)
    segments_json = out_dir / "segments.json"
    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    summary = summarize_segments(segments)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "prepared_wav": prepared_wav,
        "manifest": manifest_path,
        "nemo_out_dir": nemo_out,
        "rttm": rttm_path,
        "segments_json": segments_json,
        "summary_json": summary_path,
    }
