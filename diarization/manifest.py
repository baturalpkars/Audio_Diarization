import json
from pathlib import Path
from typing import Optional


def write_manifest(
        wav_path: str | Path,
        manifest_path: str | Path,
        num_speakers: Optional[int] = None,
) -> Path:
    wav_path = Path(wav_path).resolve()
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "audio_filepath": str(wav_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "",
        "num_speakers": num_speakers,
    }

    # JSONL: one JSON object per line
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return manifest_path
