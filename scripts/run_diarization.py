"""
Run speaker diarization using NVIDIA NeMo (your diarization.pipeline.diarize).

PyCharm-friendly:
- edit the constants at the bottom
- press Run
"""

from pathlib import Path
from diarization import diarize


def main(
    input_media: str,
    out_dir: str,
    profile: str = "meeting",
    num_speakers: int | None = None,
    config_path: str = "../diarization/configs/diarization.yaml",
):
    input_media = Path(input_media)
    out_dir = Path(out_dir)

    if not input_media.exists():
        raise FileNotFoundError(f"Input media not found: {input_media}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("▶ Running diarization")
    print(f"  Input:       {input_media}")
    print(f"  Out dir:     {out_dir}")
    print(f"  Profile:     {profile}")
    print(f"  Num speakers:{num_speakers}")
    print(f"  Config:      {config_path}")

    result = diarize(
        input_path=input_media,
        out_dir=out_dir,
        profile=profile,
        num_speakers=num_speakers,
        config_path=config_path,
    )

    print("\n✔ Diarization finished")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    # 👇 EDIT THESE in PyCharm
    INPUT_MEDIA = "../data/trimmed_zoom.mp3"          # video OR audio (mp4/mp3/wav/m4a)
    OUT_DIR = "../runs/trimmed_session_01"
    PROFILE = "meeting"                             # meeting / telephonic / general (must exist in YAML)
    NUM_SPEAKERS = None                             # set int if you know (e.g., 2), else None
    CONFIG_PATH = "../diarization/configs/diarization.yaml"

    main(
        input_media=INPUT_MEDIA,
        out_dir=OUT_DIR,
        profile=PROFILE,
        num_speakers=NUM_SPEAKERS,
        config_path=CONFIG_PATH,
    )
