import subprocess
from pathlib import Path


def prepare_audio(
        input_path: str,
        output_path: str,
        sample_rate: int = 16000,
):
    """
    Prepare audio for NeMo diarization:
    - decode (m4a, mp4, wav, etc.)
    - convert to mono
    - resample to 16 kHz
    - output PCM WAV
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
    return output_path
