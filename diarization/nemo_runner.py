import subprocess
from pathlib import Path


def run_nemo_offline_diarization(
        nemo_repo_dir: Path,
        nemo_infer_script_rel: str,
        nemo_infer_config_dir_rel: str,
        nemo_config_name: str,
        manifest_path: Path,
        out_dir: Path,
) -> None:
    nemo_repo_dir = nemo_repo_dir.resolve()
    infer_script = (nemo_repo_dir / nemo_infer_script_rel).resolve()
    config_dir = (nemo_repo_dir / nemo_infer_config_dir_rel).resolve()

    if not infer_script.exists():
        raise FileNotFoundError(f"NeMo inference script not found: {infer_script}")

    if not config_dir.exists():
        raise FileNotFoundError(f"NeMo inference config dir not found: {config_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(infer_script),
        f"--config-path={str(config_dir)}",
        f"--config-name={nemo_config_name}",
        f"diarizer.manifest_filepath={str(manifest_path.resolve())}",
        f"diarizer.out_dir={str(out_dir.resolve())}",
        "num_workers=0",
    ]

    # Run in NeMo repo root to avoid import/path issues inside NeMo scripts
    subprocess.run(cmd, check=True, cwd=str(nemo_repo_dir))
