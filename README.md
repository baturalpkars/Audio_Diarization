# Audio Diarization + Face–Speaker Alignment

This repository contains a working pipeline for speaker diarization (NVIDIA NeMo) and face–speaker alignment using FaceReader mouth signals. It also includes evaluation utilities that compare aligned results to manual annotations with permutation-invariant scoring.

## Pipeline Summary

1. **Diarization (audio/video)**  
   Run NeMo to produce diarized segments: `runs/<case>/segments.json`.

2. **FaceReader extraction (video)**  
   Extract per-participant mouth signals from FaceReader `.xlsx` into `participant_*.csv`.

3. **Alignment (audio ↔ faces)**  
   Align diarized speakers to face participants using mouth activity signals, with global consistency and ambiguity handling.

4. **Evaluation**  
   Compare alignment against GT annotations using remapped (permutation-invariant) accuracy.

## Repository Layout

- `diarization/`  
  NeMo diarization wrapper and configs.
- `fusion/`  
  FaceReader extraction and alignment logic.
- `scripts/`  
  Entry points for running diarization, extraction, alignment, and eval.
- `data/`  
  Inputs (FaceReader `.xlsx`, manual annotations).
- `runs/`  
  Outputs for diarization and evaluation.

## Key Scripts

- `scripts/run_diarization.py`  
  Runs NeMo diarization to generate `segments.json`.

- `scripts/run_extract.py`  
  Converts FaceReader `.xlsx` into `participant_*.csv`.

- `scripts/run_align.py`  
  Runs alignment using a single case (manual CLI args).

- `scripts/run_eval_alignment.py`  
  Batch evaluation using `scripts/eval_cases.json`. Produces per-case reports and a summary.

## Evaluation (Permutation-Invariant)

Speaker and participant IDs are arbitrary. Evaluation computes a best participant-ID mapping and reports:

- `overall_accuracy` (raw labels, not meaningful for alignment)
- `remapped_accuracy` (permutation-invariant, main metric)

Reports are written to `runs/eval_out/<case>/eval_report.txt`.

## Alignment Controls (Important)

Key CLI flags in `fusion/align_speakers_to_faces.py`:

- `--use-global-map`  
  Enables global speaker→participant mapping for stability.
- `--ambiguous-global-max-sec`  
  If `> 0`, fixed threshold. If `<= 0`, auto threshold is used.
- `--ambiguous-global-max-ratio`  
  Auto threshold ratio based on total meeting duration (clamped to 5–30s).
- `--ambiguous-local-margin`  
  If ambiguous and `best/top2` is very close, force local decision.

These are tuned to avoid global lock-in on long ambiguous segments while keeping global consistency on short segments.

## Typical Workflow

1. Run diarization  
   `python scripts/run_diarization.py`

2. Extract FaceReader signals  
   `python scripts/run_extract.py`

3. Run alignment  
   `python scripts/run_align.py`

4. Evaluate all test cases  
   `python scripts/run_eval_alignment.py --skip-extract`

## C# Handoff Plan (High-Level)

This pipeline is intended to be used from a C# application. The recommended split is:

- **Neural models (exportable)**  
  NeMo components (VAD / embeddings) can be exported to ONNX where feasible.
- **Algorithmic logic (re-implement)**  
  Alignment and evaluation logic should be ported to C# directly (not ONNX), since it is not a neural model.

See the “Handoff Plan” section in the collaboration notes for details.

## Notes

- Remapped accuracy is the correct indicator of alignment quality.
- Some errors are due to diarization coverage (if a GT speaker is missing, alignment cannot recover it).
- FaceReader quality and visibility strongly affect mouth signal reliability.
