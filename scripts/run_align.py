from fusion.align_speakers_to_faces import main as align_main

if __name__ == "__main__":
    import sys

    sys.argv = [
        sys.argv[0],

        # === REQUIRED INPUTS ===
        "--segments", "../runs/trimmed_session01/segments.json",
        "--participants-dir", "../data/face_reader/facereader_norm",
        "--out", "../runs/fusion_out",

        # === TUNING PARAMETERS ===
        "--min-seg-sec", "0.6",
        "--av-offset-sec", "0.0",
        "--smooth-ms", "240",
    ]

    align_main()
