from fusion.align_speakers_to_faces import main as align_main

if __name__ == "__main__":
    import sys

    sys.argv = [
        sys.argv[0],

        # === REQUIRED INPUTS ===
        "--segments", "../runs/test_02/segments.json",
        "--participants-dir", "../data/face_reader/facereader_02",
        "--out", "../runs/fusion_out_02",

        # === TUNING PARAMETERS ===
        "--min-seg-sec", "0.6",
        "--av-offset-sec", "0.0",
        "--smooth-ms", "240",
        "--margin-ratio", "1.08",
        "--best-min", "0.06",
        "--stick-ratio", "0.92",
        "--use-global-map",
    ]

    align_main()
