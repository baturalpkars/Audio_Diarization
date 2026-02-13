from fusion.extract_facereader_mouth_signals import main as extract_main

# If you want: call internal functions instead of CLI parsing later.
if __name__ == "__main__":
    # Just run via CLI-style args injection (simple + reliable)
    import sys

    sys.argv = [
        sys.argv[0],
        "--xlsx", "../data/face_reader/facereader_detailed_02.xlsx",
        "--out", "../data/face_reader/facereader_02",
        "--quality-threshold", "0.7",
    ]
    extract_main()
