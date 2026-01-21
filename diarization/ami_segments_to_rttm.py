import xml.etree.ElementTree as ET
from pathlib import Path


def segments_xml_to_rttm_lines(xml_path: Path, meeting_id: str, speaker_label: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []
    for elem in root.iter():
        if elem.tag.endswith("segment"):
            start = elem.attrib.get("transcriber_start")
            end = elem.attrib.get("transcriber_end")

            if start is None or end is None:
                continue

            start = float(start)
            end = float(end)
            dur = end - start
            if dur <= 0:
                continue

            lines.append(
                f"SPEAKER {meeting_id} 1 {start:.3f} {dur:.3f} "
                f"<NA> <NA> {speaker_label} <NA> <NA>"
            )
    return lines


def build_meeting_rttm(annotations_root: Path, meeting_id: str, out_rttm: Path):
    seg_dir = annotations_root / "segments"
    assert seg_dir.exists(), f"Segments dir not found: {seg_dir}"

    all_lines = []
    for spk in ["A", "B", "C", "D"]:
        xml_path = seg_dir / f"{meeting_id}.{spk}.segments.xml"
        if xml_path.exists():
            all_lines.extend(
                segments_xml_to_rttm_lines(xml_path, meeting_id, spk)
            )

    all_lines.sort(key=lambda x: float(x.split()[3]))

    out_rttm.parent.mkdir(parents=True, exist_ok=True)
    out_rttm.write_text("\n".join(all_lines) + "\n")
    print(f"Wrote {len(all_lines)} RTTM lines → {out_rttm}")


if __name__ == "__main__":
    meeting_id = "ES2007a"  # change as needed
    annotations_root = Path("../ami/annotations_manual")
    out_rttm = Path(f"ami/ref_rttm/{meeting_id}.rttm")

    build_meeting_rttm(annotations_root, meeting_id, out_rttm)
