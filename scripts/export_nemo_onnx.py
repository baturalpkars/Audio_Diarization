from pathlib import Path
from nemo.collections.asr.models import EncDecClassificationModel, EncDecSpeakerLabelModel

out_dir = Path("../models/onnx_export")
out_dir.mkdir(parents=True, exist_ok=True)

# VAD
vad = EncDecClassificationModel.from_pretrained(model_name="vad_multilingual_marblenet")
vad.eval()
vad.export(str(out_dir / "vad_multilingual_marblenet.onnx"))

# Speaker embeddings
spk = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
spk.eval()
spk.export(str(out_dir / "titanet_large.onnx"))

print("Exported ONNX models to:", out_dir)
