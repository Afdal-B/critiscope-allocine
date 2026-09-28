"""Télécharge les artefacts du HF Hub et le modèle d'embedding qu'ils référencent (build Docker)."""

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

path = Path(snapshot_download(repo_id=os.environ["CRITISCOPE_MODEL_REPO"], repo_type="model"))
settings = json.loads((path / "embedding.json").read_text(encoding="utf-8"))
SentenceTransformer(settings["model"], device="cpu")
print(f"Artefacts : {path} ; modèle d'embedding : {settings['model']}")
