"""Publie les artefacts de l'app sur le HF Hub (dépôt modèle) (O6).

L'app, déployée sur Streamlit Community Cloud, télécharge ces artefacts au démarrage.
Par défaut, n'affiche que ce qui serait publié ; --push publie réellement.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from safetensors.numpy import load_file

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)


def check_artifacts(path: Path) -> dict[str, Any]:
    """Refuse de publier des artefacts incohérents (ex. une ligne -1 résiduelle dans le modèle)."""
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    emb = load_file(str(path / "bertopic" / "topic_embeddings.safetensors"))["topic_embeddings"]
    outliers = json.loads((path / "bertopic" / "topics.json").read_text(encoding="utf-8"))
    expected = summary["n_topics"] + outliers["_outliers"]
    if emb.shape[0] != expected:
        raise SystemExit(
            f"Modèle incohérent : {emb.shape[0]} embeddings de thèmes pour {expected} attendus. "
            "Relancez train_bertopic, label_topics, sentiment_analysis puis export_artifacts."
        )
    return summary


def model_card(summary: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Carte du dépôt modèle, construite uniquement à partir de summary.json."""
    s, lab = summary["sentiment"], summary["labeling"]
    emb_file = ROOT / cfg["paths"]["artifacts"] / "embedding.json"
    emb_model = json.loads(emb_file.read_text(encoding="utf-8"))["model"]
    return f"""---
library_name: bertopic
language: fr
tags: [bertopic, topic-modeling, allocine]
datasets: [tblard/allocine]
---

# CritiScope — BERTopic sur les critiques Allociné

Artefacts de l'app [CritiScope]({cfg["hub"]["github"]}) : modèle BERTopic
(sérialisation safetensors), tables des thèmes et du sentiment, critiques et leurs embeddings.

- Corpus : {summary["n_docs"]} critiques de `tblard/allocine` (split train, échantillon aléatoire)
- Embeddings : `{emb_model}`
  avec une instruction orientée genre et sujet (voir `embedding.json`)
- {summary["n_topics"]} thèmes ; {100 * summary["outlier_ratio_before"]:.0f} % de critiques hors
  cluster avant réaffectation (`reduce_outliers`)
- Lien thème × sentiment : V de Cramér = {s["global_test"]["cramers_v"]:.2f} ;
  {s["n_significant_neg"]} thèmes significativement plus négatifs,
  {s["n_significant_pos"]} plus positifs
- Étiquettes des thèmes : `{lab["model"]}`, validées par {lab["method"]}

Pour prédire un thème, encoder la critique avec le même modèle **et le même préfixe**
(`embedding.json`), puis `BERTopic.load("bertopic").transform([texte], embeddings=...)`.
"""


def main() -> None:
    """CLI : vérifie puis publie (avec --push) le dépôt modèle."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="publie réellement sur le HF Hub")
    args = parser.parse_args()

    hub = cfg["hub"]
    model_repo = f"{hub['user']}/{hub['model_repo']}"
    artifacts = ROOT / cfg["paths"]["artifacts"]
    summary = check_artifacts(artifacts)
    (artifacts / "README.md").write_text(model_card(summary, cfg), encoding="utf-8")

    size = sum(f.stat().st_size for f in artifacts.rglob("*") if f.is_file()) / 1e6
    logger.info("Dépôt modèle %s (public) ← %s (%.0f Mo)", model_repo, artifacts, size)
    if not args.push:
        logger.info("Simulation uniquement : relancez avec --push pour publier.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(artifacts),
        repo_id=model_repo,
        repo_type="model",
        commit_message="Artefacts CritiScope",
    )
    logger.info("Publié : https://huggingface.co/%s", model_repo)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
