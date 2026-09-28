"""Artefacts, inférence et figures de l'app CritiScope (aucun entraînement ni LLM)."""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_REPO = "Dalfaxy/critiscope-bertopic"

POS, NEG, NEUTRAL = "#2a78d6", "#e34948", "#b8b7b1"
DIRECTION_COLORS = {"positif": POS, "négatif": NEG, "neutre": NEUTRAL}
DIRECTION_NAMES = {
    "négatif": "Plus sévère",
    "neutre": "Dans la moyenne",
    "positif": "Plus apprécié",
}


def artifacts_dir() -> Path:
    """Dossier d'artefacts : local s'il existe, sinon téléchargé depuis le HF Hub."""
    local = Path(os.environ.get("CRITISCOPE_ARTIFACTS", ROOT / "artifacts"))
    if (local / "summary.json").exists():
        return local
    repo = os.environ.get("CRITISCOPE_MODEL_REPO", DEFAULT_MODEL_REPO)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo, repo_type="model"))


@dataclass
class Tables:
    """Données tabulaires de l'app."""

    topics: pd.DataFrame
    reviews: pd.DataFrame
    class_words: pd.DataFrame
    representative: pd.DataFrame
    comparison: pd.DataFrame
    lda_grid: pd.DataFrame
    comparison_notes: str
    summary: dict[str, Any]


def load_tables(path: Path) -> Tables:
    """Lit les tables exportées par src.export_artifacts."""
    return Tables(
        topics=pd.read_parquet(path / "topics.parquet"),
        reviews=pd.read_parquet(path / "reviews.parquet"),
        class_words=pd.read_csv(path / "class_words.csv"),
        representative=pd.read_csv(path / "representative_reviews.csv"),
        comparison=pd.read_csv(path / "comparison.csv"),
        lda_grid=pd.read_csv(path / "lda_grid.csv"),
        comparison_notes=(path / "comparison_notes.md").read_text(encoding="utf-8"),
        summary=json.loads((path / "summary.json").read_text(encoding="utf-8")),
    )


class TopicPredictor:
    """Embedding d'une critique (même modèle et même préfixe qu'à l'entraînement) + BERTopic."""

    def __init__(self, path: Path) -> None:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        from src.text_cleaning import clean_text

        self.clean = clean_text

        settings = json.loads((path / "embedding.json").read_text(encoding="utf-8"))
        self.prefix = settings["prefix"]
        self.encoder = SentenceTransformer(settings["model"], device="cpu")
        self.encoder.max_seq_length = settings["max_seq_length"]
        self.model = BERTopic.load(str(path / "bertopic"), embedding_model=self.encoder)
        topic_emb = np.asarray(self.model.topic_embeddings_, dtype=np.float32)
        self.topic_emb = topic_emb / np.linalg.norm(topic_emb, axis=1, keepdims=True)
        pool = np.load(path / "review_embeddings.npy").astype(np.float32)
        self.pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)
        self.embed("échauffement")

    def embed(self, text: str) -> np.ndarray:
        """Embedding normalisé d'un texte nettoyé comme le corpus."""
        return self.encoder.encode(
            [self.prefix + self.clean(text)], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def predict(self, text: str, k_topics: int, k_similar: int) -> dict[str, Any]:
        """Thème prédit (model.transform), thèmes candidats et critiques les plus proches."""
        start = time.perf_counter()
        emb = self.embed(text)
        topics, probs = self.model.transform([text], embeddings=emb)
        sims = (self.topic_emb @ emb[0]).ravel()
        top = np.argsort(sims)[::-1][:k_topics]
        pool_sims = self.pool @ emb[0]
        similar = np.argpartition(pool_sims, -k_similar)[-k_similar:]
        similar = similar[np.argsort(pool_sims[similar])[::-1]]
        return {
            "topic": int(topics[0]),
            "similarity": float(np.max(probs)) if probs is not None else float(sims.max()),
            "candidates": [(int(t), float(sims[t])) for t in top],
            "similar": [(int(i), float(pool_sims[i])) for i in similar],
            "seconds": time.perf_counter() - start,
        }


def diverging_chart(df: pd.DataFrame, base: float) -> go.Figure:
    """Barres divergentes : écart de chaque thème à la part moyenne d'avis positifs."""
    df = df.sort_values("diff_pp")
    fig = go.Figure()
    for direction, name in DIRECTION_NAMES.items():
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["diff_pp"],
            y=sub["label"],
            orientation="h",
            name=name,
            marker_color=DIRECTION_COLORS[direction],
            customdata=np.stack([sub["n"], 100 * sub["share_pos"]], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]} critiques · %{customdata[1]:.0f} % positives"
                "<extra></extra>"
            ),
        )
    fig.add_vline(x=0, line_width=1, line_color="#8a8983")
    fig.update_layout(
        barmode="overlay",
        bargap=0.25,
        barcornerradius=4,
        height=26 * len(df) + 140,
        margin={"l": 10, "r": 10, "t": 40, "b": 40},
        xaxis={"title": f"Écart à la moyenne des avis positifs ({100 * base:.0f} %), en points"},
        yaxis={"categoryorder": "array", "categoryarray": df["label"].tolist(), "title": None},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": None},
    )
    return fig
