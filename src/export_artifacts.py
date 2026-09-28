"""Prépare le dossier artifacts/ lu par l'app (O6) : modèle, tables, critiques et embeddings.

L'app ne dépend que de ce dossier (publié sur le HF Hub) : ni données brutes, ni config, ni LLM.
"""

import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from src.config import ROOT, load_config
from src.embeddings import variant_name
from src.sentiment_analysis import load_run
from src.train_bertopic import run_tag

logger = logging.getLogger(__name__)


def topic_centroids(embeddings: np.ndarray, topics: np.ndarray) -> np.ndarray:
    """Centre normalisé de chaque thème (0..K-1), dans l'ordre des thèmes."""
    ids = sorted(t for t in set(topics.tolist()) if t != -1)
    c = np.stack([embeddings[topics == t].mean(axis=0) for t in ids])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def topic_map(centroids: np.ndarray, seed: int) -> np.ndarray:
    """Coordonnées 2D des thèmes (MDS sur la distance cosinus entre leurs centres)."""
    dist = np.clip(1 - centroids @ centroids.T, 0, None)
    np.fill_diagonal(dist, 0)
    mds = MDS(n_components=2, metric="precomputed", random_state=seed, n_init=4, init="random")
    return mds.fit_transform(dist)


def topic_table(
    cfg: dict[str, Any], run: dict[str, Any], tag: str, centroids: np.ndarray
) -> pd.DataFrame:
    """Une ligne par thème : étiquette, mots-clés pondérés, sentiment, cohérence, position."""
    outputs = ROOT / cfg["paths"]["outputs"]
    labels = json.loads((outputs / "labels.json").read_text(encoding="utf-8"))
    if labels["tag"] != tag:
        raise SystemExit(f"labels.json concerne {labels['tag']}, pas {tag} : relancez label_topics")
    reps = json.loads((ROOT / run["model_dir"] / "topics.json").read_text(encoding="utf-8"))
    sentiment = pd.read_csv(outputs / "metrics" / f"topic_sentiment_{tag}.csv")
    coherence = pd.read_csv(outputs / "metrics" / "coherence_per_topic.csv")
    coherence = coherence[coherence["model"] == "BERTopic"][["topic", "npmi", "c_v"]]

    rows = []
    for t in sorted(int(k) for k in labels["labels"]):
        lab = labels["labels"][str(t)]
        words = reps["topic_representations"][str(t)][: cfg["evaluation"]["top_n_words"]]
        rows.append(
            {
                "topic": t,
                "label": lab["label"],
                "description": lab["description"],
                "kind": lab["kind"],
                "words": [w for w, _ in words],
                "word_scores": [float(s) for _, s in words],
            }
        )
    table = pd.DataFrame(rows)
    table = table.merge(sentiment.drop(columns=["keywords"]), on="topic").merge(
        coherence, on="topic", how="left"
    )
    xy = topic_map(centroids, cfg["seed"])
    table["x"], table["y"] = xy[table["topic"], 0], xy[table["topic"], 1]
    return table.sort_values("topic").reset_index(drop=True)


def copy_file(src: Path, dst_dir: Path, name: str | None = None) -> None:
    """Copie un fichier de sortie dans le dossier d'artefacts."""
    shutil.copy2(src, dst_dir / (name or src.name))


def main() -> None:
    """CLI : (re)construit artifacts/ à partir du run BERTopic de la config."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["sample_size"])
    parser.add_argument("--tag", help="run BERTopic (défaut : celui de config.yaml)")
    args = parser.parse_args()
    tag = args.tag or run_tag(cfg, variant_name(cfg["embedding"]["model"]), args.n)

    run = load_run(cfg, tag)
    if not run.get("model_dir"):
        raise SystemExit(f"Le run {tag} n'a pas de modèle sauvegardé (lancé avec --no-save ?)")
    out = ROOT / cfg["paths"]["artifacts"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    metrics = ROOT / cfg["paths"]["outputs"] / "metrics"

    corpus = pd.read_parquet(ROOT / run["corpus"], columns=["review", "label"])
    topics = pd.read_parquet(ROOT / run["topics"])["topic"].to_numpy()
    embeddings = np.load(ROOT / run["embeddings"])
    centroids = topic_centroids(embeddings, topics)

    shutil.copytree(ROOT / run["model_dir"], out / "bertopic")
    (out / "embedding.json").write_text(
        json.dumps(run["embedding"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = topic_table(cfg, run, tag, centroids)
    table.to_parquet(out / "topics.parquet", index=False)
    np.save(out / "topic_centroids.npy", centroids.astype(np.float32))

    corpus.assign(topic=topics).to_parquet(out / "reviews.parquet", index=False)
    np.save(out / "review_embeddings.npy", embeddings.astype(np.float16))

    copy_file(metrics / f"topic_sentiment_words_{tag}.csv", out, "class_words.csv")
    copy_file(metrics / f"representative_reviews_{tag}.csv", out, "representative_reviews.csv")
    copy_file(metrics / "comparison.csv", out)
    copy_file(metrics / "comparison_notes.md", out)
    copy_file(metrics / f"lda_grid_{args.n}.csv", out, "lda_grid.csv")

    runs = pd.read_csv(metrics / "bertopic_runs.csv")
    last = runs[runs["tag"] == tag].iloc[-1]
    summary = {
        "tag": tag,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_docs": int(len(corpus)),
        "n_topics": int(len(table)),
        "outlier_ratio_before": float(last["outlier_ratio_before"]),
        "sentiment": json.loads((metrics / f"sentiment_summary_{tag}.json").read_text("utf-8")),
        "labeling": {
            "model": json.loads((ROOT / cfg["paths"]["outputs"] / "labels.json").read_text())[
                "model"
            ],
            **json.loads((metrics / "labels_agreement.json").read_text(encoding="utf-8")),
        },
        "app": cfg["app"],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    logger.info(
        "Artefacts écrits dans %s (%.0f Mo, %d thèmes, %d critiques)",
        out,
        size,
        len(table),
        len(corpus),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
