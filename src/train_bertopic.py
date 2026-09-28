"""Entraînement de BERTopic (O2) : thèmes, réduction des outliers, sauvegarde et figures."""

import argparse
import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.config import ROOT, load_config
from src.embeddings import Embedder, cache_path, load_or_compute, variant_name

logger = logging.getLogger(__name__)


def build_representation(cfg: dict[str, Any]) -> Any:
    """Modèle de représentation des mots-clés (None = c-TF-IDF seul)."""
    b = cfg["bertopic"]
    if b["representation"] == "mmr":
        return MaximalMarginalRelevance(diversity=b["mmr_diversity"], top_n_words=b["top_n_words"])
    if b["representation"] == "keybert":
        return KeyBERTInspired(top_n_words=b["top_n_words"], random_state=cfg["seed"])
    return None


NGRAM_SEP = "|"


def build_docs(tokens: pd.Series, cfg: dict[str, Any]) -> list[str]:
    """Documents textuels pour le c-TF-IDF.

    Avec dedupe_tokens, chaque n-gramme ne compte qu'une fois par critique : une critique qui
    répète un mot 200 fois ne peut plus dominer les mots-clés de son thème. Les n-grammes
    uniques sont séparés par NGRAM_SEP (voir segment_analyzer) pour ne pas créer de faux bigrammes.
    """
    b = cfg["bertopic"]
    if not b["dedupe_tokens"]:
        return [" ".join(t) for t in tokens]
    lo, hi = b["ngram_range"]
    docs = []
    for toks in tokens:
        toks = list(toks)
        grams = (
            " ".join(toks[i : i + n]) for n in range(lo, hi + 1) for i in range(len(toks) - n + 1)
        )
        docs.append(f" {NGRAM_SEP} ".join(dict.fromkeys(grams)) + f" {NGRAM_SEP}")
    return docs


def segment_analyzer(text: str) -> list[str]:
    """Analyseur du CountVectorizer quand les n-grammes sont déjà construits par build_docs."""
    return [g.strip() for g in text.split(NGRAM_SEP) if g.strip()]


def build_vectorizer(cfg: dict[str, Any]) -> CountVectorizer:
    """CountVectorizer sur les tokens déjà lemmatisés et filtrés (pas de stopwords ici)."""
    b = cfg["bertopic"]
    if b["dedupe_tokens"]:
        return CountVectorizer(analyzer=segment_analyzer, min_df=b["min_df"])
    return CountVectorizer(ngram_range=tuple(b["ngram_range"]), min_df=b["min_df"])


def build_ctfidf(cfg: dict[str, Any]) -> ClassTfidfTransformer:
    """Pondération c-TF-IDF (option : atténuer les mots fréquents dans tous les thèmes)."""
    return ClassTfidfTransformer(reduce_frequent_words=cfg["bertopic"]["reduce_frequent_words"])


def build_model(cfg: dict[str, Any], embedder: Embedder) -> BERTopic:
    """Assemble UMAP, HDBSCAN, vectorizer et représentation à partir de la config."""
    u, h, b = cfg["umap"], cfg["hdbscan"], cfg["bertopic"]
    umap_model = UMAP(
        n_neighbors=u["n_neighbors"],
        n_components=u["n_components"],
        min_dist=u["min_dist"],
        metric=u["metric"],
        random_state=cfg["seed"],
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=h["min_cluster_size"],
        min_samples=h["min_samples"],
        metric="euclidean",
        cluster_selection_method=h["cluster_selection_method"],
        prediction_data=True,
    )
    return BERTopic(
        embedding_model=embedder.model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=build_vectorizer(cfg),
        ctfidf_model=build_ctfidf(cfg),
        representation_model=build_representation(cfg),
        top_n_words=b["top_n_words"],
        nr_topics=b["target_nr_topics"],
        language="multilingual",
        verbose=True,
    )


def outlier_ratio(topics: list[int]) -> float:
    """Part des documents dans le thème -1."""
    return float(np.mean(np.asarray(topics) == -1))


def reduce_outliers_if_needed(
    model: BERTopic,
    docs: list[str],
    topics: list[int],
    embeddings: np.ndarray,
    cfg: dict[str, Any],
) -> list[int]:
    """Réaffecte les outliers si le thème -1 dépasse max_outlier_ratio, puis met à jour les mots."""
    b = cfg["bertopic"]
    ratio = outlier_ratio(topics)
    if ratio <= b["max_outlier_ratio"]:
        logger.info(
            "Thème -1 : %.1f %% (≤ %.0f %%) : pas de réduction",
            100 * ratio,
            100 * b["max_outlier_ratio"],
        )
        return topics
    new_topics = model.reduce_outliers(
        docs,
        topics,
        strategy=b["outlier_strategy"],
        embeddings=embeddings,
        threshold=b["outlier_threshold"],
    )
    model.update_topics(
        docs,
        topics=new_topics,
        vectorizer_model=build_vectorizer(cfg),
        ctfidf_model=build_ctfidf(cfg),
        representation_model=build_representation(cfg),
        top_n_words=b["top_n_words"],
    )
    logger.info(
        "Thème -1 : %.1f %% → %.1f %% après reduce_outliers (%s)",
        100 * ratio,
        100 * outlier_ratio(new_topics),
        b["outlier_strategy"],
    )
    return new_topics


def set_topic_embeddings(model: BERTopic, topics: list[int], embeddings: np.ndarray) -> None:
    """Recalcule les embeddings de thèmes (moyenne des documents de chaque thème).

    Après reduce_outliers + update_topics, BERTopic peut garder la ligne du thème -1 en tête de
    topic_embeddings_ alors qu'il n'y a plus d'outliers : transform() renvoie alors le thème
    voisin (décalage d'un rang). On repart donc des embeddings des documents.
    """
    topics_arr = np.asarray(topics)
    ids = sorted(set(topics_arr.tolist()))
    model.topic_embeddings_ = np.stack([embeddings[topics_arr == t].mean(axis=0) for t in ids])
    expected = len(model.topic_representations_)
    if len(model.topic_embeddings_) != expected:
        raise RuntimeError(f"{len(model.topic_embeddings_)} embeddings de thèmes pour {expected}")


def export_figures(model: BERTopic, out_dir: Path) -> None:
    """Exporte les visualisations BERTopic en HTML (une figure qui échoue n'arrête pas le reste)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "topics_map": lambda: model.visualize_topics(),
        "barchart": lambda: model.visualize_barchart(top_n_topics=30, n_words=10),
        "hierarchy": lambda: model.visualize_hierarchy(),
    }
    for name, make in figures.items():
        try:
            make().write_html(out_dir / f"{name}.html", include_plotlyjs="cdn")
        except Exception as exc:
            logger.warning("Figure %s non générée : %s", name, exc)
    logger.info("Figures exportées dans %s", out_dir)


def append_run(path: Path, row: dict[str, Any]) -> None:
    """Ajoute le résumé d'un entraînement à outputs/metrics/bertopic_runs.csv."""
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def run_tag(cfg: dict[str, Any], variant: str, n: int, representation: str | None = None) -> str:
    """Identifiant d'un entraînement : variante d'embedding, taille, réglages HDBSCAN et fusion."""
    h = cfg["hdbscan"]
    tag = (
        f"{variant}_{n}"
        f"_mcs{h['min_cluster_size']}_ms{h['min_samples']}_{h['cluster_selection_method']}"
    )
    if cfg["bertopic"]["target_nr_topics"] is not None:
        tag += f"_nr{cfg['bertopic']['target_nr_topics']}"
    if representation is not None:
        tag += f"_{representation}"
    return tag


def parse_args(cfg: dict[str, Any]) -> argparse.Namespace:
    """Arguments CLI ; les surcharges HDBSCAN servent aux recherches de réglages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["dev_sample_size"])
    parser.add_argument("--model", help="surcharge embedding.model (comparaison de modèles)")
    parser.add_argument("--prefix", help="surcharge embedding.prefix ('' pour MiniLM)")
    parser.add_argument("--max-seq-length", type=int, help="surcharge la fenêtre du modèle")
    parser.add_argument("--min-cluster-size", type=int, help="surcharge hdbscan.min_cluster_size")
    parser.add_argument("--min-samples", type=int, help="surcharge hdbscan.min_samples")
    parser.add_argument("--selection", choices=["eom", "leaf"], help="surcharge la sélection")
    parser.add_argument("--nr-topics", type=int, help="surcharge bertopic.target_nr_topics")
    parser.add_argument(
        "--representation", choices=["ctfidf", "mmr", "keybert"], help="surcharge la représentation"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="essai rapide : ni modèle ni figures sauvegardés"
    )
    return parser.parse_args()


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Copie de la config avec les surcharges CLI (la config en cache n'est pas modifiée)."""
    cfg = copy.deepcopy(cfg)
    h = cfg["hdbscan"]
    if args.min_cluster_size is not None:
        h["min_cluster_size"] = args.min_cluster_size
    if args.min_samples is not None:
        h["min_samples"] = args.min_samples
    if args.selection is not None:
        h["cluster_selection_method"] = args.selection
    if args.nr_topics is not None:
        cfg["bertopic"]["target_nr_topics"] = args.nr_topics
    if args.representation is not None:
        cfg["bertopic"]["representation"] = args.representation
    return cfg


def main() -> None:
    """CLI : entraîne BERTopic sur corpus_<n> avec les embeddings mis en cache."""
    args = parse_args(load_config())
    cfg = apply_overrides(load_config(), args)
    h = cfg["hdbscan"]

    processed, outputs = ROOT / cfg["paths"]["processed"], ROOT / cfg["paths"]["outputs"]
    corpus = pd.read_parquet(processed / f"corpus_{args.n}.parquet")
    docs = build_docs(corpus["tokens"], cfg)

    embedder = Embedder.from_config(
        cfg, model_name=args.model, prefix=args.prefix, max_seq_length=args.max_seq_length
    )
    variant = variant_name(embedder.model_name, args.max_seq_length)
    emb_path = cache_path(cfg, variant, args.n)
    embeddings = load_or_compute(embedder, corpus["text_clean"].tolist(), emb_path)

    model = build_model(cfg, embedder)
    start = time.perf_counter()
    topics_raw, _ = model.fit_transform(docs, embeddings)
    ratio_before = outlier_ratio(topics_raw)
    topics = reduce_outliers_if_needed(model, docs, list(topics_raw), embeddings, cfg)
    set_topic_embeddings(model, topics, embeddings)
    fit_seconds = time.perf_counter() - start

    tag = run_tag(cfg, variant, args.n, args.representation)
    info = model.get_topic_info()
    n_topics = int((info["Topic"] != -1).sum())
    logger.info("%d thèmes (hors -1) en %.0f s", n_topics, fit_seconds)
    for _, row in info.head(40).iterrows():
        logger.info(
            "  %3d | %5d docs | %s",
            row["Topic"],
            row["Count"],
            ", ".join(w for w, _ in model.get_topic(row["Topic"])[:8]),
        )

    metrics_dir = outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    info.to_csv(metrics_dir / f"topic_info_{tag}.csv", index=False)
    topics_path = processed / f"topics_{tag}.parquet"
    corpus.assign(topic_raw=topics_raw, topic=topics)[["label", "topic_raw", "topic"]].to_parquet(
        topics_path, index=False
    )
    model_dir = None
    if not args.no_save:
        export_figures(model, outputs / "figures" / f"bertopic_{tag}")
        model_dir = ROOT / cfg["paths"]["models"] / f"bertopic_{tag}"
        model.save(
            model_dir,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=embedder.model_name,
        )
        logger.info("Modèle sauvegardé : %s", model_dir)

    manifest = {
        "tag": tag,
        "n": args.n,
        "embedding": embedder.settings,
        "corpus": str((processed / f"corpus_{args.n}.parquet").relative_to(ROOT)),
        "embeddings": str(emb_path.relative_to(ROOT)),
        "topics": str(topics_path.relative_to(ROOT)),
        "topic_info": str((metrics_dir / f"topic_info_{tag}.csv").relative_to(ROOT)),
        "model_dir": str(model_dir.relative_to(ROOT)) if model_dir else None,
    }
    (metrics_dir / f"run_{tag}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    append_run(
        metrics_dir / "bertopic_runs.csv",
        {
            "tag": tag,
            "embedding_model": embedder.model_name,
            "n_docs": len(docs),
            "min_cluster_size": h["min_cluster_size"],
            "min_samples": h["min_samples"],
            "selection": h["cluster_selection_method"],
            "target_nr_topics": cfg["bertopic"]["target_nr_topics"],
            "reduce_frequent_words": cfg["bertopic"]["reduce_frequent_words"],
            "dedupe_tokens": cfg["bertopic"]["dedupe_tokens"],
            "representation": cfg["bertopic"]["representation"],
            "n_topics": n_topics,
            "outlier_ratio_before": round(ratio_before, 4),
            "outlier_ratio_after": round(outlier_ratio(topics), 4),
            "fit_seconds": round(fit_seconds, 1),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
