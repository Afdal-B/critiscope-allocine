"""Baseline LDA (O4) : même tokens et même dictionnaire que l'évaluation de BERTopic."""

import argparse
import json
import logging
import time
from typing import Any

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from src.config import ROOT, load_config
from src.evaluate import build_texts, evaluate_topics, load_or_build_dictionary

logger = logging.getLogger(__name__)


def train_lda(bow: list, dictionary: Dictionary, k: int, cfg: dict[str, Any]) -> LdaModel:
    """LDA gensim mono-cœur (reproductible avec random_state, contrairement à LdaMulticore)."""
    lda = cfg["lda"]
    return LdaModel(
        corpus=bow,
        id2word=dictionary,
        num_topics=k,
        passes=lda["passes"],
        iterations=lda["iterations"],
        chunksize=lda["chunksize"],
        alpha=lda["alpha"],
        eta=lda["eta"],
        random_state=cfg["seed"],
        eval_every=None,
    )


def topic_words(model: LdaModel, topn: int) -> list[list[str]]:
    """Mots de chaque thème, par probabilité décroissante."""
    return [[w for w, _ in model.show_topic(t, topn=topn)] for t in range(model.num_topics)]


def bertopic_k(cfg: dict[str, Any], n: int) -> int:
    """Nombre de thèmes (hors -1) du run BERTopic de la config."""
    from src.embeddings import variant_name
    from src.train_bertopic import run_tag

    tag = run_tag(cfg, variant_name(cfg["embedding"]["model"]), n)
    info = pd.read_csv(ROOT / cfg["paths"]["outputs"] / "metrics" / f"topic_info_{tag}.csv")
    return int((info["Topic"] != -1).sum())


def main() -> None:
    """CLI : entraîne LDA pour K = nb de thèmes BERTopic et pour chaque K de la grille."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["sample_size"])
    parser.add_argument("--k", type=int, help="K principal (défaut : nb de thèmes BERTopic)")
    parser.add_argument("--no-grid", action="store_true", help="n'entraîne que le K principal")
    args = parser.parse_args()

    corpus = pd.read_parquet(ROOT / cfg["paths"]["processed"] / f"corpus_{args.n}.parquet")
    texts = build_texts(corpus["tokens"], cfg["bertopic"]["ngram_range"])
    dictionary = load_or_build_dictionary(cfg, args.n, texts)
    bow = [dictionary.doc2bow(t) for t in texts]

    k_main = args.k or bertopic_k(cfg, args.n)
    ks = [k_main] if args.no_grid else sorted({k_main, *cfg["lda"]["k_grid"]})
    logger.info("%d documents, %d termes, K = %s", len(bow), len(dictionary), ks)

    metrics = ROOT / cfg["paths"]["outputs"] / "metrics"
    rows = []
    for k in ks:
        start = time.perf_counter()
        model = train_lda(bow, dictionary, k, cfg)
        fit_seconds = time.perf_counter() - start
        words = topic_words(model, cfg["evaluation"]["diversity_top_n"])
        summary, _ = evaluate_topics(f"LDA K={k}", words, texts, dictionary, cfg)
        rows.append({"k": k, **summary, "fit_seconds": round(fit_seconds, 1)})
        logger.info(
            "K=%2d | %.0f s | NPMI=%.3f C_V=%.3f diversité=%.2f",
            k,
            fit_seconds,
            summary["npmi"],
            summary["c_v"],
            summary["diversity"],
        )

        (metrics / f"lda_topics_{args.n}_k{k}.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        model_dir = ROOT / cfg["paths"]["models"] / f"lda_{args.n}_k{k}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir / "lda.model"))

    grid_path = metrics / f"lda_grid_{args.n}.csv"
    grid = pd.DataFrame(rows).drop(columns="model")
    if grid_path.exists():
        old = pd.read_csv(grid_path)
        grid = pd.concat([old[~old["k"].isin(grid["k"])], grid]).sort_values("k")
    grid.to_csv(grid_path, index=False)
    logger.info("Grille écrite : %s\n%s", grid_path, grid.round(3).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("gensim").setLevel(logging.WARNING)
    main()
