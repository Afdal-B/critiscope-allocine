"""Évaluation commune BERTopic / LDA (O4) : cohérence NPMI et C_V, diversité, comparaison.

Les deux modèles sont évalués avec les mêmes textes (tokens + bigrammes) et le même dictionnaire.
"""

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)


def build_texts(tokens: Iterable[Sequence[str]], ngram_range: Sequence[int]) -> list[list[str]]:
    """Textes pour gensim : unigrammes et bigrammes entrelacés par position.

    L'entrelacement garde chaque bigramme à côté de ses mots, pour les fenêtres glissantes
    des mesures de cohérence. Les bigrammes s'écrivent comme dans BERTopic (« mise scène »).
    """
    lo, hi = ngram_range
    texts = []
    for toks in tokens:
        toks = list(toks)
        texts.append(
            [
                " ".join(toks[i : i + n])
                for i in range(len(toks))
                for n in range(lo, hi + 1)
                if i + n <= len(toks)
            ]
        )
    return texts


def dictionary_path(cfg: dict[str, Any], n: int) -> Path:
    """Chemin du dictionnaire commun : data/processed/dictionary_<n>.gensim."""
    return ROOT / cfg["paths"]["processed"] / f"dictionary_{n}.gensim"


def load_or_build_dictionary(cfg: dict[str, Any], n: int, texts: list[list[str]]) -> Dictionary:
    """Recharge le dictionnaire commun, ou le construit et le filtre (no_below, no_above)."""
    path = dictionary_path(cfg, n)
    if path.exists():
        dictionary = Dictionary.load(str(path))
        logger.info("Dictionnaire chargé : %s (%d termes)", path, len(dictionary))
        return dictionary
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(
        no_below=cfg["lda"]["no_below"], no_above=cfg["lda"]["no_above"], keep_n=None
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(path))
    logger.info("Dictionnaire construit : %s (%d termes)", path, len(dictionary))
    return dictionary


def restrict_to_dictionary(
    topics: list[list[str]], dictionary: Dictionary, topn: int
) -> tuple[list[list[str]], float]:
    """Garde, dans l'ordre, les topn premiers mots présents dans le dictionnaire.

    Renvoie aussi la couverture : part des topn mots d'origine présents dans le dictionnaire.
    """
    vocab = dictionary.token2id
    kept = [[w for w in words if w in vocab][:topn] for words in topics]
    coverage = sum(w in vocab for words in topics for w in words[:topn]) / (topn * len(topics))
    return kept, coverage


def coherence(
    topics: list[list[str]], texts: list[list[str]], dictionary: Dictionary, measure: str, topn: int
) -> list[float]:
    """Cohérence par thème (c_npmi ou c_v) calculée par gensim sur les textes communs."""
    model = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=measure,
        topn=topn,
        processes=1,
    )
    return [float(c) for c in model.get_coherence_per_topic()]


def diversity(topics: list[list[str]], topn: int) -> float:
    """Diversité des thèmes : part de mots uniques parmi les topn premiers de chaque thème."""
    words = [w for t in topics for w in t[:topn]]
    return len(set(words)) / len(words)


def evaluate_topics(
    name: str,
    topics: list[list[str]],
    texts: list[list[str]],
    dictionary: Dictionary,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Métriques globales et par thème d'un modèle (mots triés par importance décroissante)."""
    e = cfg["evaluation"]
    kept, coverage = restrict_to_dictionary(topics, dictionary, e["top_n_words"])
    npmi = coherence(kept, texts, dictionary, "c_npmi", e["top_n_words"])
    cv = coherence(kept, texts, dictionary, "c_v", e["top_n_words"])
    per_topic = pd.DataFrame(
        {
            "model": name,
            "topic": range(len(kept)),
            "npmi": npmi,
            "c_v": cv,
            "words": [", ".join(w) for w in kept],
        }
    )
    summary = {
        "model": name,
        "n_topics": len(topics),
        "npmi": float(pd.Series(npmi).mean()),
        "c_v": float(pd.Series(cv).mean()),
        "diversity": diversity(topics, e["diversity_top_n"]),
        "coverage": coverage,
    }
    return summary, per_topic


def bertopic_topics(topic_info_path: Path) -> list[list[str]]:
    """Mots-clés des thèmes BERTopic (hors -1), dans l'ordre des thèmes."""
    info = pd.read_csv(topic_info_path)
    info = info[info["Topic"] != -1].sort_values("Topic")
    return [json.loads(r.replace("'", '"')) for r in info["Representation"]]


def main() -> None:
    """CLI : compare BERTopic (run de la config) et LDA (K principal et meilleur K de la grille)."""
    from src.embeddings import variant_name
    from src.train_bertopic import run_tag

    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["sample_size"])
    parser.add_argument("--tag", help="run BERTopic à évaluer (défaut : celui de config.yaml)")
    args = parser.parse_args()
    tag = args.tag or run_tag(cfg, variant_name(cfg["embedding"]["model"]), args.n)

    metrics = ROOT / cfg["paths"]["outputs"] / "metrics"
    run = json.loads((metrics / f"run_{tag}.json").read_text(encoding="utf-8"))
    corpus = pd.read_parquet(ROOT / run["corpus"], columns=["tokens"])
    texts = build_texts(corpus["tokens"], cfg["bertopic"]["ngram_range"])
    dictionary = load_or_build_dictionary(cfg, args.n, texts)

    grid = pd.read_csv(metrics / f"lda_grid_{args.n}.csv")
    runs = pd.read_csv(metrics / "bertopic_runs.csv")
    bt_fit = float(runs.loc[runs["tag"] == tag, "fit_seconds"].iloc[-1])
    bt_topics = bertopic_topics(ROOT / run["topic_info"])
    k_main = len(bt_topics)
    k_best = int(grid.loc[grid["npmi"].idxmax(), "k"])

    candidates = [("BERTopic", bt_topics, bt_fit)]
    lda_runs = {k_main: f"LDA (K={k_main}, = BERTopic)"}
    lda_runs.setdefault(k_best, f"LDA (K={k_best}, meilleur NPMI)")
    for k, label in lda_runs.items():
        words = json.loads((metrics / f"lda_topics_{args.n}_k{k}.json").read_text("utf-8"))
        fit = float(grid.loc[grid["k"] == k, "fit_seconds"].iat[0])
        candidates.append((label, words, fit))

    rows, per_topic = [], []
    for name, topics, fit in candidates:
        summary, detail = evaluate_topics(name, topics, texts, dictionary, cfg)
        summary["fit_seconds"] = fit
        rows.append(summary)
        per_topic.append(detail)
        logger.info(
            "%-32s NPMI=%.3f C_V=%.3f diversité=%.2f couverture=%.0f %% (%d thèmes)",
            name,
            summary["npmi"],
            summary["c_v"],
            summary["diversity"],
            100 * summary["coverage"],
            summary["n_topics"],
        )

    comparison = pd.DataFrame(rows)
    comparison["note"] = [
        "fit_seconds hors calcul des embeddings (e5-large-instruct, ~6 min sur MPS)"
        if r.startswith("BERTopic")
        else ""
        for r in comparison["model"]
    ]
    comparison.to_csv(metrics / "comparison.csv", index=False)
    pd.concat(per_topic).to_csv(metrics / "coherence_per_topic.csv", index=False)
    logger.info("Comparaison écrite : %s", metrics / "comparison.csv")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("gensim").setLevel(logging.WARNING)
    main()
