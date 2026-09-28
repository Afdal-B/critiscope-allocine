"""Thèmes × sentiment (O3) : part d'avis positifs par thème, tests, mots par label et figure."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binomtest, chi2_contingency, pearsonr, spearmanr

from src.config import ROOT, load_config
from src.embeddings import variant_name
from src.train_bertopic import build_ctfidf, build_docs, build_vectorizer, run_tag

logger = logging.getLogger(__name__)

COLORS = {"positif": "#2a78d6", "négatif": "#e34948", "neutre": "#c3c2b7"}


def load_run(cfg: dict[str, Any], tag: str) -> dict[str, Any]:
    """Lit le manifeste écrit par train_bertopic pour ce run."""
    path = ROOT / cfg["paths"]["outputs"] / "metrics" / f"run_{tag}.json"
    if not path.exists():
        raise SystemExit(
            f"{path} introuvable : relancez `python -m src.train_bertopic` pour ce run"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def topic_names(topic_info: pd.DataFrame, k: int = 4) -> dict[int, str]:
    """Nom court de chaque thème : ses k premiers mots-clés."""
    names = {}
    for row in topic_info.itertuples():
        words = row.Representation
        words = json.loads(words.replace("'", '"')) if isinstance(words, str) else words
        names[row.Topic] = ", ".join(words[:k])
    return names


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """p-values corrigées pour les tests multiples (taux de fausses découvertes)."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = q
    return out


def sentiment_table(labels: pd.Series, topics: pd.Series, alpha: float) -> pd.DataFrame:
    """Par thème : effectif, part de positifs, écart à la moyenne, IC de Wilson et test binomial.

    Chaque thème est comparé à la part globale de positifs (H0 : même proportion).
    """
    base = labels.mean()
    rows = []
    for topic, grp in labels.groupby(topics):
        if topic == -1:
            continue
        n, k = len(grp), int(grp.sum())
        test = binomtest(k, n, p=base)
        ci = test.proportion_ci(method="wilson")
        rows.append(
            {
                "topic": int(topic),
                "n": n,
                "n_pos": k,
                "share_pos": k / n,
                "diff_pp": 100 * (k / n - base),
                "ci_low": ci.low,
                "ci_high": ci.high,
                "p_value": test.pvalue,
            }
        )
    df = pd.DataFrame(rows)
    df["q_value"] = benjamini_hochberg(df["p_value"].to_numpy())
    df["significant"] = df["q_value"] < alpha
    df["direction"] = np.where(
        ~df["significant"], "neutre", np.where(df["diff_pp"] > 0, "positif", "négatif")
    )
    return df.sort_values("diff_pp").reset_index(drop=True)


def global_test(labels: pd.Series, topics: pd.Series) -> dict[str, float]:
    """Test du χ² d'indépendance thème × label, et V de Cramér (taille d'effet)."""
    mask = topics != -1
    table = pd.crosstab(topics[mask], labels[mask])
    chi2, p, dof, _ = chi2_contingency(table)
    cramers_v = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": cramers_v}


def class_words(
    tokens: pd.Series, labels: pd.Series, topics: pd.Series, cfg: dict[str, Any], k: int
) -> pd.DataFrame:
    """Mots distinctifs de chaque (thème, label) par c-TF-IDF, comme topics_per_class.

    Recalculé ici sur les mêmes documents et le même vectorizer que l'entraînement, sans
    recharger le modèle (et son modèle d'embedding de 2 Go).
    """
    docs = pd.Series(build_docs(tokens, cfg), index=tokens.index)
    frame = pd.DataFrame({"doc": docs, "label": labels, "topic": topics})
    frame = frame[frame["topic"] != -1]
    grouped = frame.groupby(["topic", "label"])["doc"]
    groups, sizes = grouped.apply(" ".join), grouped.size()
    vectorizer = build_vectorizer(cfg).fit(groups.tolist())
    counts = vectorizer.transform(groups.tolist())
    scores = build_ctfidf(cfg).fit_transform(counts).toarray()
    vocab = vectorizer.get_feature_names_out()
    rows = []
    for (topic, label), row in zip(groups.index, scores, strict=True):
        top = np.argsort(row)[::-1][:k]
        rows.append(
            {
                "topic": topic,
                "label": label,
                "n_docs": int(sizes[(topic, label)]),
                "words": ", ".join(vocab[top]),
            }
        )
    return pd.DataFrame(rows)


def representative_reviews(
    corpus: pd.DataFrame, topics_df: pd.DataFrame, embeddings: np.ndarray, k: int
) -> pd.DataFrame:
    """Pour chaque thème et chaque label, les k critiques les plus proches du centre du thème.

    Le centre est calculé sur les membres « natifs » (topic_raw == topic), pas sur les
    critiques réaffectées par reduce_outliers.
    """
    rows = []
    for topic in sorted(t for t in topics_df["topic"].unique() if t != -1):
        members = np.flatnonzero(topics_df["topic"].to_numpy() == topic)
        core = members[topics_df["topic_raw"].to_numpy()[members] == topic]
        centroid = embeddings[core if len(core) else members].mean(axis=0)
        sims = embeddings[members] @ (centroid / np.linalg.norm(centroid))
        for label in (0, 1):
            idx = [i for i in members[np.argsort(sims)[::-1]] if corpus["label"].iat[i] == label]
            for rank, i in enumerate(idx[:k], start=1):
                rows.append(
                    {
                        "topic": int(topic),
                        "label": label,
                        "rank": rank,
                        "similarity": float(sims[np.flatnonzero(members == i)[0]]),
                        "review": corpus["review"].iat[i],
                    }
                )
    return pd.DataFrame(rows)


def robustness(final: pd.DataFrame, raw: pd.DataFrame) -> dict[str, float]:
    """Compare les résultats avec et sans les critiques réaffectées par reduce_outliers."""
    m = final.merge(raw, on="topic", suffixes=("", "_raw"))
    return {
        "pearson_diff": float(pearsonr(m["diff_pp"], m["diff_pp_raw"]).statistic),
        "spearman_diff": float(spearmanr(m["diff_pp"], m["diff_pp_raw"]).statistic),
        "same_direction": float((m["direction"] == m["direction_raw"]).mean()),
    }


def diverging_figure(df: pd.DataFrame, names: dict[int, str], base: float, title: str) -> go.Figure:
    """Barres divergentes : écart de chaque thème à la part moyenne d'avis positifs."""
    df = df.assign(name=[f"{t} · {names.get(t, '')}" for t in df["topic"]])
    fig = go.Figure()
    legend = {
        "négatif": "Plus négatif (significatif)",
        "neutre": "Non significatif",
        "positif": "Plus positif (significatif)",
    }
    for direction, label in legend.items():
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["diff_pp"],
            y=sub["name"],
            orientation="h",
            name=label,
            marker_color=COLORS[direction],
            error_x={
                "type": "data",
                "symmetric": False,
                "array": 100 * sub["ci_high"] - 100 * sub["share_pos"],
                "arrayminus": 100 * sub["share_pos"] - 100 * sub["ci_low"],
                "color": "#8a8983",
                "thickness": 1,
                "width": 3,
            },
            customdata=np.stack([sub["n"], 100 * sub["share_pos"], sub["q_value"]], axis=1),
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[0]} critiques · %{customdata[1]:.0f} % positives"
                "<br>écart : %{x:+.1f} points · q = %{customdata[2]:.2g}<extra></extra>"
            ),
        )
    fig.add_vline(x=0, line_width=1, line_color="#52514e")
    fig.update_layout(
        title=f"{title}<br><sup>Part moyenne d'avis positifs : {100 * base:.1f} % ; "
        "barres d'erreur : IC 95 % (Wilson)</sup>",
        template="simple_white",
        barmode="overlay",
        bargap=0.25,
        barcornerradius=4,
        height=26 * len(df) + 180,
        margin={"l": 20, "r": 20, "t": 90, "b": 50},
        xaxis={
            "title": "Écart à la moyenne (points de pourcentage)",
            "zeroline": False,
            "gridcolor": "#ecebe7",
            "showgrid": True,
        },
        yaxis={"categoryorder": "array", "categoryarray": df["name"].tolist(), "automargin": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        font={"color": "#0b0b0b"},
    )
    return fig


def log_extremes(df: pd.DataFrame, names: dict[int, str], reviews: pd.DataFrame, k: int) -> None:
    """Affiche les k thèmes les plus négatifs et les plus positifs, avec deux exemples chacun."""
    for title, part in (("négatifs", df.head(k)), ("positifs", df.tail(k)[::-1])):
        logger.info("Top %d des thèmes les plus %s :", k, title)
        for row in part.itertuples():
            logger.info(
                "  %2d | %-45s | n=%5d | %3.0f %% positifs (%+.1f pts) | q=%.2g",
                row.topic,
                names.get(row.topic, "")[:45],
                row.n,
                100 * row.share_pos,
                row.diff_pp,
                row.q_value,
            )
            ex = reviews[(reviews["topic"] == row.topic) & (reviews["rank"] == 1)]
            for ex_row in ex.itertuples():
                tone = "+" if ex_row.label == 1 else "-"
                logger.info("       (%s) %s", tone, ex_row.review[:160].replace("\n", " "))


def main() -> None:
    """CLI : analyse thèmes × sentiment pour un run de train_bertopic."""
    cfg = load_config()
    s = cfg["sentiment"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["sample_size"])
    parser.add_argument("--tag", help="run à analyser (défaut : celui de config.yaml)")
    args = parser.parse_args()
    default_tag = run_tag(cfg, variant_name(cfg["embedding"]["model"]), args.n)
    tag = args.tag or default_tag

    run = load_run(cfg, tag)
    corpus = pd.read_parquet(ROOT / run["corpus"])
    topics_df = pd.read_parquet(ROOT / run["topics"])
    embeddings = np.load(ROOT / run["embeddings"])
    names = topic_names(pd.read_csv(ROOT / run["topic_info"]))
    labels = corpus["label"]
    base = float(labels.mean())
    logger.info("Run %s : %d critiques, %.1f %% positives", tag, len(corpus), 100 * base)

    final = sentiment_table(labels, topics_df["topic"], s["alpha"])
    raw = sentiment_table(labels, topics_df["topic_raw"], s["alpha"])
    for df in (final, raw):
        df.insert(1, "keywords", df["topic"].map(names))
    words = class_words(corpus["tokens"], labels, topics_df["topic"], cfg, s["n_class_words"])
    reviews = representative_reviews(corpus, topics_df, embeddings, s["n_examples"])

    summary = {
        "tag": tag,
        "n_docs": len(corpus),
        "base_share_pos": base,
        "global_test": global_test(labels, topics_df["topic"]),
        "global_test_raw": global_test(labels, topics_df["topic_raw"]),
        "n_significant_neg": int((final["direction"] == "négatif").sum()),
        "n_significant_pos": int((final["direction"] == "positif").sum()),
        "n_topics": len(final),
        "robustness_raw_vs_final": robustness(final, raw),
    }
    g = summary["global_test"]
    logger.info(
        "χ² thème × label : χ²=%.0f (ddl %d), p=%.2g, V de Cramér=%.2f | %d thèmes plus négatifs, "
        "%d plus positifs, %d non significatifs (q < %.2f)",
        g["chi2"],
        g["dof"],
        g["p_value"],
        g["cramers_v"],
        summary["n_significant_neg"],
        summary["n_significant_pos"],
        len(final) - summary["n_significant_neg"] - summary["n_significant_pos"],
        s["alpha"],
    )
    logger.info(
        "Robustesse (affectations d'origine vs réaffectées) : %s",
        summary["robustness_raw_vs_final"],
    )
    log_extremes(final, names, reviews, k=5)

    metrics = ROOT / cfg["paths"]["outputs"] / "metrics"
    final.to_csv(metrics / f"topic_sentiment_{tag}.csv", index=False)
    raw.to_csv(metrics / f"topic_sentiment_raw_{tag}.csv", index=False)
    words.to_csv(metrics / f"topic_sentiment_words_{tag}.csv", index=False)
    reviews.to_csv(metrics / f"representative_reviews_{tag}.csv", index=False)
    (metrics / f"sentiment_summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if tag == default_tag:
        final.to_csv(metrics / "topic_sentiment.csv", index=False)

    fig_path: Path = ROOT / cfg["paths"]["outputs"] / "figures" / f"sentiment_{tag}.html"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    diverging_figure(final, names, base, "Thèmes × sentiment").write_html(
        fig_path, include_plotlyjs="cdn"
    )
    logger.info("Sorties écrites dans %s et %s", metrics, fig_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
