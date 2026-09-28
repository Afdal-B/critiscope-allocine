"""Étiquetage des thèmes par LLM (O5) : étiquette, description, sujet/ton ; validation humaine.

Hors ligne uniquement : l'app lit outputs/labels.json et n'appelle jamais de LLM.
"""

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import ROOT, load_config
from src.embeddings import variant_name
from src.sentiment_analysis import load_run, topic_names
from src.train_bertopic import run_tag

logger = logging.getLogger(__name__)

PROMPT = """Tu analyses un thème extrait automatiquement (topic modeling) de critiques de films \
publiées sur Allociné.

Mots-clés du thème, du plus au moins important (lemmatisés : « animer » peut venir de « animé », \
et quelques erreurs de lemmatisation sont possibles) :
{keywords}

Critiques représentatives du thème :
{reviews}

Réponds en JSON avec :
- "label" : une étiquette en français de 2 à 4 mots qui nomme le thème (ex. « Films d'horreur », \
« Cinéma japonais », « Critiques d'ennui »), sans guillemets ni numéro ;
- "description" : une phrase en français qui décrit ce dont parlent les critiques de ce thème ;
- "kind" : "sujet" si le thème regroupe les critiques par genre, sujet ou cinématographie ; "ton" \
s'il les regroupe surtout par un jugement (ennui, enthousiasme, nullité…) sans sujet commun."""


APPROVED = "validé tel quel"


class TopicLabel(BaseModel):
    """Réponse attendue du LLM pour un thème."""

    label: str = Field(description="Étiquette de 2 à 4 mots, en français")
    description: str = Field(description="Une phrase en français")
    kind: Literal["sujet", "ton"]

    @field_validator("label")
    @classmethod
    def two_to_four_words(cls, v: str) -> str:
        v = v.strip().strip("«»\"' ")
        n_words = len(v.replace("'", " ").replace("’", " ").split())
        if not 2 <= n_words <= 5:
            raise ValueError(f"l'étiquette doit faire 2 à 4 mots, reçu {n_words} : {v!r}")
        return v


def build_prompt(keywords: list[str], reviews: list[str], max_chars: int) -> str:
    """Prompt d'un thème : mots-clés et critiques représentatives (tronquées)."""
    shown = [
        r if len(r) <= max_chars else r[:max_chars].rsplit(" ", 1)[0] + " […]" for r in reviews
    ]
    return PROMPT.format(
        keywords=", ".join(keywords),
        reviews="\n".join(f"{i}. « {r} »" for i, r in enumerate(shown, start=1)),
    )


def is_transient(exc: Exception) -> bool:
    """Erreur qui peut disparaître en réessayant : quota (429), serveur (5xx) ou réseau."""
    from google.genai import errors

    if isinstance(exc, errors.ClientError):
        return exc.code == 429
    return True


class GeminiLabeler:
    """Appelle Gemini avec une sortie JSON contrainte par le schéma TopicLabel."""

    def __init__(
        self, model: str, api_key: str, temperature: float, retries: int = 2, timeout_s: int = 60
    ) -> None:
        from google import genai
        from google.genai import types

        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000)
        )
        self.model = model
        self.retries = retries
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TopicLabel,
            temperature=temperature,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _call(self, prompt: str) -> str:
        """Appel API avec reprise simple en cas d'erreur réseau, de quota ou de surcharge."""
        retries = self.retries
        for attempt in range(retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.config
                ).text
            except Exception as exc:
                if not is_transient(exc) or attempt == retries:
                    raise
                wait = 30 * (attempt + 1)
                logger.warning("Appel Gemini échoué (%s) : nouvel essai dans %d s", exc, wait)
                time.sleep(wait)
        raise RuntimeError("inaccessible")

    def label(self, prompt: str) -> TopicLabel:
        """Étiquette validée ; relance une fois avec l'erreur si la réponse est invalide."""
        text = self._call(prompt)
        try:
            return TopicLabel.model_validate_json(text)
        except ValidationError as exc:
            logger.warning("Réponse invalide, relance : %s", exc.errors()[0]["msg"])
            retry = f"{prompt}\n\nTa réponse précédente était invalide ({exc.errors()[0]['msg']}). "
            retry += "Corrige-la en respectant exactement le format demandé."
            return TopicLabel.model_validate_json(self._call(retry))


def load_inputs(cfg: dict[str, Any], run: dict[str, Any], tag: str) -> pd.DataFrame:
    """Par thème : mots-clés et critiques les plus proches du centre.

    La meilleure critique positive et la meilleure négative sont toujours incluses : trois
    exemples d'un seul label pousseraient le LLM vers une étiquette de ton.
    """
    lab = cfg["labeling"]
    info = pd.read_csv(ROOT / run["topic_info"])
    keywords = topic_names(info, k=lab["n_keywords"])
    reviews = pd.read_csv(
        ROOT / cfg["paths"]["outputs"] / "metrics" / f"representative_reviews_{tag}.csv"
    )
    rows = []
    for topic, grp in reviews.groupby("topic"):
        grp = grp.sort_values("similarity", ascending=False)
        best_each = grp.drop_duplicates("label")
        top = pd.concat([best_each, grp.drop(best_each.index)]).head(lab["n_representative_docs"])
        rows.append(
            {
                "topic": int(topic),
                "keywords": keywords[topic].split(", "),
                "reviews": top["review"].tolist(),
            }
        )
    return pd.DataFrame(rows)


def label_all(cfg: dict[str, Any], inputs: pd.DataFrame, existing: dict[str, Any]) -> dict:
    """Étiquette chaque thème absent de `existing` (reprise possible après une interruption)."""
    lab = cfg["labeling"]
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get(lab["api_key_env"], "")
    if not api_key:
        raise SystemExit(f"Clé absente : renseignez {lab['api_key_env']}=... dans .env")
    labeler = GeminiLabeler(
        lab["model"], api_key, lab["temperature"], lab["max_retries"], lab["timeout_seconds"]
    )

    labels = dict(existing)
    for row in inputs.itertuples():
        key = str(row.topic)
        if key in labels:
            continue
        prompt = build_prompt(row.keywords, row.reviews, lab["max_review_chars"])
        try:
            result = labeler.label(prompt)
        except Exception as exc:
            if not isinstance(exc, ValidationError) and not is_transient(exc):
                raise SystemExit(f"Erreur Gemini non récupérable : {exc}") from exc
            logger.error("Thème %s non étiqueté : %s", key, exc)
            continue
        labels[key] = {**result.model_dump(), "keywords": row.keywords}
        logger.info("%3s | %-30s | %-5s | %s", key, result.label, result.kind, result.description)
        time.sleep(lab["sleep_seconds"])
    return labels


def write_validation(path: Path, inputs: pd.DataFrame, labels: dict[str, Any], force: bool) -> None:
    """Crée la grille de validation humaine (sans écraser une grille déjà remplie)."""
    if path.exists() and not force:
        filled = pd.read_csv(path)["human_label"].notna().any()
        if filled:
            logger.warning("%s déjà remplie : non écrasée (utilisez --force)", path)
            return
    rows = []
    for row in inputs.itertuples():
        lab = labels.get(str(row.topic), {})
        rows.append(
            {
                "topic": row.topic,
                "keywords": ", ".join(row.keywords),
                "example": row.reviews[0][:300],
                "llm_label": lab.get("label"),
                "llm_description": lab.get("description"),
                "llm_kind": lab.get("kind"),
                "human_label": None,
                "match": None,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Grille de validation écrite : %s", path)


def agreement(path: Path, out: Path) -> dict[str, Any]:
    """Taux d'accord LLM / humain à partir de la colonne `match` remplie à la main."""
    df = pd.read_csv(path)
    scored = df.dropna(subset=["match"])
    if scored.empty:
        raise SystemExit(f"Colonne `match` vide dans {path} : remplissez-la (1 / 0.5 / 0)")
    result = {
        "n_topics": len(df),
        "n_scored": len(scored),
        "agreement": float(scored["match"].mean()),
        "exact_share": float((scored["match"] == 1).mean()),
        "by_kind": scored.groupby("llm_kind")["match"].mean().round(3).to_dict(),
        "threshold": 0.70,
        "method": "approbation"
        if (scored["human_label"].fillna("").str.strip() == APPROVED).all()
        else "étiquetage humain indépendant",
    }
    result["passed"] = result["agreement"] >= result["threshold"]
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def inject_labels(cfg: dict[str, Any], run: dict[str, Any], labels: dict[str, Any]) -> None:
    """Injecte les étiquettes dans le modèle BERTopic sauvegardé (set_topic_labels)."""
    from bertopic import BERTopic

    model_dir = ROOT / run["model_dir"]
    model = BERTopic.load(str(model_dir))
    model.set_topic_labels({int(t): v["label"] for t, v in labels.items()})
    model.save(
        str(model_dir),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=run["embedding"]["model"],
    )
    logger.info("Étiquettes injectées dans %s", model_dir)


def main() -> None:
    """CLI : étiquette les thèmes (labels.json + grille de validation) ou calcule l'accord."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["sample_size"])
    parser.add_argument("--tag", help="run BERTopic (défaut : celui de config.yaml)")
    parser.add_argument("--force", action="store_true", help="réétiquette tout et écrase la grille")
    parser.add_argument("--no-inject", action="store_true", help="ne modifie pas le modèle")
    parser.add_argument("--agreement", action="store_true", help="calcule l'accord LLM / humain")
    args = parser.parse_args()
    tag = args.tag or run_tag(cfg, variant_name(cfg["embedding"]["model"]), args.n)
    outputs = ROOT / cfg["paths"]["outputs"]
    validation_path = outputs / "labels_validation.csv"

    if args.agreement:
        res = agreement(validation_path, outputs / "metrics" / "labels_agreement.json")
        logger.info(
            "Accord LLM / humain : %.0f %% (%d thèmes notés) → %s",
            100 * res["agreement"],
            res["n_scored"],
            "critère atteint" if res["passed"] else "critère non atteint",
        )
        return

    run = load_run(cfg, tag)
    inputs = load_inputs(cfg, run, tag)
    labels_path = outputs / "labels.json"
    existing = {}
    if labels_path.exists() and not args.force:
        previous = json.loads(labels_path.read_text(encoding="utf-8"))
        existing = previous["labels"] if previous.get("tag") == tag else {}

    labels = label_all(cfg, inputs, existing)
    labels_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "model": cfg["labeling"]["model"],
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "labels": labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    missing = sorted(set(inputs["topic"].astype(str)) - set(labels), key=int)
    logger.info(
        "%d/%d thèmes étiquetés → %s%s",
        len(labels),
        len(inputs),
        labels_path,
        f" (manquants : {', '.join(missing)} ; relancez pour compléter)" if missing else "",
    )

    write_validation(validation_path, inputs, labels, args.force)
    if not args.no_inject and not missing and run.get("model_dir"):
        inject_labels(cfg, run, labels)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    main()
