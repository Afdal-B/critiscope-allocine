"""Prétraitement des critiques : text_clean (embeddings) et tokens (LDA, mots-clés)."""

import argparse
import logging
from typing import Any

import pandas as pd
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.tokens import Token

from src.config import ROOT, load_config
from src.text_cleaning import clean_text, prepare_for_tokens

__all__ = ["Preprocessor", "clean_text", "prepare_for_tokens"]

logger = logging.getLogger(__name__)


class Preprocessor:
    """Applique le nettoyage et la tokenisation définis dans config.yaml."""

    def __init__(
        self,
        spacy_model: str,
        stopwords: set[str],
        remove_persons: bool,
        min_tokens: int,
        batch_size: int = 64,
    ) -> None:
        self.stopwords = {w.lower() for w in stopwords}
        self.remove_persons = remove_persons
        self.min_tokens = min_tokens
        self.batch_size = batch_size
        disabled = ["parser"] if remove_persons else ["parser", "ner"]
        self.nlp = spacy.load(spacy_model, disable=disabled)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Preprocessor":
        """Construit le préprocesseur à partir de config.yaml."""
        pre = cfg["preprocessing"]
        stopwords = set(STOP_WORDS) | set(pre["extra_stopwords"])
        if pre["remove_sentiment_words"]:
            stopwords |= set(pre["sentiment_words"])
        return cls(
            spacy_model=pre["spacy_model"],
            stopwords=stopwords,
            remove_persons=pre["remove_person_entities"],
            min_tokens=cfg["data"]["min_tokens"],
            batch_size=pre["batch_size"],
        )

    def _keep(self, token: Token) -> bool:
        """Indique si un token doit figurer dans la sortie `tokens`."""
        if not token.is_alpha or len(token) < 2:
            return False
        if self.remove_persons and token.ent_type_ == "PER":
            return False
        return token.lemma_.lower() not in self.stopwords

    def tokenize(self, texts: list[str]) -> list[list[str]]:
        """Lemmatise et filtre un lot de textes (déjà passés par clean_text)."""
        prepared = [prepare_for_tokens(t) for t in texts]
        docs = self.nlp.pipe(prepared, batch_size=self.batch_size)
        return [[t.lemma_.lower() for t in doc if self._keep(t)] for doc in docs]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute text_clean, tokens, n_tokens et filtre les critiques trop courtes."""
        df = df.copy()
        df["text_clean"] = df["review"].map(clean_text)
        df["tokens"] = self.tokenize(df["text_clean"].tolist())
        df["n_tokens"] = df["tokens"].map(len)
        n_before = len(df)
        df = df[df["n_tokens"] >= self.min_tokens].reset_index(drop=True)
        logger.info(
            "%d critiques retirées (< %d tokens), %d conservées",
            n_before - len(df),
            self.min_tokens,
            len(df),
        )
        return df


def log_examples(df: pd.DataFrame, k: int, seed: int) -> None:
    """Affiche k exemples avant/après prétraitement dans les logs."""
    for i, row in enumerate(df.sample(k, random_state=seed).itertuples(), start=1):
        logger.info(
            "Exemple %d (label=%d)\n  brut   : %s\n  clean  : %s\n  tokens : %s",
            i,
            row.label,
            row.review[:150],
            row.text_clean[:150],
            row.tokens[:15],
        )


def main() -> None:
    """CLI : lit data/raw/allocine_<n>.parquet et écrit data/processed/corpus_<n>.parquet."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["dev_sample_size"])
    args = parser.parse_args()

    src = ROOT / cfg["paths"]["raw"] / f"allocine_{args.n}.parquet"
    if not src.exists():
        raise SystemExit(f"{src} introuvable : lancez d'abord `python -m src.data --n {args.n}`")

    df = pd.read_parquet(src)
    logger.info("%d critiques chargées depuis %s", len(df), src)
    corpus = Preprocessor.from_config(cfg).transform(df)

    out = ROOT / cfg["paths"]["processed"] / f"corpus_{args.n}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out, index=False)
    logger.info("Corpus sauvegardé : %s", out)
    log_examples(corpus, k=10, seed=cfg["seed"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
