"""Calcul et mise en cache des embeddings des critiques (entrée de BERTopic)."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)


class Embedder:
    """Encapsule le modèle d'embedding et applique toujours le même préfixe.

    Utilisé à l'entraînement comme dans l'app, pour garantir des embeddings comparables.
    """

    def __init__(
        self,
        model_name: str,
        prefix: str = "",
        max_seq_length: int | None = None,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        logger.info(
            "Modèle %s chargé (device=%s, max_seq_length=%d, préfixe=%r)",
            model_name,
            self.model.device,
            self.model.max_seq_length,
            prefix,
        )

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        model_name: str | None = None,
        prefix: str | None = None,
        max_seq_length: int | None = None,
    ) -> "Embedder":
        """Construit l'embedder depuis config.yaml ; les arguments non nuls surchargent la config.

        Sans surcharge, max_seq_length de la config ne s'applique qu'au modèle de la config :
        un autre modèle garde sa valeur native.
        """
        emb = cfg["embedding"]
        device = emb.get("device", "auto")
        name = model_name or emb["model"]
        if max_seq_length is None and name == emb["model"]:
            max_seq_length = emb.get("max_seq_length")
        return cls(
            model_name=name,
            prefix=emb.get("prefix", "") if prefix is None else prefix,
            max_seq_length=max_seq_length,
            batch_size=emb["batch_size"],
            device=None if device == "auto" else device,
        )

    @property
    def settings(self) -> dict[str, Any]:
        """Réglages qui déterminent les embeddings (enregistrés à côté du cache)."""
        return {
            "model": self.model_name,
            "prefix": self.prefix,
            "max_seq_length": self.model.max_seq_length,
        }

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode des textes (préfixe ajouté) en vecteurs normalisés, float32."""
        return self.model.encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        ).astype(np.float32)


def model_slug(model_name: str) -> str:
    """Nom du modèle utilisable dans un nom de fichier."""
    return model_name.split("/")[-1]


def variant_name(model_name: str, max_seq_length: int | None = None) -> str:
    """Identifiant d'une variante : modèle, plus la longueur si elle est surchargée en CLI."""
    suffix = f"_len{max_seq_length}" if max_seq_length is not None else ""
    return model_slug(model_name) + suffix


def cache_path(cfg: dict[str, Any], variant: str, n: int) -> Path:
    """Chemin du cache : data/processed/embeddings_<variante>_<n>.npy."""
    return ROOT / cfg["paths"]["processed"] / f"embeddings_{variant}_{n}.npy"


def _cache_is_valid(emb: np.ndarray, meta_path: Path, embedder: Embedder, n_texts: int) -> bool:
    """Vérifie que le cache couvre le corpus et a été calculé avec les mêmes réglages."""
    if emb.shape[0] != n_texts:
        logger.warning("Cache incohérent (%d ≠ %d textes)", emb.shape[0], n_texts)
        return False
    if not meta_path.exists():
        logger.warning("Cache sans métadonnées (%s) : réglages non vérifiés", meta_path.name)
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta != embedder.settings:
        logger.warning("Réglages du cache différents : %s ≠ %s", meta, embedder.settings)
        return False
    return True


def load_or_compute(embedder: Embedder, texts: list[str], path: Path) -> np.ndarray:
    """Recharge le cache s'il correspond au corpus et aux réglages, sinon calcule."""
    meta_path = path.with_suffix(".json")
    if path.exists():
        emb = np.load(path)
        if _cache_is_valid(emb, meta_path, embedder, len(texts)):
            logger.info("Embeddings chargés depuis le cache : %s %s", path, emb.shape)
            return emb
        logger.info("Recalcul des embeddings")
    emb = embedder.encode(texts, show_progress_bar=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    meta_path.write_text(json.dumps(embedder.settings, ensure_ascii=False, indent=2), "utf-8")
    logger.info("Embeddings sauvegardés : %s %s", path, emb.shape)
    return emb


def main() -> None:
    """CLI : calcule (ou recharge) les embeddings de data/processed/corpus_<n>.parquet."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["dev_sample_size"])
    parser.add_argument("--model", help="surcharge embedding.model (comparaison de modèles)")
    parser.add_argument("--prefix", help="surcharge embedding.prefix ('' pour MiniLM)")
    parser.add_argument("--max-seq-length", type=int, help="surcharge la fenêtre du modèle")
    args = parser.parse_args()

    corpus_path = ROOT / cfg["paths"]["processed"] / f"corpus_{args.n}.parquet"
    if not corpus_path.exists():
        raise SystemExit(
            f"{corpus_path} introuvable : lancez d'abord `python -m src.preprocessing --n {args.n}`"
        )
    texts = pd.read_parquet(corpus_path, columns=["text_clean"])["text_clean"].tolist()

    embedder = Embedder.from_config(
        cfg, model_name=args.model, prefix=args.prefix, max_seq_length=args.max_seq_length
    )
    variant = variant_name(embedder.model_name, args.max_seq_length)
    load_or_compute(embedder, texts, cache_path(cfg, variant, args.n))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
