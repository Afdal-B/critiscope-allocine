"""Chargement et échantillonnage des critiques Allociné."""

import argparse
import logging

import pandas as pd
from datasets import load_dataset

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)


def load_reviews(n: int | None, seed: int) -> pd.DataFrame:
    """Charge le split depuis le Hub HF et renvoie n critiques tirées au hasard (tout si None)."""
    cfg = load_config()
    ds = load_dataset(cfg["data"]["dataset"], split=cfg["data"]["split"])
    ds_sample = ds.shuffle(seed=seed).select(range(n)) if n is not None else ds
    return ds_sample.to_pandas()[["review", "label"]]


def main() -> None:
    """CLI : échantillonne n critiques et les écrit dans data/raw/allocine_<n>.parquet."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=cfg["data"]["dev_sample_size"])
    args = parser.parse_args()

    df = load_reviews(args.n, cfg["seed"])
    out = ROOT / cfg["paths"]["raw"] / f"allocine_{args.n}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Échantillon sauvegardé : %s (%d critiques)", out, len(df))
    logger.info("Distribution des labels : %s", df["label"].value_counts().to_dict())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
