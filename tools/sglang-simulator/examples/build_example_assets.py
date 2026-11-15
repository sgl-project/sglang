#!/usr/bin/env python3
"""Rebuild the small ML and tokenizer assets used by examples and tests."""

from pathlib import Path

import joblib
import numpy as np
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sklearn.dummy import DummyRegressor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

ASSETS = Path(__file__).parent / "assets"


def build_ml_model() -> None:
    model = DummyRegressor(strategy="constant", constant=0.001)
    model.fit(np.zeros((1, len(MLTimePredictor.FEATURE_NAMES))), [0.001])
    joblib.dump(
        {"model": model, "features": MLTimePredictor.FEATURE_NAMES},
        ASSETS / "model.pkl",
    )


def build_tokenizer() -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "prefix": 1,
                "caching": 2,
                "latency": 3,
                "decode": 4,
                "token": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
    ).save_pretrained(ASSETS / "tokenizer")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_ml_model()
    build_tokenizer()


if __name__ == "__main__":
    main()
