"""Compatibility layer for the original Siamese-LSTM repository.

This module is the Python 3.10/PyTorch replacement for the original
``SiameseLSTM.py``.  The recurrent computation itself lives in ``lstm.py``
and is implemented explicitly so that the original LSTM equations and
parameter layout remain unchanged.

The Siamese branches deliberately use one encoder instance.  Therefore the
same PyTorch parameters are used for both sentences, rather than creating
independently parameterized LSTMs.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from random import randint
from typing import Any, Mapping, MutableSequence, Sequence

import numpy as np
import scipy.stats as meas

from lstm import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    HIDDEN_DIM,
    INPUT_DIM,
    _MaLSTM,
    creatrnnx,
    getlayerx,
    getpl2 as _getpl2,
    get_layer as _get_layer,
    init_tparams,
    lstm_layer2,
    numpy_floatX,
    unzip,
    zipp,
)


# ---------------------------------------------------------------------------
# Legacy-compatible configuration names
# ---------------------------------------------------------------------------

prefix = "lstm"
noise_std = 0.0
training = True
Syn_aug = True

# These names existed as module-level objects in the Theano repository.  They
# are retained as simple Python state for compatibility with code that reads
# them; the PyTorch implementation does not use Theano shared variables.
use_noise = 0.0
flg = 1
rate = 0.5

# Optional embedding/synonym resources.  They are intentionally not loaded at
# import time because the original binary resources are large and may not be
# present in every reproduction environment.
model: Any = None
dtr: Mapping[str, Any] = {}
d2: Mapping[str, Any] = {}
cachedStopWords: set[str] = set()


class _SharedValue:
    """Small compatibility object exposing Theano-like ``set_value``."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def set_value(self, value: float) -> None:
        """Set the stored scalar value."""
        self.value = float(value)

    def get_value(self) -> float:
        """Return the stored scalar value."""
        return self.value


# Reassign to a compatibility object rather than a Theano dependency.
use_noise = _SharedValue(0.0)


# ---------------------------------------------------------------------------
# Data preparation helpers retained from the original repository
# ---------------------------------------------------------------------------


def prepare_data(
    data: Sequence[Sequence[Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare sentence pairs and labels in the original batch layout.

    Parameters
    ----------
    data:
        Sequence containing ``[sentence_a, sentence_b, score]`` records.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Token matrices, masks, and float32 labels for both sentence branches.
    """
    sentence_a = [str(item[0]) for item in data]
    sentence_b = [str(item[1]) for item in data]
    labels = np.asarray([item[2] for item in data], dtype=np.float32)

    lengths = [len(sentence.split()) for sentence in sentence_a]
    lengths.extend(len(sentence.split()) for sentence in sentence_b)
    if not lengths:
        raise ValueError("Cannot prepare an empty sentence-pair batch.")

    max_length = max(lengths)
    tokens_a, mask_a = getmtr(sentence_a, max_length)
    tokens_b, mask_b = getmtr(sentence_b, max_length)
    return tokens_a, mask_a, tokens_b, mask_b, labels


def getmtr(
    sentences: Sequence[str],
    maxlen: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sentences to padded token matrices and time-major masks."""
    if maxlen <= 0:
        raise ValueError("maxlen must be positive.")

    n_samples = len(sentences)
    token_rows: list[list[str]] = []
    mask = np.zeros((maxlen, n_samples), dtype=np.float32)

    for sample_index, sentence in enumerate(sentences):
        tokens = str(sentence).split()
        if len(tokens) > maxlen:
            raise ValueError("A sentence exceeds the supplied maxlen.")
        mask[: len(tokens), sample_index] = 1.0
        token_rows.append(tokens + [","] * (maxlen - len(tokens)))

    return np.asarray(token_rows), mask


def configure_resources(
    word2vec_model: Any,
    word_dictionary: Mapping[str, Any] | None = None,
    synonym_dictionary: Mapping[str, Any] | None = None,
    stop_words: Sequence[str] | None = None,
) -> None:
    """Configure optional Word2Vec and synonym resources used by helpers."""
    global model, dtr, d2, cachedStopWords
    model = word2vec_model
    dtr = word_dictionary or {}
    d2 = synonym_dictionary or {}
    cachedStopWords = set(stop_words or ())


def embed(stmx: np.ndarray) -> np.ndarray:
    """Map a token row to 300-D Word2Vec vectors using the configured model."""
    if model is None:
        raise RuntimeError(
            "No Word2Vec model is configured. Call configure_resources first."
        )
    if np.asarray(stmx).ndim != 1:
        raise ValueError("embed expects a one-dimensional token array.")

    vectors = np.zeros((len(stmx), INPUT_DIM), dtype=np.float32)
    for index, token in enumerate(stmx):
        token_text = str(token)
        if token_text == ",":
            continue

        lookup = token_text
        if token_text in dtr:
            lookup = dtr[token_text]

        try:
            vectors[index] = np.asarray(model[lookup], dtype=np.float32)
        except (KeyError, TypeError, IndexError) as exc:
            raise KeyError(f"Word2Vec vocabulary entry not found: {lookup!r}") from exc

    return vectors


# ---------------------------------------------------------------------------
# Synonym augmentation compatibility helpers
# ---------------------------------------------------------------------------


def pfl(sentence_list: MutableSequence[Any]) -> MutableSequence[Any]:
    """Append the original repository's synonym pool to a training list."""
    synonyms = dtr.get("syn")
    if synonyms is None:
        return sentence_list
    sentence_list.extend(synonyms[0])
    return sentence_list


def findsim(word: str) -> str:
    """Select one synonym using the original random-selection behavior."""
    synonyms = d2[word]
    if not synonyms:
        return ""
    return synonyms[randint(0, len(synonyms) - 1)]


def chsyn(sentence: str, train_data: MutableSequence[Any]) -> tuple[str, int]:
    """Apply the repository's synonym replacement rules to one sentence."""
    global flg

    words = str(sentence).split()
    replacements = 0
    result = list(words)

    for index, word in enumerate(words):
        if word not in d2 and word.lower() not in d2:
            continue

        if flg == 1:
            pfl(train_data)
            flg = 0

        if (
            word in cachedStopWords
            or word.title() in cachedStopWords
            or word.lower() in cachedStopWords
        ):
            continue

        lookup = word if word in d2 else word.lower()
        synonym = findsim(lookup)

        if not synonym or model is None:
            continue
        if word not in model:
            continue

        try:
            similarity = float(model.similarity(word, synonym))
        except (KeyError, TypeError, AttributeError):
            continue
        if similarity < 0.6:
            continue

        result[index] = synonym
        if "ing" in word:
            if result[index] + "ing" in model:
                result[index] += "ing"
            elif result[index][:-1] + "ing" in model:
                result[index] = result[index][:-1] + "ing"
        if "ed" in word:
            if result[index] + "ed" in model:
                result[index] += "ed"
            elif result[index][:-1] + "ed" in model:
                result[index] = result[index][:-1] + "ed"
        replacements += 1

    return " ".join(result), replacements


def check(sentence_a: str, sentence_b: str, data: Sequence[Sequence[Any]]) -> bool:
    """Return whether a sentence pair is absent in either orientation."""
    return not any(
        (sentence_a == item[0] and sentence_b == item[1])
        or (sentence_a == item[1] and sentence_b == item[0])
        for item in data
    )


def expand(data: MutableSequence[list[Any]]) -> MutableSequence[list[Any]]:
    """Expand training data using the repository's synonym augmentation."""
    new_pairs: list[list[Any]] = []
    for _ in range(10):
        for item in list(data):
            sentence_a, count_a = chsyn(item[0], data)
            sentence_b, count_b = chsyn(item[1], data)
            if count_a > 0 and count_b > 0:
                new_pairs.append([sentence_a, sentence_b, item[2]])

    for item in new_pairs:
        if check(item[0], item[1], data):
            data.append(item)
    return data


# ---------------------------------------------------------------------------
# PyTorch Siamese model compatibility surface
# ---------------------------------------------------------------------------


class SiameseLSTMModel(_MaLSTM):
    """Expose the shared-weight MaLSTM model used by the reproduction.

    There is intentionally one ``encoder`` object.  Both sentence branches
    call that same encoder, so its ``U``, ``W``, and ``b`` parameters are
    exactly shared rather than merely initialized to equal values.
    """

    def forward(
        self,
        embeddings_a: Any,
        mask_a: Any,
        embeddings_b: Any,
        mask_b: Any,
        target: Any | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:
        """Compute representations, similarity, and optional MSE loss."""
        import torch

        if not isinstance(embeddings_a, torch.Tensor):
            embeddings_a = torch.as_tensor(embeddings_a, dtype=torch.float32)
        if not isinstance(mask_a, torch.Tensor):
            mask_a = torch.as_tensor(mask_a, dtype=torch.float32)
        if not isinstance(embeddings_b, torch.Tensor):
            embeddings_b = torch.as_tensor(embeddings_b, dtype=torch.float32)
        if not isinstance(mask_b, torch.Tensor):
            mask_b = torch.as_tensor(mask_b, dtype=torch.float32)

        representation_a = self.encode(embeddings_a, mask_a)
        representation_b = self.encode(embeddings_b, mask_b)
        distance = torch.sum(
            torch.abs(representation_a - representation_b),
            dim=1,
        )
        similarity = torch.clamp(
            torch.exp(-distance),
            min=1e-7,
            max=1.0 - 1e-7,
        )

        loss = None
        if target is not None:
            if not isinstance(target, torch.Tensor):
                target = torch.as_tensor(target, dtype=torch.float32)
            target = torch.clamp((target - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)
            loss = torch.mean((similarity - target) ** 2)

        return representation_a, representation_b, similarity, loss


# ---------------------------------------------------------------------------
# Legacy function names
# ---------------------------------------------------------------------------


def dropout_layer(
    state_before: Any,
    use_noise_value: Any,
    rrng: Any,
    dropout_rate: float,
) -> Any:
    """Compatibility helper for the original optional dropout operation."""
    if hasattr(use_noise_value, "get_value"):
        enabled = bool(use_noise_value.get_value())
    else:
        enabled = bool(use_noise_value)

    if enabled:
        return state_before * rrng
    return state_before * (1.0 - dropout_rate)


def getpl2(
    prevlayer: Any,
    pre: str,
    mymask: Any,
    used: bool,
    rrng: Any,
    size: int,
    tnewp: Mapping[str, Any],
) -> Any:
    """Forward through the explicit LSTM while retaining the old API."""
    projection = _getpl2(
        prevlayer,
        pre,
        mymask,
        used,
        rrng,
        size,
        tnewp,
    )
    return projection


def get_layer(name: str) -> Any:
    """Return a legacy layer descriptor by name."""
    return _get_layer(name)


# Keep the original helper names available from wildcard imports.
layers: dict[str, Any] = {}


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "HIDDEN_DIM",
    "INPUT_DIM",
    "SiameseLSTMModel",
    "cachedStopWords",
    "chsyn",
    "check",
    "configure_resources",
    "creatrnnx",
    "d2",
    "dtr",
    "dropout_layer",
    "embed",
    "expand",
    "findsim",
    "flg",
    "get_layer",
    "getlayerx",
    "getmtr",
    "getpl2",
    "init_tparams",
    "lstm_layer2",
    "meas",
    "model",
    "noise_std",
    "numpy",
    "numpy_floatX",
    "pfl",
    "prepare_data",
    "prefix",
    "rate",
    "training",
    "unzip",
    "use_noise",
    "zipp",
]


# ``numpy`` was a module-level name in the original file and is intentionally
# exported for compatibility with old ``from SiameseLSTM import *`` callers.
numpy = np
