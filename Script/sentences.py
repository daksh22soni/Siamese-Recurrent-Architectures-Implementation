"""Sentence preprocessing and synonym augmentation for Siamese-LSTM.

This module is a Python 3.10 compatibility implementation of the supplied
Siamese-LSTM repository's ``sentences.py``.  The research methodology is kept
unchanged: sentence tokenization, padding, masking, Google News Word2Vec
lookup, and the ten-pass synonym augmentation algorithm remain repository
compatible.

The augmentation implementation deliberately does not force the paper's
reported 10,022-example count.  At runtime it reports the number of generated
candidates, the number accepted after duplicate filtering, and the resulting
training-set size so that the exact count can be verified with the intended
Google News Word2Vec resource.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Mapping, MutableSequence, Sequence

import numpy as np

try:
    from gensim.models import KeyedVectors
except ImportError as exc:  # pragma: no cover - environment dependent.
    raise ImportError(
        "Gensim is required for the Siamese-LSTM reproduction."
    ) from exc


# ---------------------------------------------------------------------------
# Reproduction constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 300
PADDING_TOKEN = ","
WORD2VEC_FILENAME = "GoogleNews-vectors-negative300.bin.gz"
SYNONYM_FILENAME = "synsem.p"
WORD_DICTIONARY_FILENAME = "dwords.p"
AUGMENTATION_PASSES = 10
WORD_SIMILARITY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Legacy module-level resources
# ---------------------------------------------------------------------------

model: Any = None
dtr: Mapping[str, Any] = {}
d2: Mapping[str, Any] = {}
flg = 1

# Runtime audit information.  This does not affect the augmentation result.
last_augmentation_stats: dict[str, int] = {
    "original_examples": 0,
    "generated_candidates": 0,
    "accepted_examples": 0,
    "final_examples": 0,
}

try:
    from nltk.corpus import stopwords

    cachedStopWords = set(stopwords.words("english"))
except (ImportError, LookupError):
    cachedStopWords: set[str] = set()


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------


def _load_pickle(path: Path) -> Any:
    """Load a legacy pickle using Python 3-compatible decoding.

    Parameters
    ----------
    path:
        Path to the legacy pickle resource.

    Returns
    -------
    Any
        Deserialized object.
    """
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except (UnicodeDecodeError, ValueError):
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _load_word2vec(path: Path) -> KeyedVectors:
    """Load a word2vec-format binary using the modern Gensim API."""
    return KeyedVectors.load_word2vec_format(str(path), binary=True)


def _load_default_resources(base_dir: Path) -> None:
    """Load legacy resources when expected files are available."""
    global d2, dtr, model

    synonym_path = base_dir / SYNONYM_FILENAME
    dictionary_path = base_dir / WORD_DICTIONARY_FILENAME
    word2vec_path = base_dir / WORD2VEC_FILENAME

    if synonym_path.exists():
        d2 = _load_pickle(synonym_path)
    if dictionary_path.exists():
        dtr = _load_pickle(dictionary_path)
    if word2vec_path.exists():
        model = _load_word2vec(word2vec_path)


# Do not force the multi-gigabyte Google News model to be present merely to
# import this module.  main.py configures resources explicitly for execution.
_load_default_resources(Path.cwd())


# ---------------------------------------------------------------------------
# Sentence batching
# ---------------------------------------------------------------------------


def prepare_data(
    data: Sequence[Sequence[Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare paired sentences using the original batch representation.

    Each record is expected to contain ``[sentence_a, sentence_b, score]``.
    Tokens are padded with ``,`` and masks have shape ``(time, batch)``.
    Labels remain on the original [1, 5] scale.
    """
    sentence_a: list[Any] = []
    sentence_b: list[Any] = []
    labels: list[Any] = []

    for item in data:
        sentence_a.append(item[0])
        sentence_b.append(item[1])
        labels.append(item[2])

    lengths = [len(str(sentence).split()) for sentence in sentence_a]
    lengths.extend(len(str(sentence).split()) for sentence in sentence_b)
    if not lengths:
        raise ValueError("Cannot prepare an empty sentence-pair dataset.")

    max_length = int(np.max(lengths))
    embeddings_a, mask_a = getmtr(sentence_a, max_length)
    embeddings_b, mask_b = getmtr(sentence_b, max_length)
    labels_array = np.asarray(labels, dtype=np.float32)

    return embeddings_a, mask_a, embeddings_b, mask_b, labels_array


def getmtr(
    xa: Sequence[str],
    maxlen: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the original padded token matrix and time-major mask."""
    n_samples = len(xa)
    token_rows: list[list[str]] = []
    x_mask = np.zeros((maxlen, n_samples), dtype=np.float32)

    for sample_index, sentence in enumerate(xa):
        tokens = str(sentence).split()
        for token_index in range(len(tokens)):
            x_mask[token_index, sample_index] = 1.0
        while len(tokens) < maxlen:
            tokens.append(PADDING_TOKEN)
        token_rows.append(tokens)

    return np.asarray(token_rows), x_mask


# ---------------------------------------------------------------------------
# Word-vector conversion
# ---------------------------------------------------------------------------


def embed(stmx: np.ndarray) -> np.ndarray:
    """Convert one padded token sequence into 300-D Word2Vec vectors.

    Padding remains a zero vector.  Non-padding vocabulary misses raise
    ``KeyError`` rather than being silently mapped to an unknown vector.
    """
    if model is None:
        raise RuntimeError(
            "Word2Vec model is not configured. Configure it before calling "
            "embed()."
        )

    tokens = np.asarray(stmx)
    if tokens.ndim != 1:
        raise ValueError("embed() expects a one-dimensional token sequence.")

    vectors = np.zeros((tokens.shape[0], EMBEDDING_DIM), dtype=np.float32)

    for index, token in enumerate(tokens):
        token_text = str(token)
        if token_text == PADDING_TOKEN:
            continue

        lookup_key = dtr[token_text] if token_text in dtr else token_text
        try:
            vectors[index] = np.asarray(model[lookup_key], dtype=np.float32)
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise KeyError(
                f"Word2Vec vocabulary entry not found: {lookup_key!r}"
            ) from exc

    return vectors


# ---------------------------------------------------------------------------
# Synonym augmentation
# ---------------------------------------------------------------------------


def chsyn(sentence: str, trn: MutableSequence[Any]) -> tuple[str, int]:
    """Apply the original repository's synonym replacement procedure.

    The algorithm intentionally preserves the supplied repository behavior:

    * only tokens present in ``d2`` reach synonym selection;
    * stop words are skipped;
    * one random synonym is selected from ``synsem.p``;
    * the candidate must exist in the Word2Vec vocabulary;
    * Word2Vec similarity must be at least 0.6;
    * ``ing``/``ed`` restoration follows the original string logic;
    * multiple eligible words in a sentence may be replaced in one pass.

    ``trn`` is retained because it is part of the original function API.  The
    supplied implementation does not otherwise use it inside this function.
    """
    del trn

    count = 0
    words = list(str(sentence).split())

    for index in range(len(words)):
        word = words[index]
        synonym = ""

        # This ordering is intentional.  In the supplied repository, the
        # initial membership test means the later lower-case branch is only
        # reachable when the exact token is already in d2.
        if word not in d2:
            continue

        if (
            word in cachedStopWords
            or word.title() in cachedStopWords
            or word.lower() in cachedStopWords
        ):
            continue

        if word in d2 or word.lower() in d2:
            if word in d2:
                synonym = findsim(word)
            elif word.lower() in d2:
                synonym = findsim(word)

            if model is None:
                raise RuntimeError(
                    "Word2Vec model is required for synonym augmentation."
                )

            if word not in model:
                synonym = ""
                continue

        if model is None:
            raise RuntimeError(
                "Word2Vec model is required for synonym augmentation."
            )

        if synonym in model:
            if word == synonym:
                synonym = ""
                continue

            if float(model.similarity(word, synonym)) < WORD_SIMILARITY_THRESHOLD:
                continue

            words[index] = synonym

            # Preserve the original repository's morphological restoration
            # behavior exactly, including its substring-based checks.
            if "ing" in word:
                if words[index] + "ing" in model:
                    words[index] += "ing"
                if words[index][:-1] + "ing" in model:
                    words[index] = words[index][:-1] + "ing"

            if "ed" in word:
                if words[index] + "ed" in model:
                    words[index] += "ed"
                if words[index][:-1] + "ed" in model:
                    words[index] = words[index][:-1] + "ed"

            count += 1
            synonym = ""

    return " ".join(words), count


def findsim(word: str) -> str:
    """Select one random synonym from ``synsem.p`` using repository logic."""
    synonyms = d2[word]
    if not synonyms:
        return ""
    index = random.randint(0, len(synonyms) - 1)
    return synonyms[index]


def check(
    sentence_a: str,
    sentence_b: str,
    data: Sequence[Sequence[Any]],
) -> bool:
    """Check whether a sentence pair is absent in either orientation."""
    for item in data:
        if sentence_a == item[0] and sentence_b == item[1]:
            return False
        if sentence_a == item[1] and sentence_b == item[0]:
            return False
    return True


def expand(data: MutableSequence[list[Any]]) -> MutableSequence[list[Any]]:
    """Perform the original ten-pass synonym augmentation.

    The supplied repository performs ten passes over the current input data,
    creates a candidate only when both sentences receive at least one
    replacement, and appends candidates that are not already present in the
    original input data.  This function preserves that behavior rather than
    forcing the paper-reported 10,022 examples.

    The original relatedness label is copied unchanged into every accepted
    augmented pair.

    Parameters
    ----------
    data:
        Mutable SICK training sequence.  The sequence is modified in place,
        matching the original repository.

    Returns
    -------
    MutableSequence[list[Any]]
        The same sequence object after augmentation.
    """
    global last_augmentation_stats

    original_count = len(data)
    new_examples: list[list[Any]] = []

    for _ in range(AUGMENTATION_PASSES):
        for item in data:
            sentence_a, count_a = chsyn(item[0], data)
            sentence_b, count_b = chsyn(item[1], data)

            # The original repository requires both sentences to have at least
            # one replacement before creating an augmented pair.
            if count_a > 0 and count_b > 0:
                new_examples.append([sentence_a, sentence_b, item[2]])

    generated_candidates = len(new_examples)
    accepted_examples = 0

    for item in new_examples:
        if check(item[0], item[1], data):
            data.append(item)
            accepted_examples += 1

    final_count = len(data)

    last_augmentation_stats = {
        "original_examples": original_count,
        "generated_candidates": generated_candidates,
        "accepted_examples": accepted_examples,
        "final_examples": final_count,
    }

    print(
        "Synonym augmentation: "
        f"original={original_count}, "
        f"generated_candidates={generated_candidates}, "
        f"accepted={accepted_examples}, "
        f"total={final_count}"
    )

    return data


def configure_resources(
    word2vec_model: Any,
    word_dictionary: Mapping[str, Any] | None = None,
    synonym_dictionary: Mapping[str, Any] | None = None,
    stop_words: Sequence[str] | None = None,
) -> None:
    """Configure Word2Vec and legacy lexical resources explicitly.

    Parameters
    ----------
    word2vec_model:
        Gensim ``KeyedVectors`` or compatible embedding interface.
    word_dictionary:
        ``dwords.p`` mapping used for token lookup normalization.
    synonym_dictionary:
        ``synsem.p`` mapping used by synonym augmentation.
    stop_words:
        Optional stop-word collection.
    """
    global cachedStopWords, d2, dtr, model

    model = word2vec_model
    dtr = word_dictionary or {}
    d2 = synonym_dictionary or {}
    if stop_words is not None:
        cachedStopWords = set(stop_words)


__all__ = [
    "AUGMENTATION_PASSES",
    "EMBEDDING_DIM",
    "PADDING_TOKEN",
    "WORD2VEC_FILENAME",
    "SYNONYM_FILENAME",
    "WORD_DICTIONARY_FILENAME",
    "WORD_SIMILARITY_THRESHOLD",
    "cachedStopWords",
    "check",
    "chsyn",
    "configure_resources",
    "d2",
    "dtr",
    "embed",
    "expand",
    "findsim",
    "flg",
    "getmtr",
    "last_augmentation_stats",
    "model",
    "prepare_data",
]
