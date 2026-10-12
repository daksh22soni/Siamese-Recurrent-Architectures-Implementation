"""Compatibility adapter for the original Google News Word2Vec vectors.

The original Siamese-LSTM repository loads the 300-dimensional Google News
Word2Vec binary file through Gensim's legacy ``Word2Vec.load_word2vec_format``
API. Modern Gensim exposes the same word2vec-format loader through
``KeyedVectors``. This module provides a deliberately small adapter around
that interface without changing the embedding model or vector values.

Expected source file
--------------------
``GoogleNews-vectors-negative300.bin.gz``

The adapter does not train, transform, normalize, replace, or otherwise alter
the embeddings. It only loads and exposes the operations required by the
original repository: vocabulary membership, vector lookup, dimensionality,
and pairwise word similarity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from gensim.models import KeyedVectors


GOOGLE_NEWS_FILENAME = "GoogleNews-vectors-negative300.bin.gz"
EXPECTED_VECTOR_SIZE = 300
DEFAULT_BINARY = True

PathLike = Union[str, Path]


class GoogleNewsWordVectors:
    """Minimal compatibility wrapper for Google News Word2Vec vectors.

    Parameters
    ----------
    keyed_vectors:
        A Gensim ``KeyedVectors`` instance containing the original Google News
        300-dimensional Word2Vec embeddings.

    Raises
    ------
    ValueError
        If the supplied vectors are not 300-dimensional.
    TypeError
        If ``keyed_vectors`` does not expose the required Gensim vector API.
    """

    def __init__(self, keyed_vectors: KeyedVectors) -> None:
        if not isinstance(keyed_vectors, KeyedVectors):
            raise TypeError(
                "keyed_vectors must be a gensim.models.KeyedVectors instance."
            )

        vector_size = int(keyed_vectors.vector_size)
        if vector_size != EXPECTED_VECTOR_SIZE:
            raise ValueError(
                "The reproduction requires 300-dimensional Google News "
                f"vectors; received vector_size={vector_size}."
            )

        self._vectors = keyed_vectors

    @property
    def vector_size(self) -> int:
        """Return the embedding dimensionality."""

        return int(self._vectors.vector_size)

    @property
    def key_to_index(self) -> dict[str, int]:
        """Return the Gensim vocabulary mapping."""

        return self._vectors.key_to_index

    @property
    def vectors(self) -> np.ndarray:
        """Return the underlying embedding matrix without modification."""

        return self._vectors.vectors

    def __contains__(self, word: object) -> bool:
        """Return whether ``word`` exists in the original vocabulary."""

        return word in self._vectors.key_to_index

    def __getitem__(self, word: str) -> np.ndarray:
        """Return the original vector for ``word``.

        ``KeyError`` is intentionally preserved for unknown words, matching
        the original repository's direct ``model[word]`` behavior.
        """

        return self._vectors[word]

    def get_vector(self, word: str) -> np.ndarray:
        """Return the original vector for ``word``."""

        return self._vectors.get_vector(word)

    def similarity(self, word_a: str, word_b: str) -> float:
        """Return Gensim's cosine similarity between two vocabulary words.

        This operation is used only by the repository's synonym-augmentation
        logic. It does not alter the Siamese-LSTM sentence similarity function,
        which remains Manhattan distance followed by ``exp(-L1)``.
        """

        return float(self._vectors.similarity(word_a, word_b))

    def save(self, path: PathLike) -> None:
        """Save the underlying Gensim ``KeyedVectors`` object unchanged."""

        self._vectors.save(str(path))

    @classmethod
    def load(cls, path: PathLike) -> "GoogleNewsWordVectors":
        """Load a Gensim ``KeyedVectors`` file and validate its dimension."""

        return cls(KeyedVectors.load(str(path), mmap="r"))

    @classmethod
    def load_word2vec_format(
        cls,
        path: PathLike,
        *,
        binary: bool = DEFAULT_BINARY,
        limit: Optional[int] = None,
    ) -> "GoogleNewsWordVectors":
        """Load vectors from the original word2vec-format file.

        Parameters
        ----------
        path:
            Path to ``GoogleNews-vectors-negative300.bin.gz``.
        binary:
            Whether the source is in binary word2vec format. The original
            repository uses ``binary=True`` and this remains the default.
        limit:
            Optional Gensim vocabulary limit. ``None`` loads the complete
            source vocabulary, as in the original repository.
        """

        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Google News Word2Vec file was not found: {source_path}"
            )

        keyed_vectors = KeyedVectors.load_word2vec_format(
            str(source_path),
            binary=binary,
            limit=limit,
        )
        return cls(keyed_vectors)


def load_google_news_word_vectors(
    path: PathLike = GOOGLE_NEWS_FILENAME,
    *,
    limit: Optional[int] = None,
) -> GoogleNewsWordVectors:
    """Load the original 300-D Google News Word2Vec embeddings.

    The loader intentionally uses the modern Gensim ``KeyedVectors`` API while
    preserving the original repository's binary word2vec source file and
    300-dimensional representation.
    """

    return GoogleNewsWordVectors.load_word2vec_format(
        path,
        binary=DEFAULT_BINARY,
        limit=limit,
    )


__all__ = [
    "EXPECTED_VECTOR_SIZE",
    "GOOGLE_NEWS_FILENAME",
    "GoogleNewsWordVectors",
    "load_google_news_word_vectors",
]
