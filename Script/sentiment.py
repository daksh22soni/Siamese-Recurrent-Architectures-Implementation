"""SICK entailment transfer experiment using fixed MaLSTM representations.

This module reproduces the transfer experiment described by Mueller and
Thyagarajan (AAAI 2016): a trained MaLSTM encoder is kept fixed, sentence
pairs are represented with the element-wise absolute difference and
 element-wise product of their 50-dimensional final hidden states, and an
RBF-kernel SVM performs the three-way entailment classification.

The supplied legacy ``sentiment.py`` in the original repository instead
contains a separate Kaggle sentiment experiment.  The paper's entailment
experiment is the authoritative behavior requested for this reproduction.
The legacy SVM settings exposed by that repository (C=100, gamma=3.1,
RBF kernel) are retained as the default classifier configuration.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.svm import SVC

from lstm import HIDDEN_DIM, INPUT_DIM, lstm


# ---------------------------------------------------------------------------
# Reproduction constants
# ---------------------------------------------------------------------------

FEATURE_DIM = 2 * HIDDEN_DIM
DEFAULT_C = 100.0
DEFAULT_GAMMA = 3.1
DEFAULT_KERNEL = "rbf"
DEFAULT_BATCH_SIZE = 64

SICK_TRAIN_FILENAME = "semtrain.p"
SICK_TEST_FILENAME = "semtest.p"
CHECKPOINT_FILENAME = "bestsem.p"

# SICK label position in the original repository's records:
# [sentence_a, sentence_b, relatedness, entailment_label, ...]
SENTENCE_A_INDEX = 0
SENTENCE_B_INDEX = 1
ENTAILMENT_LABEL_INDEX = 3


@dataclass(frozen=True)
class EntailmentResult:
    """Container for transfer-classification results.

    Attributes
    ----------
    accuracy:
        Classification accuracy as a percentage.
    predictions:
        Predicted entailment labels in input order.
    labels:
        Gold entailment labels in input order.
    features:
        Fixed 100-dimensional transfer features used by the SVM.
    """

    accuracy: float
    predictions: np.ndarray
    labels: np.ndarray
    features: np.ndarray


def _load_pickle(path: Path) -> Any:
    """Load an original Python 2-era pickle under Python 3.10."""
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except (UnicodeDecodeError, ValueError, EOFError):
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _normalise_label(label: Any) -> str:
    """Convert an SICK entailment label to a stable Python string."""
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def _records_to_pairs(
    data: Sequence[Sequence[Any]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Extract sentence pairs and entailment labels from SICK records."""
    sentence_a: list[str] = []
    sentence_b: list[str] = []
    labels: list[str] = []

    for record in data:
        if len(record) <= ENTAILMENT_LABEL_INDEX:
            raise ValueError(
                "Each SICK record must contain sentence A, sentence B, "
                "relatedness, and entailment label."
            )
        sentence_a.append(str(record[SENTENCE_A_INDEX]))
        sentence_b.append(str(record[SENTENCE_B_INDEX]))
        labels.append(_normalise_label(record[ENTAILMENT_LABEL_INDEX]))

    return sentence_a, sentence_b, np.asarray(labels, dtype=object)


def _sentence_vectors(
    encoder: lstm,
    sentences: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Encode sentences as fixed 50-D final hidden-state vectors.

    The existing ``lstm`` compatibility wrapper is used only as the model
    interface.  Its underlying explicit PyTorch encoder is called once per
    batch and the final hidden state ``h_T`` is extracted exactly as required
    by the paper.
    """
    if not sentences:
        return np.empty((0, HIDDEN_DIM), dtype=np.float32)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    from sentences import embed, getmtr

    representations: list[np.ndarray] = []

    for start in range(0, len(sentences), batch_size):
        batch = list(sentences[start : start + batch_size])
        max_length = max(len(sentence.split()) for sentence in batch)
        tokens, mask = getmtr(batch, max_length)

        vectors = np.stack([embed(row) for row in tokens], axis=0)
        # sentences.embed returns [batch, time, 300]; the explicit LSTM uses
        # [time, batch, 300], matching the original Theano state layout.
        time_major = np.swapaxes(vectors, 0, 1).astype(np.float32, copy=False)

        import torch

        with torch.no_grad():
            embedding_tensor = torch.as_tensor(
                time_major,
                dtype=torch.float32,
                device=encoder.device,
            )
            mask_tensor = torch.as_tensor(
                mask,
                dtype=torch.float32,
                device=encoder.device,
            )
            hidden_states = encoder.model.encoder(embedding_tensor, mask_tensor)
            final_hidden = hidden_states[-1]

        representations.append(final_hidden.detach().cpu().numpy())

    result = np.concatenate(representations, axis=0)
    if result.shape[1] != HIDDEN_DIM:
        raise RuntimeError(
            f"Expected {HIDDEN_DIM}-D sentence representations, "
            f"got {result.shape[1]}-D."
        )
    return result.astype(np.float32, copy=False)


def build_transfer_features(
    representation_a: np.ndarray,
    representation_b: np.ndarray,
) -> np.ndarray:
    """Construct the paper's 100-D entailment feature representation.

    Features are concatenated in the order ``|hA-hB|`` followed by
    ``hA*hB``.  With 50-D sentence representations this produces exactly
    100 features per sentence pair.
    """
    first = np.asarray(representation_a, dtype=np.float32)
    second = np.asarray(representation_b, dtype=np.float32)

    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Sentence representations must be two-dimensional.")
    if first.shape != second.shape:
        raise ValueError(
            "Both sentence representation matrices must have identical shapes."
        )
    if first.shape[1] != HIDDEN_DIM:
        raise ValueError(
            f"Expected {HIDDEN_DIM}-D sentence vectors, got {first.shape[1]}-D."
        )

    absolute_difference = np.abs(first - second)
    elementwise_product = first * second
    features = np.concatenate(
        (absolute_difference, elementwise_product),
        axis=1,
    )

    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FEATURE_DIM} transfer features, got {features.shape[1]}."
        )
    return features.astype(np.float32, copy=False)


def prepare_data2(
    data: Sequence[Sequence[Any]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Extract the SICK sentence-pair inputs and entailment labels.

    This name is retained as a compatibility-oriented helper from the legacy
    experiment module while its output now matches the paper's entailment task.
    """
    return _records_to_pairs(data)


def getmtr(
    sentences: Sequence[str],
    maxlen: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the original token matrix and time-major mask.

    This helper mirrors ``sentences.getmtr`` for callers that historically
    imported it from ``sentiment.py``.
    """
    from sentences import getmtr as sentence_getmtr

    return sentence_getmtr(sentences, maxlen)


def fpro(
    data: Sequence[Sequence[Any]],
    encoder: lstm,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed transfer features and entailment labels."""
    sentence_a, sentence_b, labels = _records_to_pairs(data)
    vectors_a = _sentence_vectors(encoder, sentence_a, batch_size)
    vectors_b = _sentence_vectors(encoder, sentence_b, batch_size)
    features = build_transfer_features(vectors_a, vectors_b)
    return features, labels


def getacc(
    classifier: SVC,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return classification accuracy as a percentage."""
    predictions = classifier.predict(features)
    return float(np.mean(predictions == labels) * 100.0)


class EntailmentSVM:
    """RBF-kernel SVM operating on fixed MaLSTM transfer features."""

    def __init__(
        self,
        c: float = DEFAULT_C,
        gamma: float = DEFAULT_GAMMA,
        kernel: str = DEFAULT_KERNEL,
    ) -> None:
        """Initialize the classifier with the repository-compatible SVM setup."""
        if kernel != DEFAULT_KERNEL:
            raise ValueError(
                f"This reproduction requires the RBF kernel; got {kernel!r}."
            )
        if c <= 0:
            raise ValueError("SVM C must be positive.")
        if gamma <= 0:
            raise ValueError("SVM gamma must be positive.")

        self.classifier = SVC(C=c, gamma=gamma, kernel=kernel)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EntailmentSVM":
        """Fit the RBF SVM without modifying the learned LSTM representation."""
        features_array = np.asarray(features, dtype=np.float32)
        labels_array = np.asarray(labels)
        if features_array.ndim != 2 or features_array.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected feature matrix [n, {FEATURE_DIM}], "
                f"got {features_array.shape}."
            )
        self.classifier.fit(features_array, labels_array)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict SICK entailment labels."""
        features_array = np.asarray(features, dtype=np.float32)
        if features_array.ndim != 2 or features_array.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected feature matrix [n, {FEATURE_DIM}], "
                f"got {features_array.shape}."
            )
        return self.classifier.predict(features_array)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Return accuracy as a percentage."""
        return float(self.classifier.score(features, labels) * 100.0)


def run_entailment_experiment(
    train_data: Sequence[Sequence[Any]],
    test_data: Sequence[Sequence[Any]],
    encoder: lstm,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    c: float = DEFAULT_C,
    gamma: float = DEFAULT_GAMMA,
) -> EntailmentResult:
    """Run fixed-representation SICK entailment classification."""
    train_features, train_labels = fpro(train_data, encoder, batch_size)
    test_features, test_labels = fpro(test_data, encoder, batch_size)

    classifier = EntailmentSVM(c=c, gamma=gamma)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = float(np.mean(predictions == test_labels) * 100.0)

    return EntailmentResult(
        accuracy=accuracy,
        predictions=predictions,
        labels=test_labels,
        features=test_features,
    )


def load_sick_data(
    train_path: str | Path = SICK_TRAIN_FILENAME,
    test_path: str | Path = SICK_TEST_FILENAME,
) -> tuple[Any, Any]:
    """Load the original SICK train/test pickle resources."""
    train_data = _load_pickle(Path(train_path))
    test_data = _load_pickle(Path(test_path))
    return train_data, test_data


def load_encoder(
    checkpoint_path: str | Path = CHECKPOINT_FILENAME,
) -> lstm:
    """Load the trained MaLSTM checkpoint used for transfer learning."""
    model = lstm(str(checkpoint_path), load=True, training=False)
    model._sync_mode(training=False)
    return model


def main(
    checkpoint_path: str | Path = CHECKPOINT_FILENAME,
    train_path: str | Path = SICK_TRAIN_FILENAME,
    test_path: str | Path = SICK_TEST_FILENAME,
) -> float:
    """Run the complete fixed-representation entailment experiment."""
    encoder = load_encoder(checkpoint_path)
    train_data, test_data = load_sick_data(train_path, test_path)
    result = run_entailment_experiment(train_data, test_data, encoder)
    print(f"Test accuracy: {result.accuracy:.2f}%")
    return result.accuracy


__all__ = [
    "DEFAULT_C",
    "DEFAULT_GAMMA",
    "DEFAULT_KERNEL",
    "FEATURE_DIM",
    "EntailmentResult",
    "EntailmentSVM",
    "build_transfer_features",
    "fpro",
    "getacc",
    "getmtr",
    "load_encoder",
    "load_sick_data",
    "main",
    "prepare_data2",
    "run_entailment_experiment",
]


if __name__ == "__main__":
    main()
