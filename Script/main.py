"""Entry point for the faithful Siamese-LSTM reproduction.

The training pipeline is intentionally kept explicit and close to the
original repository:

    fresh model -> STS pre-training -> SICK synonym augmentation
    -> SICK training -> SICK test evaluation

``bestsem.p`` is never loaded before STS pre-training.  It remains available
for evaluation-only execution through ``--no-training``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from calibration import (
    LocalLinearCalibrator,
    calculate_calibrated_metrics,
)
from mainmodel import MainModel, chkterr2, get_model, train_lstm
from sentences import configure_resources
from word_vectors import load_google_news_word_vectors


CHECKPOINT_FILENAME = "bestsem.p"
STS_FILENAME = "stsallrmf.p"
SICK_TRAIN_FILENAME = "semtrain.p"
SICK_TEST_FILENAME = "semtest.p"
WORD2VEC_FILENAMES = (
    "GoogleNews-vectors-negative300.bin.gz",
    "GoogleNews-vectors-negative300.bin",
)
SYNONYM_FILENAME = "synsem.p"
WORD_DICTIONARY_FILENAME = "dwords.p"

STS_EXPECTED_RECORDS = 11_219
SICK_TRAIN_EXPECTED_RECORDS = 5_000
SICK_TEST_EXPECTED_RECORDS = 4_927

STS_EPOCHS = 66
SICK_AUGMENTED_EPOCHS = 375
SICK_STANDARD_EPOCHS = 330

training = True
Syn_aug = True

console = Console()


def _load_pickle(path: Path) -> Any:
    """Load an original Python 2/3 pickle with compatibility decoding."""
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except (UnicodeDecodeError, ValueError):
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _require_file(path: Path, description: str) -> Path:
    """Return a required file path or raise a descriptive error."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required {description} was not found: {path}"
        )
    return path


def _resolve_word2vec_path(base_dir: Path) -> Path:
    """Resolve the original Google News binary in compressed or raw form."""
    for filename in WORD2VEC_FILENAMES:
        candidate = base_dir / filename
        if candidate.is_file():
            return candidate
    names = ", ".join(WORD2VEC_FILENAMES)
    raise FileNotFoundError(
        f"Google News Word2Vec file not found. Expected one of: {names}"
    )


def load_resources(base_dir: Path) -> None:
    """Load the original Word2Vec and lexical resources."""
    word2vec_path = _resolve_word2vec_path(base_dir)
    synonym_path = _require_file(
        base_dir / SYNONYM_FILENAME,
        "synonym dictionary",
    )
    dictionary_path = _require_file(
        base_dir / WORD_DICTIONARY_FILENAME,
        "word dictionary",
    )

    console.print("[bold]Loading Google News Word2Vec...[/bold]")
    vectors = load_google_news_word_vectors(word2vec_path)

    if vectors.vector_size != 300:
        raise RuntimeError(
            "The reproduction requires 300-dimensional Google News vectors."
        )

    synonym_dictionary = _load_pickle(synonym_path)
    word_dictionary = _load_pickle(dictionary_path)

    configure_resources(
        word2vec_model=vectors,
        word_dictionary=word_dictionary,
        synonym_dictionary=synonym_dictionary,
    )

    console.print(
        f"Loaded Google News Word2Vec: {vectors.vector_size}-D vectors"
    )
    console.print(
        f"Loaded synonym mappings: {len(synonym_dictionary):,}"
    )
    console.print(
        f"Loaded word dictionary entries: {len(word_dictionary):,}"
    )


def load_dataset(path: Path, description: str) -> Sequence[Sequence[Any]]:
    """Load one original repository dataset without changing its records."""
    _require_file(path, description)
    data = _load_pickle(path)
    if not isinstance(data, Sequence):
        raise TypeError(f"{description} must deserialize to a sequence")
    return data


def _verify_dataset_count(
    data: Sequence[Sequence[Any]],
    expected: int,
    description: str,
) -> None:
    """Verify an original serialized dataset has its audited record count."""
    actual = len(data)
    if actual != expected:
        raise ValueError(
            f"{description} count mismatch: expected {expected}, got {actual}"
        )


def _parameter_snapshot(model: MainModel) -> dict[str, Any]:
    """Capture model parameters for a lightweight update verification."""
    return {
        name: parameter.detach().clone()
        for name, parameter in model.model.named_parameters()
    }


def _verify_parameter_update(
    before: dict[str, Any],
    model: MainModel,
) -> None:
    """Confirm that at least one trainable model parameter changed."""
    changed = False
    for name, parameter in model.model.named_parameters():
        previous = before.get(name)
        if previous is None:
            raise RuntimeError(
                f"Training parameter set changed unexpectedly: {name}"
            )
        if not parameter.detach().equal(previous):
            changed = True
            break

    if not changed:
        raise RuntimeError(
            "STS training completed without changing any model parameter."
        )


def _raw_similarity_predictions(
    data: Sequence[Sequence[Any]],
    model: MainModel,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return raw [0, 1] MaLSTM predictions and original labels.

    The existing model preprocessing and embedding APIs are reused directly.
    No labels are consumed by the model prediction itself; labels are returned
    only so the caller can perform post-training evaluation.
    """
    import numpy as np

    if not data:
        raise ValueError("Cannot predict similarities for an empty dataset")

    predictions: list[float] = []
    targets: list[float] = []

    model._sync_mode(training=False)
    for start in range(0, len(data), 256):
        batch = data[start : start + 256]
        x1, mask1, x2, mask2, labels = model._prepare_sentence_batch(batch)
        emb1 = model._batch_embeddings(x1, model._embed_matrix)
        emb2 = model._batch_embeddings(x2, model._embed_matrix)
        batch_predictions = model.f2sim(
            emb1,
            mask1,
            emb2,
            mask2,
        )
        predictions.extend(
            np.asarray(batch_predictions, dtype=np.float64)
            .reshape(-1)
            .tolist()
        )
        targets.extend(
            np.asarray(labels, dtype=np.float64)
            .reshape(-1)
            .tolist()
        )

    raw = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)

    if raw.shape != target.shape:
        raise RuntimeError(
            "Raw similarity and target arrays have inconsistent shapes: "
            f"{raw.shape} vs {target.shape}"
        )
    if raw.size != len(data):
        raise RuntimeError(
            "Raw similarity prediction count mismatch: "
            f"expected {len(data)}, got {raw.size}"
        )
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(target)):
        raise RuntimeError(
            "Raw evaluation produced NaN or infinite values"
        )
    if np.any((raw < 0.0) | (raw > 1.0)):
        raise RuntimeError(
            "MaLSTM raw similarity must remain within [0, 1]"
        )

    return raw, target


def _report_paper_calibration(
    sick_train: Sequence[Sequence[Any]],
    sick_test: Sequence[Sequence[Any]],
    model: MainModel,
    raw_metrics: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Fit calibration on SICK train data and evaluate SICK test data.

    The test labels are retained only for final metric calculation. They are
    never passed to the calibrator's ``fit`` or bandwidth-selection methods.
    """
    import numpy as np

    train_raw, train_targets = _raw_similarity_predictions(
        sick_train,
        model,
    )
    test_raw, test_targets = _raw_similarity_predictions(
        sick_test,
        model,
    )

    if train_raw.size != SICK_TRAIN_EXPECTED_RECORDS:
        raise RuntimeError(
            "Calibration training sample count mismatch: "
            f"expected {SICK_TRAIN_EXPECTED_RECORDS}, "
            f"got {train_raw.size}"
        )
    if test_raw.size != SICK_TEST_EXPECTED_RECORDS:
        raise RuntimeError(
            "Calibration test sample count mismatch: "
            f"expected {SICK_TEST_EXPECTED_RECORDS}, "
            f"got {test_raw.size}"
        )

    raw_predictions = test_raw * 4.0 + 1.0
    raw_mse, raw_pearson, raw_spearman = raw_metrics

    # Keep the established chkterr2() raw evaluation as the authoritative
    # reproduction metric, while independently extracting raw similarities
    # for the calibration stage.  This guards the integration against an
    # accidental change to the existing evaluation path.
    extracted_raw_metrics = calculate_calibrated_metrics(
        raw_predictions,
        test_targets,
    )
    if not np.allclose(
        extracted_raw_metrics.as_tuple(),
        raw_metrics,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            "Existing raw evaluation and extracted raw predictions disagree: "
            f"existing={raw_metrics}, "
            f"extracted={extracted_raw_metrics.as_tuple()}"
        )

    calibrator = LocalLinearCalibrator()
    calibrator.fit(
        train_raw,
        train_targets,
    )
    calibrated_predictions = calibrator.predict(test_raw)
    calibrated_metrics = calculate_calibrated_metrics(
        calibrated_predictions,
        test_targets,
    )

    console.print(
        "\n"
        "============================================================\n"
        "PAPER REGRESSION CALIBRATION\n"
        "============================================================"
    )
    console.print("\nRaw MaLSTM evaluation:")
    console.print(f"MSE: {raw_mse:.6f}")
    console.print(f"Pearson: {raw_pearson:.6f}")
    console.print(f"Spearman: {raw_spearman:.6f}")

    console.print("\nCalibration:")
    console.print("Method: Univariate local-linear regression")
    console.print("Covariate: Raw MaLSTM similarity")
    console.print("Target: SICK human relatedness [1,5]")
    console.print(f"Bandwidth: {calibrator.bandwidth:.6f}")
    console.print("Bandwidth selection: Leave-one-out CV")
    console.print(
        f"Calibration training samples: {train_raw.size:,}"
    )
    console.print(f"Test samples: {test_raw.size:,}")
    console.print(
        f"Raw prediction range: [{raw_predictions.min():.6f}, "
        f"{raw_predictions.max():.6f}]"
    )
    console.print(
        f"Calibrated prediction range: ["
        f"{calibrated_predictions.min():.6f}, "
        f"{calibrated_predictions.max():.6f}]"
    )

    console.print("\nCalibrated evaluation:")
    console.print(f"MSE: {calibrated_metrics.mse:.6f}")
    console.print(f"Pearson: {calibrated_metrics.pearson:.6f}")
    console.print(f"Spearman: {calibrated_metrics.spearman:.6f}")

    console.print(
        "Calibration metrics are best-effort reproduction metrics; "
        "they are not claimed to reproduce the paper unless independently verified."
    )

    return calibrated_metrics.as_tuple()


def run_experiment(
    base_dir: Path,
    *,
    model_path: Path | None = None,
    training_enabled: bool = training,
    synonym_augmentation: bool = Syn_aug,
) -> tuple[MainModel, tuple[float, float, float]]:
    """Run fresh STS → SICK training followed by test evaluation.

    Parameters
    ----------
    base_dir:
        Directory containing the original repository resources.
    model_path:
        Legacy checkpoint used only for evaluation-only mode.
    training_enabled:
        When true, create a fresh model and train it from STS onward.  When
        false, load the legacy checkpoint and evaluate it directly.
    synonym_augmentation:
        Apply the original synonym expansion before SICK training.

    Returns
    -------
    tuple[MainModel, tuple[float, float, float]]
        The final model and ``(MSE, Pearson, Spearman)`` metrics.
    """
    base_dir = base_dir.resolve()

    # Resource loading is common to both training and evaluation paths.
    load_resources(base_dir)

    checkpoint = model_path or (base_dir / CHECKPOINT_FILENAME)
    sick_calibration_train: Sequence[Sequence[Any]] | None = None

    if training_enabled:
        # CRITICAL: this is deliberately a fresh model.  bestsem.p is not
        # loaded before STS pre-training.
        console.print(
            Panel(
                "Fresh model initialization",
                title="Reproduction initialization",
            )
        )
        model = get_model(
            model_path=None,
            load=False,
            training=True,
        )

        # --------------------------------------------------------------
        # Phase 1: STS pre-training
        # --------------------------------------------------------------
        sts_train = load_dataset(
            base_dir / STS_FILENAME,
            "STS training dataset",
        )
        _verify_dataset_count(
            sts_train,
            STS_EXPECTED_RECORDS,
            "STS training dataset",
        )
        console.print(
            f"STS records: {len(sts_train):,} | epochs: {STS_EPOCHS}"
        )

        parameters_before_sts = _parameter_snapshot(model)
        train_lstm(sts_train, STS_EPOCHS, model)
        _verify_parameter_update(parameters_before_sts, model)
        console.print("STS pre-training done; model parameters updated")

        # --------------------------------------------------------------
        # Phase 2: SICK loading and synonym augmentation
        # --------------------------------------------------------------
        sick_train = load_dataset(
            base_dir / SICK_TRAIN_FILENAME,
            "SICK training dataset",
        )
        _verify_dataset_count(
            sick_train,
            SICK_TRAIN_EXPECTED_RECORDS,
            "SICK training dataset",
        )
        console.print(
            f"SICK original training records: {len(sick_train):,}"
        )
        # Keep an independent copy of the original 5,000 SICK examples.
        # sentences.expand() mutates its input list in place, so assigning
        # sick_train directly here would also append augmented examples to
        # the calibration data and violate the training-split count check.
        sick_calibration_train = [list(record) for record in sick_train]

        if synonym_augmentation:
            from sentences import expand

            console.print(
                Panel(
                    "SICK synonym augmentation",
                    title="Phase 2",
                )
            )
            augmented_train = expand(sick_train)

            # The updated sentences.py records exact augmentation statistics.
            # Do not force the paper's 10,022 count: verify and report what the
            # original algorithm actually generated with the supplied resources.
            try:
                from sentences import last_augmentation_stats

                stats = last_augmentation_stats
                console.print(
                    "Augmentation result: "
                    f"original={stats['original_examples']:,}, "
                    f"generated_candidates="
                    f"{stats['generated_candidates']:,}, "
                    f"accepted={stats['accepted_examples']:,}, "
                    f"total={stats['final_examples']:,}"
                )
            except (ImportError, KeyError, TypeError):
                console.print(
                    f"Augmented SICK training records: "
                    f"{len(augmented_train):,}"
                )

            sick_train = augmented_train
            sick_epochs = SICK_AUGMENTED_EPOCHS
        else:
            sick_epochs = SICK_STANDARD_EPOCHS

        # --------------------------------------------------------------
        # Phase 3: SICK training
        # --------------------------------------------------------------
        console.print(
            Panel(
                f"SICK training ({sick_epochs} epochs)",
                title="Phase 3",
            )
        )
        train_lstm(sick_train, sick_epochs, model)

    else:
        # Checkpoint loading is intentionally restricted to evaluation-only
        # mode.  It is never inserted into the fresh STS training path.
        checkpoint = _require_file(checkpoint, "model checkpoint")
        console.print("[bold]Loading Siamese-LSTM checkpoint...[/bold]")
        model = get_model(
            model_path=checkpoint,
            load=True,
            training=False,
        )

    # --------------------------------------------------------------
    # Phase 4: SICK test evaluation and paper regression calibration
    # --------------------------------------------------------------
    if sick_calibration_train is None:
        sick_calibration_train = load_dataset(
            base_dir / SICK_TRAIN_FILENAME,
            "SICK calibration training dataset",
        )
        _verify_dataset_count(
            sick_calibration_train,
            SICK_TRAIN_EXPECTED_RECORDS,
            "SICK calibration training dataset",
        )

    sick_test = load_dataset(
        base_dir / SICK_TEST_FILENAME,
        "SICK test dataset",
    )
    _verify_dataset_count(
        sick_test,
        SICK_TEST_EXPECTED_RECORDS,
        "SICK test dataset",
    )

    console.print(
        f"SICK calibration training records: {len(sick_calibration_train):,}"
    )
    console.print(
        f"SICK test records: {len(sick_test):,}"
    )
    console.print(Panel("SICK evaluation", title="Phase 4"))

    # Preserve the existing raw evaluation path unchanged.
    metrics = chkterr2(sick_test, model)
    _report_paper_calibration(
        sick_calibration_train,
        sick_test,
        model,
        metrics,
    )

    # This example is diagnostic only and is not part of training.
    sentence_a = "A truly wise man"
    sentence_b = "He is smart"
    raw_similarity = float(model.predict_similarity(sentence_a, sentence_b))
    if not 0.0 <= raw_similarity <= 1.0:
        raise RuntimeError(
            "Raw example similarity must remain in [0, 1] before inverse "
            "scaling."
        )
    example_score = raw_similarity * 4.0 + 1.0

    console.print("Similarity of")
    console.print(f"{sentence_a} / {sentence_b}")
    console.print(f"on a scale of 1-5: {example_score:.6f}")

    return model, metrics


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the compatibility command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the faithful Siamese-LSTM STS → SICK reproduction flow."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the original repository resources.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to bestsem.p for evaluation-only mode; defaults to "
            "--base-dir/bestsem.p."
        ),
    )
    parser.add_argument(
        "--no-training",
        action="store_true",
        help=(
            "Load bestsem.p and evaluate without STS/SICK training. "
            "This is the only path that loads the legacy checkpoint."
        ),
    )
    parser.add_argument(
        "--no-synonym-augmentation",
        action="store_true",
        help=(
            "Use the original 330-epoch SICK path without synonym "
            "augmentation."
        ),
    )
    return parser


def main() -> int:
    """Execute the reproduction entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    with tqdm(
        total=1,
        desc="Initializing reproduction",
        leave=False,
    ) as progress:
        run_experiment(
            args.base_dir,
            model_path=args.checkpoint,
            training_enabled=not args.no_training,
            synonym_augmentation=not args.no_synonym_augmentation,
        )
        progress.update(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
