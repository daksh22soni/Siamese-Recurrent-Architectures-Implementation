"""Python 3.10/PyTorch reproduction of the original ``mainmodel.py``.

The original module builds the Manhattan LSTM training graph with Theano and
then performs STS pre-training followed by SICK training.  This compatibility
module keeps that public training/evaluation surface while delegating the
actual recurrent computation to the explicit PyTorch implementation in
``lstm.py``.

No new encoder or optimizer is introduced here.  The Siamese branches use the
single tied encoder supplied by :class:`lstm.lstm`, the similarity is
``exp(-L1)``, the loss is MSE on labels rescaled from ``[1, 5]`` to ``[0, 1]``,
and the optimizer is the repository's explicit AdaDelta formulation.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from random import sample
from typing import Any, Sequence

import numpy as np
import scipy.stats as meas
import torch

from lstm import (
    ADADELTA_EPSILON,
    ADADELTA_RHO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    INPUT_DIM,
    HIDDEN_DIM,
    creatrnnx,
    getlayerx,
    init_tparams,
    lstm,
    numpy_floatX,
)
from sentences import embed, expand, prepare_data


# ---------------------------------------------------------------------------
# Original module-level configuration
# ---------------------------------------------------------------------------

prefix = "lstm"
training = True
Syn_aug = True
rate = 0.5

# The paper states that gradient clipping is used, but neither the paper nor
# the supplied executable repository exposes a reproducible numeric threshold.
# The original repository therefore remains the executable source for this
# compatibility implementation: clipping is disabled unless an exact source
# threshold is established in a later forensic step.  No value is invented.
GRADIENT_CLIP_THRESHOLD: float | None = None


class MainModel(lstm):
    """Compatibility model for the original Theano ``mainmodel.py``.

    The parent implementation owns the single shared LSTM encoder and the
    exact repository AdaDelta state/update equations.  This class preserves
    the repository training behavior because the paper's clipping threshold
    cannot be recovered exactly from the supplied sources.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        load: bool = False,
        training: bool = True,
        device: str | torch.device | None = None,
        gradient_clip_threshold: float | None = GRADIENT_CLIP_THRESHOLD,
    ) -> None:
        """Initialize the reproduction model.

        Parameters
        ----------
        model_path:
            Optional legacy pickle containing the six original LSTM arrays.
        load:
            Load ``model_path`` when true.
        training:
            Whether the model is initialized for training.
        device:
            PyTorch device.  CPU is used when omitted and CUDA is unavailable.
        gradient_clip_threshold:
            Optional global gradient-norm clipping threshold.  The default is
            ``None`` because neither the paper nor the supplied repository
            specifies a reproducible numeric threshold.  A threshold must not
            be invented for the research reproduction.
        """
        if gradient_clip_threshold is not None and gradient_clip_threshold <= 0:
            raise ValueError("gradient_clip_threshold must be positive or None")

        super().__init__(
            nam=model_path if model_path is not None else "bestsem.p",
            load=load,
            training=training,
        )
        if device is not None:
            requested_device = torch.device(device)
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            self.device = requested_device
            self.model = self.model.to(self.device)
        self.gradient_clip_threshold = gradient_clip_threshold

    def train_lstm(
        self,
        train: Sequence[Sequence[Any]],
        max_epochs: int,
    ) -> None:
        """Train using the original ``mainmodel.py`` batch ordering.

        The original mainmodel creates embeddings per sentence, stacks them
        into ``[time, batch, 300]``, averages the two independently computed
        Siamese branch gradients, and then applies custom AdaDelta.  With one
        shared PyTorch encoder, autograd sums the branch contributions, so the
        equivalent normalization is division by two.
        """
        if max_epochs < 0:
            raise ValueError("max_epochs must be non-negative")
        if not train:
            return

        self._sync_mode(training=True)
        self.training = True
        self._initialize_adadelta_state()

        for epoch in range(max_epochs):
            epoch_start = time.perf_counter()
            indices = sample(range(len(train)), len(train))
            for start in range(0, len(train), DEFAULT_BATCH_SIZE):
                batch_indices = indices[start : start + DEFAULT_BATCH_SIZE]
                batch = [train[index] for index in batch_indices]
                x1, mask1, x2, mask2, labels = prepare_data(batch)
                emb1 = _batch_embeddings(x1)
                emb2 = _batch_embeddings(x2)
                self._train_batch(emb2, mask2, emb1, mask1, labels)

            elapsed = time.perf_counter() - epoch_start
            if elapsed < 0.0:  # pragma: no cover - defensive clock guard
                raise RuntimeError("System clock produced a negative duration")

    def _train_batch(
        self,
        emb_a: np.ndarray,
        mask_a: np.ndarray,
        emb_b: np.ndarray,
        mask_b: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Run one MSE/backpropagation/AdaDelta update.

        When an explicitly sourced clipping threshold is configured, clipping
        is applied after backpropagation and before the AdaDelta accumulator
        update.  The default reproduction configuration leaves this disabled
        because no exact threshold is available from the sources.
        """
        self._initialize_adadelta_state()
        self.model.zero_grad(set_to_none=True)
        prediction = self.model.similarity(
            self._to_tensor(emb_a, self.device),
            self._to_tensor(mask_a, self.device),
            self._to_tensor(emb_b, self.device),
            self._to_tensor(mask_b, self.device),
        )
        target = self._to_tensor(labels, self.device)
        target = torch.clamp((target - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)
        cost = torch.mean((prediction - target) ** 2)
        cost.backward()

        if self.gradient_clip_threshold is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_threshold
            )

        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue

            state = self._adadelta_state[name]
            # The original mainmodel computes (g_a + g_b) / 2 before
            # duplicating that gradient onto the two tied Theano parameters.
            # PyTorch's shared parameter receives g_a + g_b directly, so /2
            # is the mathematically equivalent tied-weight gradient.
            gradient = parameter.grad.detach() / 2.0
            state["gradient"].copy_(gradient)
            state["running_gradient_sq"].mul_(ADADELTA_RHO).add_(
                (1.0 - ADADELTA_RHO) * gradient.square()
            )
            update_direction = -(
                torch.sqrt(state["running_update_sq"] + ADADELTA_EPSILON)
                / torch.sqrt(state["running_gradient_sq"] + ADADELTA_EPSILON)
                * state["gradient"]
            )
            state["running_update_sq"].mul_(ADADELTA_RHO).add_(
                (1.0 - ADADELTA_RHO) * update_direction.square()
            )
            with torch.no_grad():
                parameter.add_(update_direction)

        return float(cost.detach().cpu())


def chkterr2(
    mydata: Sequence[Sequence[Any]],
    model_instance: MainModel | lstm | None = None,
) -> tuple[float, float, float]:
    """Evaluate MSE, Pearson, and Spearman metrics on sentence pairs.

    Predictions are converted from the internal ``[0, 1]`` similarity scale
    back to the original ``[1, 5]`` relatedness scale before evaluation.

    Parameters
    ----------
    mydata:
        Sentence-pair records ``[sentence_a, sentence_b, score]``.
    model_instance:
        Existing reproduction model.  A default instance is created when
        omitted.
    """
    model = model_instance or _get_default_model(training=False)
    predictions: list[float] = []
    targets: list[float] = []

    for start in range(0, len(mydata), 256):
        batch = mydata[start : start + 256]
        if not batch:
            continue

        x1, mask1, x2, mask2, labels = prepare_data(batch)
        emb1 = _batch_embeddings(x1)
        emb2 = _batch_embeddings(x2)
        batch_predictions = model.f2sim(emb1, mask1, emb2, mask2) * 4.0 + 1.0
        predictions.extend(np.asarray(batch_predictions).reshape(-1).tolist())
        targets.extend(np.asarray(labels).reshape(-1).tolist())

    if not predictions:
        raise ValueError("Cannot evaluate an empty dataset")

    px = np.asarray(predictions, dtype=np.float32)
    yx = np.asarray(targets, dtype=np.float32)
    pearson = float(meas.pearsonr(px, yx)[0])
    spearman = float(meas.spearmanr(yx, px)[0])
    mse = float(np.mean(np.square(px - yx)))
    return mse, pearson, spearman


def train_lstm(
    train: Sequence[Sequence[Any]],
    max_epochs: int,
    model_instance: MainModel | lstm | None = None,
) -> MainModel | lstm:
    """Train the reproduction model using the original batch/data flow.

    Parameters
    ----------
    train:
        Sentence-pair training examples.
    max_epochs:
        Number of complete passes through ``train``.
    model_instance:
        Existing model to update; otherwise a new :class:`MainModel` is used.

    Returns
    -------
    MainModel | lstm
        The updated model instance.
    """
    if max_epochs < 0:
        raise ValueError("max_epochs must be non-negative")

    model = model_instance or _get_default_model(training=True)
    model.train_lstm(train, max_epochs)
    return model


def _batch_embeddings(token_matrix: np.ndarray) -> np.ndarray:
    """Create the original time-major ``[time, batch, 300]`` embedding array."""
    if token_matrix.ndim != 2:
        raise ValueError("token_matrix must be a two-dimensional array")

    vectors = [embed(token_matrix[index]) for index in range(token_matrix.shape[0])]
    if not vectors:
        raise ValueError("token_matrix cannot be empty")

    stacked = np.dstack(vectors)
    return np.swapaxes(stacked, 1, 2).astype(np.float32)



def f2sim(
    emb11: np.ndarray | torch.Tensor,
    mask11: np.ndarray | torch.Tensor,
    emb21: np.ndarray | torch.Tensor,
    mask21: np.ndarray | torch.Tensor,
    model_instance: MainModel | lstm | None = None,
) -> np.ndarray:
    """Return the module-level compatibility similarity function."""
    model = model_instance or _get_default_model(training=False)
    return model.f2sim(emb11, mask11, emb21, mask21)


def f_cost(
    emb11: np.ndarray | torch.Tensor,
    mask11: np.ndarray | torch.Tensor,
    emb21: np.ndarray | torch.Tensor,
    mask21: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    model_instance: MainModel | lstm | None = None,
) -> float:
    """Return the module-level compatibility MSE cost."""
    model = model_instance or _get_default_model(training=False)
    return model.f_cost(emb11, mask11, emb21, mask21, y)


_DEFAULT_MODEL: MainModel | None = None


def _get_default_model(training: bool) -> MainModel:
    """Return the lazily initialized module-level compatibility model."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = MainModel(training=training)
    else:
        _DEFAULT_MODEL.training = training
        _DEFAULT_MODEL._sync_mode(training=training)
    return _DEFAULT_MODEL


def get_model(
    model_path: str | Path | None = None,
    load: bool = False,
    training: bool = True,
    device: str | torch.device | None = None,
    gradient_clip_threshold: float | None = GRADIENT_CLIP_THRESHOLD,
) -> MainModel:
    """Create an explicit mainmodel compatibility instance."""
    return MainModel(
        model_path=model_path,
        load=load,
        training=training,
        device=device,
        gradient_clip_threshold=gradient_clip_threshold,
    )


def run_original_training_flow(
    sts_path: str | Path,
    sick_train_path: str | Path,
    sick_test_path: str | Path,
    model_instance: MainModel | None = None,
    synonym_augmentation: bool = Syn_aug,
) -> tuple[MainModel, tuple[float, float, float]]:
    """Run the original STS → SICK → evaluation sequence.

    The supplied pickle datasets are loaded without changing their structure.
    The original epoch counts are retained: 66 STS epochs, followed by 375
    augmented SICK epochs or 330 non-augmented SICK epochs.
    """
    model = model_instance or _get_default_model(training=True)

    with open(sts_path, "rb") as handle:
        sts_train = pickle.load(handle, encoding="latin1")
    model.train_lstm(sts_train, 66)

    with open(sick_train_path, "rb") as handle:
        sick_train = pickle.load(handle, encoding="latin1")

    if synonym_augmentation:
        sick_train = expand(sick_train)
        model.train_lstm(sick_train, 375)
    else:
        model.train_lstm(sick_train, 330)

    with open(sick_test_path, "rb") as handle:
        sick_test = pickle.load(handle, encoding="latin1")

    metrics = chkterr2(sick_test, model)
    return model, metrics


# Original names retained as explicit aliases where useful to callers.
__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "GRADIENT_CLIP_THRESHOLD",
    "HIDDEN_DIM",
    "INPUT_DIM",
    "MainModel",
    "Syn_aug",
    "chkterr2",
    "f2sim",
    "f_cost",
    "creatrnnx",
    "get_model",
    "getlayerx",
    "init_tparams",
    "numpy_floatX",
    "prefix",
    "prepare_data",
    "run_original_training_flow",
    "train_lstm",
    "training",
]
