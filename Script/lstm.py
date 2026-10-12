"""PyTorch reproduction of the LSTM encoder used by the MaLSTM repository.

This module is a Python 3.10 compatibility replacement for the original
Theano implementation in ``lstm.py`` from Mueller and Thyagarajan's
Siamese-LSTM repository.  The recurrent computation is intentionally written
explicitly rather than delegated to ``torch.nn.LSTM`` so that the original
gate equations, parameter layout, masking semantics, initialization, and
final hidden-state representation remain visible and reproducible.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from random import sample
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import scipy.stats as meas
import torch
from torch import Tensor, nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproduction constants
# ---------------------------------------------------------------------------

INPUT_DIM = 300
HIDDEN_DIM = 50
GATE_COUNT = 4
FORGET_GATE_INDEX = 1
FORGET_GATE_BIAS = 1.5
INITIALIZATION_MU = 0.0
INITIALIZATION_SIGMA = 0.2
ADADELTA_RHO = 0.95
ADADELTA_EPSILON = 1e-6
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 32


# The original repository uses the gate ordering i, f, o, c.
INPUT_GATE = 0
FORGET_GATE = 1
OUTPUT_GATE = 2
CELL_GATE = 3


ArrayLike = np.ndarray | Sequence[float]


def numpy_floatX(data: Any) -> np.ndarray:
    """Convert input data to the original single-precision NumPy dtype."""
    return np.asarray(data, dtype=np.float32)


def _parameter_key(prefix: str, name: str) -> str:
    """Build a parameter key using the original repository naming scheme."""
    return f"{prefix}_{name}"


def _random_matrix(rows: int, columns: int, sigma: float) -> np.ndarray:
    """Create a Gaussian matrix using the original initialization scale."""
    return np.random.normal(
        INITIALIZATION_MU,
        sigma,
        (rows, columns),
    ).astype(np.float32)


def getlayerx(
    parameters: MutableMapping[str, np.ndarray],
    prefix: str,
    hidden_dim: int,
    input_dim: int,
) -> MutableMapping[str, np.ndarray]:
    """Initialize one LSTM layer exactly as the original repository does.

    The original implementation creates four independent Gaussian blocks for
    the recurrent matrix and input matrix, concatenated in the order
    ``input, forget, output, cell``.  Biases are sampled uniformly from
    ``[-0.5, 0.5]`` and the forget-gate slice is then set to ``1.5``.
    """
    recurrent = np.concatenate(
        [
            _random_matrix(hidden_dim, hidden_dim, INITIALIZATION_SIGMA)
            for _ in range(GATE_COUNT)
        ],
        axis=0,
    ) / np.sqrt(hidden_dim)

    input_weights = np.concatenate(
        [
            _random_matrix(hidden_dim, input_dim, INITIALIZATION_SIGMA)
            for _ in range(GATE_COUNT)
        ],
        axis=0,
    ) / np.sqrt(np.sqrt(hidden_dim * input_dim))

    bias = np.random.uniform(
        -0.5,
        0.5,
        size=(GATE_COUNT * hidden_dim,),
    ).astype(np.float32)
    bias[FORGET_GATE * hidden_dim : (FORGET_GATE + 1) * hidden_dim] = (
        FORGET_GATE_BIAS
    )

    parameters[_parameter_key(prefix, "U")] = recurrent.astype(np.float32)
    parameters[_parameter_key(prefix, "W")] = input_weights.astype(np.float32)
    parameters[_parameter_key(prefix, "b")] = bias
    return parameters


def creatrnnx() -> OrderedDict[str, np.ndarray]:
    """Create the original MaLSTM parameter dictionary.

    The original repository constructs two Siamese branches with identical
    initial values.  The PyTorch model below represents those branches with
    one shared parameter set; this preserves the intended tied-weight
    architecture without maintaining duplicate trainable tensors.
    """
    parameters: OrderedDict[str, np.ndarray] = OrderedDict()
    getlayerx(parameters, "1lstm1", HIDDEN_DIM, INPUT_DIM)

    # Preserve the original serialized API: both branch keys are present and
    # contain identical arrays, while the PyTorch module uses one shared set.
    parameters[_parameter_key("2lstm1", "U")] = parameters[
        _parameter_key("1lstm1", "U")
    ].copy()
    parameters[_parameter_key("2lstm1", "W")] = parameters[
        _parameter_key("1lstm1", "W")
    ].copy()
    parameters[_parameter_key("2lstm1", "b")] = parameters[
        _parameter_key("1lstm1", "b")
    ].copy()
    return parameters


def zipp(
    parameters: Mapping[str, np.ndarray],
    tparameters: MutableMapping[str, Tensor],
) -> None:
    """Copy NumPy parameters into a PyTorch parameter mapping."""
    for key, value in parameters.items():
        if key not in tparameters:
            raise KeyError(f"Unknown parameter key: {key}")
        tparameters[key].copy_(torch.as_tensor(value, dtype=torch.float32))


def unzip(zipped: Mapping[str, Tensor]) -> OrderedDict[str, np.ndarray]:
    """Return detached NumPy arrays from a PyTorch parameter mapping."""
    result: OrderedDict[str, np.ndarray] = OrderedDict()
    for key, value in zipped.items():
        result[key] = value.detach().cpu().numpy().copy()
    return result


def init_tparams(
    parameters: Mapping[str, np.ndarray],
) -> OrderedDict[str, Tensor]:
    """Create detached PyTorch tensors matching the old Theano parameters."""
    result: OrderedDict[str, Tensor] = OrderedDict()
    for key, value in parameters.items():
        result[key] = torch.as_tensor(value, dtype=torch.float32).clone()
    return result


class _ExplicitLSTMEncoder(nn.Module):
    """Single-layer explicit LSTM encoder with the repository's equations."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.U = nn.Parameter(torch.empty(GATE_COUNT * hidden_dim, hidden_dim))
        self.W = nn.Parameter(torch.empty(GATE_COUNT * hidden_dim, input_dim))
        self.b = nn.Parameter(torch.empty(GATE_COUNT * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using the original repository scheme."""
        params: OrderedDict[str, np.ndarray] = OrderedDict()
        getlayerx(params, "lstm", self.hidden_dim, self.input_dim)
        with torch.no_grad():
            self.U.copy_(torch.from_numpy(params["lstm_U"]))
            self.W.copy_(torch.from_numpy(params["lstm_W"]))
            self.b.copy_(torch.from_numpy(params["lstm_b"]))

    def load_original_parameters(
        self,
        parameters: Mapping[str, np.ndarray],
        prefix: str = "1lstm1",
    ) -> None:
        """Load an original repository parameter dictionary into the encoder."""
        required = (
            _parameter_key(prefix, "U"),
            _parameter_key(prefix, "W"),
            _parameter_key(prefix, "b"),
        )
        missing = [key for key in required if key not in parameters]
        if missing:
            raise KeyError(f"Missing original LSTM parameters: {missing}")

        recurrent = np.asarray(parameters[required[0]], dtype=np.float32)
        input_weights = np.asarray(parameters[required[1]], dtype=np.float32)
        bias = np.asarray(parameters[required[2]], dtype=np.float32)

        expected_u = (GATE_COUNT * self.hidden_dim, self.hidden_dim)
        expected_w = (GATE_COUNT * self.hidden_dim, self.input_dim)
        expected_b = (GATE_COUNT * self.hidden_dim,)
        if recurrent.shape != expected_u:
            raise ValueError(f"Expected U shape {expected_u}, got {recurrent.shape}")
        if input_weights.shape != expected_w:
            raise ValueError(
                f"Expected W shape {expected_w}, got {input_weights.shape}"
            )
        if bias.shape != expected_b:
            raise ValueError(f"Expected b shape {expected_b}, got {bias.shape}")

        with torch.no_grad():
            self.U.copy_(torch.from_numpy(recurrent))
            self.W.copy_(torch.from_numpy(input_weights))
            self.b.copy_(torch.from_numpy(bias))

    def export_original_parameters(self) -> OrderedDict[str, np.ndarray]:
        """Export this encoder in the original parameter layout."""
        return OrderedDict(
            [
                (_parameter_key("1lstm1", "U"), self.U.detach().cpu().numpy().copy()),
                (_parameter_key("1lstm1", "W"), self.W.detach().cpu().numpy().copy()),
                (_parameter_key("1lstm1", "b"), self.b.detach().cpu().numpy().copy()),
                (_parameter_key("2lstm1", "U"), self.U.detach().cpu().numpy().copy()),
                (_parameter_key("2lstm1", "W"), self.W.detach().cpu().numpy().copy()),
                (_parameter_key("2lstm1", "b"), self.b.detach().cpu().numpy().copy()),
            ]
        )

    def forward(self, state_below: Tensor, mask: Tensor) -> Tensor:
        """Encode a sequence and return all hidden states.

        Args:
            state_below: Tensor shaped ``[time, batch, input_dim]``.
            mask: Tensor shaped ``[time, batch]`` with 1 for valid tokens and
                0 for padding. The original implementation keeps the previous
                hidden/cell state unchanged at masked positions.

        Returns:
            Tensor shaped ``[time, batch, hidden_dim]`` containing the hidden
            state after every sequence step.
        """
        if state_below.ndim != 3:
            raise ValueError(
                "state_below must have shape [time, batch, input_dim]"
            )
        if mask.ndim != 2:
            raise ValueError("mask must have shape [time, batch]")
        if state_below.shape[:2] != mask.shape:
            raise ValueError(
                "state_below and mask must have matching time and batch dimensions"
            )
        if state_below.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {state_below.shape[2]}"
            )

        state_below = state_below.to(dtype=self.W.dtype, device=self.W.device)
        mask = mask.to(dtype=self.W.dtype, device=self.W.device)
        batch_size = state_below.shape[1]
        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            dtype=state_below.dtype,
            device=state_below.device,
        )
        cell = torch.zeros_like(hidden)
        hidden_states: list[Tensor] = []

        # Original equation: W*x_t + U*h_(t-1) + b.
        # Gate slices are [i, f, o, c] exactly as in the repository.
        for timestep in range(state_below.shape[0]):
            x_t = state_below[timestep]
            mask_t = mask[timestep].unsqueeze(1)
            preactivation = (
                torch.matmul(x_t, self.W.transpose(0, 1))
                + torch.matmul(hidden, self.U.transpose(0, 1))
                + self.b
            )

            input_gate = torch.sigmoid(
                preactivation[:, INPUT_GATE * self.hidden_dim : (INPUT_GATE + 1) * self.hidden_dim]
            )
            forget_gate = torch.sigmoid(
                preactivation[:, FORGET_GATE * self.hidden_dim : (FORGET_GATE + 1) * self.hidden_dim]
            )
            output_gate = torch.sigmoid(
                preactivation[:, OUTPUT_GATE * self.hidden_dim : (OUTPUT_GATE + 1) * self.hidden_dim]
            )
            cell_candidate = torch.tanh(
                preactivation[:, CELL_GATE * self.hidden_dim : (CELL_GATE + 1) * self.hidden_dim]
            )

            new_cell = forget_gate * cell + input_gate * cell_candidate
            cell = mask_t * new_cell + (1.0 - mask_t) * cell

            new_hidden = output_gate * torch.tanh(cell)
            hidden = mask_t * new_hidden + (1.0 - mask_t) * hidden
            hidden_states.append(hidden)

        return torch.stack(hidden_states, dim=0)


class _MaLSTM(nn.Module):
    """Shared-weight MaLSTM encoder and similarity computation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ExplicitLSTMEncoder(INPUT_DIM, HIDDEN_DIM)

    def encode(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        """Return the final hidden state for each sequence in a batch."""
        hidden_states = self.encoder(embeddings, mask)
        return hidden_states[-1]

    def similarity(
        self,
        embeddings_a: Tensor,
        mask_a: Tensor,
        embeddings_b: Tensor,
        mask_b: Tensor,
    ) -> Tensor:
        """Compute the original exp(-L1) Manhattan similarity."""
        representation_a = self.encode(embeddings_a, mask_a)
        representation_b = self.encode(embeddings_b, mask_b)
        distance = torch.sum(torch.abs(representation_b - representation_a), dim=1)
        similarity = torch.exp(-distance)
        return torch.clamp(similarity, 1e-7, 1.0 - 1e-7)


class lstm:
    """Compatibility wrapper exposing the original ``lstm`` class API.

    The public methods intentionally retain the names used by ``main.py`` and
    ``sentiment.py`` in the original repository: ``train_lstm``, ``chkterr2``,
    ``predict_similarity``, ``f2sim``, ``f_proj11``, and ``f_cost``.
    """

    def __init__(
        self,
        nam: str | bool = "bestsem.p",
        load: bool = False,
        training: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _MaLSTM().to(self.device)
        self.training = training
        self._adadelta_state: dict[str, dict[str, Tensor]] = {}

        if load:
            self.load_original_weights(nam)

        self._sync_mode(training=False)

    def _sync_mode(self, training: bool) -> None:
        """Set the PyTorch module's mode without changing model mathematics."""
        self.model.train(training)

    @staticmethod
    def _load_pickle(path: str | Path) -> Any:
        """Load an original Python 2/3 pickle with compatibility handling."""
        import pickle

        with open(path, "rb") as handle:
            try:
                return pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                return pickle.load(handle, encoding="latin1")

    def load_original_weights(self, path: str | Path) -> None:
        """Load ``bestsem.p``-style original repository parameters."""
        parameters = self._load_pickle(path)
        if not isinstance(parameters, Mapping):
            raise TypeError("Original LSTM weights must be a mapping")
        self.model.encoder.load_original_parameters(parameters, prefix="1lstm1")

        # The original repository stores both Siamese branches.  They are
        # expected to be tied and the supplied repository weights are equal.
        for name in ("U", "W", "b"):
            first = np.asarray(parameters[_parameter_key("1lstm1", name)])
            second = np.asarray(parameters[_parameter_key("2lstm1", name)])
            if not np.array_equal(first, second):
                raise ValueError(
                    f"Siamese branch parameters differ for {name}; "
                    "cannot preserve tied-weight behavior."
                )

    def save_original_weights(self, path: str | Path) -> None:
        """Save weights in the original repository's pickle layout."""
        import pickle

        with open(path, "wb") as handle:
            pickle.dump(self.model.encoder.export_original_parameters(), handle)

    @staticmethod
    def _to_tensor(array: np.ndarray | Tensor, device: torch.device) -> Tensor:
        """Convert a NumPy array or tensor to a float32 tensor on the device."""
        if isinstance(array, Tensor):
            return array.to(device=device, dtype=torch.float32)
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    def f2sim(
        self,
        emb11: np.ndarray | Tensor,
        mask11: np.ndarray | Tensor,
        emb21: np.ndarray | Tensor,
        mask21: np.ndarray | Tensor,
    ) -> np.ndarray:
        """Return batch MaLSTM similarity predictions as a NumPy array."""
        self._sync_mode(training=False)
        with torch.no_grad():
            prediction = self.model.similarity(
                self._to_tensor(emb11, self.device),
                self._to_tensor(mask11, self.device),
                self._to_tensor(emb21, self.device),
                self._to_tensor(mask21, self.device),
            )
        return prediction.detach().cpu().numpy()

    def f_proj11(
        self,
        emb11: np.ndarray | Tensor,
        mask11: np.ndarray | Tensor,
    ) -> np.ndarray:
        """Return all hidden states for the first Siamese branch."""
        self._sync_mode(training=False)
        with torch.no_grad():
            hidden_states = self.model.encoder(
                self._to_tensor(emb11, self.device),
                self._to_tensor(mask11, self.device),
            )
        return hidden_states.detach().cpu().numpy()

    def f_cost(
        self,
        emb11: np.ndarray | Tensor,
        mask11: np.ndarray | Tensor,
        emb21: np.ndarray | Tensor,
        mask21: np.ndarray | Tensor,
        y: np.ndarray | Tensor,
    ) -> float:
        """Return the original MSE cost on labels scaled to [0, 1]."""
        self._sync_mode(training=False)
        with torch.no_grad():
            prediction = self.model.similarity(
                self._to_tensor(emb11, self.device),
                self._to_tensor(mask11, self.device),
                self._to_tensor(emb21, self.device),
                self._to_tensor(mask21, self.device),
            )
            labels = self._to_tensor(y, self.device)
            labels = torch.clamp((labels - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)
            cost = torch.mean((prediction - labels) ** 2)
        return float(cost.detach().cpu())

    def _prepare_sentence_batch(
        self,
        data: Sequence[Sequence[Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Delegate sentence preparation to the repository's sentence module."""
        try:
            from sentences import prepare_data
        except ImportError as exc:
            raise ImportError(
                "The original sentences.py module is required for text-level "
                "methods. It should remain the repository's preprocessing API."
            ) from exc
        return prepare_data(data)

    def _embed_matrix(self, token_matrix: np.ndarray) -> np.ndarray:
        """Convert token strings to 300-D Word2Vec vectors via repository API."""
        try:
            from sentences import embed
        except ImportError as exc:
            raise ImportError(
                "The original sentences.py module is required for Word2Vec "
                "embedding lookup."
            ) from exc
        return embed(token_matrix)

    @staticmethod
    def _batch_embeddings(
        token_matrix: np.ndarray,
        embed_fn: Any,
    ) -> np.ndarray:
        """Build the original ``[time, batch, 300]`` embedding tensor.

        ``sentences.getmtr`` returns tokens as ``[batch, time]`` while its
        mask is already ``[time, batch]``.  Each row therefore represents one
        sentence and must be embedded independently before transposing the
        resulting ``[batch, time, 300]`` array to the LSTM's required
        ``[time, batch, 300]`` layout.
        """
        token_array = np.asarray(token_matrix)
        if token_array.ndim != 2:
            raise ValueError(
                "token_matrix must have shape [batch, time]"
            )

        vectors = [
            np.asarray(embed_fn(token_array[index]), dtype=np.float32)
            for index in range(token_array.shape[0])
        ]
        if not vectors:
            raise ValueError("Cannot embed an empty sentence batch.")

        batch_embeddings = np.stack(vectors, axis=0)
        if batch_embeddings.ndim != 3 or batch_embeddings.shape[2] != INPUT_DIM:
            raise ValueError(
                "Embedding function must return [time, 300] vectors; "
                f"got {batch_embeddings.shape}."
            )

        return np.transpose(batch_embeddings, (1, 0, 2)).astype(
            np.float32, copy=False
        )

    def train_lstm(
        self,
        train: Sequence[Sequence[Any]],
        max_epochs: int,
    ) -> None:
        """Train the tied MaLSTM using the repository's AdaDelta formulation.

        The implementation retains the original batch size, learning rate,
        sequence preprocessing, label scaling, and two-branch shared-weight
        computation.  AdaDelta state follows the explicit equations in the
        original Theano implementation rather than ``torch.optim.Adadelta``.
        """
        if max_epochs < 0:
            raise ValueError("max_epochs must be non-negative")

        self._sync_mode(training=True)
        self.training = True
        self._initialize_adadelta_state()

        for epoch in range(max_epochs):
            epoch_start = torch.cuda.Event(enable_timing=False) if self.device.type == "cuda" else None
            if epoch_start is not None:
                epoch_start.record()

            indices = sample(range(len(train)), len(train))
            updates = 0
            progress = tqdm(
                range(0, len(train), DEFAULT_BATCH_SIZE),
                desc=f"Epoch {epoch}",
                leave=False,
            )
            for start in progress:
                batch_indices = indices[start : start + DEFAULT_BATCH_SIZE]
                batch = [train[index] for index in batch_indices]
                x1, mask1, x2, mask2, labels = self._prepare_sentence_batch(batch)
                emb1 = self._batch_embeddings(x1, self._embed_matrix)
                emb2 = self._batch_embeddings(x2, self._embed_matrix)

                cost = self._train_batch(emb2, mask2, emb1, mask1, labels)
                updates += 1
                progress.set_postfix(cost=f"{cost:.6f}")

            if epoch_start is not None:
                torch.cuda.synchronize(self.device)

            if updates == 0:
                progress.close()

    def _initialize_adadelta_state(self) -> None:
        """Initialize the original AdaDelta accumulators once per model."""
        if self._adadelta_state:
            return
        for name, parameter in self.model.named_parameters():
            self._adadelta_state[name] = {
                "gradient": torch.zeros_like(parameter),
                "running_gradient_sq": torch.zeros_like(parameter),
                "running_update_sq": torch.zeros_like(parameter),
            }

    def _train_batch(
        self,
        emb_a: np.ndarray,
        mask_a: np.ndarray,
        emb_b: np.ndarray,
        mask_b: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Perform one exact two-branch MSE/AdaDelta update."""
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

        # The original repository averages the tied Siamese branch gradients
        # before applying its update. With genuinely shared PyTorch parameters,
        # autograd has already accumulated both branch contributions. The
        # original class then divides by four, which is retained here.
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            state = self._adadelta_state[name]
            gradient = parameter.grad.detach() / 4.0
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

    def chkterr2(self, mydata: Sequence[Sequence[Any]]) -> tuple[float, float, float]:
        """Evaluate MSE, Pearson correlation, and Spearman correlation."""
        predictions: list[float] = []
        targets: list[float] = []
        self._sync_mode(training=False)

        for start in range(0, len(mydata), 256):
            batch = mydata[start : start + 256]
            x1, mask1, x2, mask2, labels = self._prepare_sentence_batch(batch)
            emb1 = self._batch_embeddings(x1, self._embed_matrix)
            emb2 = self._batch_embeddings(x2, self._embed_matrix)
            batch_predictions = self.f2sim(emb1, mask1, emb2, mask2) * 4.0 + 1.0
            predictions.extend(np.asarray(batch_predictions).reshape(-1).tolist())
            targets.extend(np.asarray(labels).reshape(-1).tolist())

        px = np.asarray(predictions, dtype=np.float32)
        yx = np.asarray(targets, dtype=np.float32)
        if px.size == 0:
            raise ValueError("Cannot evaluate an empty dataset")

        pearson = float(meas.pearsonr(px, yx)[0])
        spearman = float(meas.spearmanr(yx, px)[0])
        mse = float(np.mean(np.square(px - yx)))
        return mse, pearson, spearman

    def predict_similarity(self, sa: str, sb: str) -> np.ndarray:
        """Predict MaLSTM similarity for a pair of sentences."""
        data = [[sa, sb, 0.0]]
        x1, mask1, x2, mask2, _ = self._prepare_sentence_batch(data)
        emb1 = self._batch_embeddings(x1, self._embed_matrix)
        emb2 = self._batch_embeddings(x2, self._embed_matrix)
        return self.f2sim(emb1, mask1, emb2, mask2)

    def parameters_numpy(self) -> OrderedDict[str, np.ndarray]:
        """Return the shared model parameters in repository-compatible form."""
        return self.model.encoder.export_original_parameters()


def lstm_layer2(
    tparams: Mapping[str, Tensor | np.ndarray],
    state_below: Tensor | np.ndarray,
    options: Optional[Mapping[str, Any]] = None,
    prefix: str = "lstm",
    mask: Optional[Tensor | np.ndarray] = None,
    nhd: Optional[int] = None,
) -> Tensor:
    """Compatibility implementation of the original ``lstm_layer2`` API.

    The computation is performed directly from ``tparams`` so gradients
    continue to flow to caller-owned PyTorch tensors.
    """
    del options
    if mask is None:
        raise ValueError("mask is required")

    x = (
        state_below
        if isinstance(state_below, Tensor)
        else torch.as_tensor(state_below, dtype=torch.float32)
    )
    m = mask if isinstance(mask, Tensor) else torch.as_tensor(mask, dtype=torch.float32)
    if x.ndim != 3 or m.ndim != 2:
        raise ValueError("state_below must be [time,batch,input] and mask [time,batch]")

    u_value = tparams[_parameter_key(prefix, "U")]
    w_value = tparams[_parameter_key(prefix, "W")]
    b_value = tparams[_parameter_key(prefix, "b")]
    u = u_value if isinstance(u_value, Tensor) else torch.as_tensor(u_value, dtype=torch.float32)
    w = w_value if isinstance(w_value, Tensor) else torch.as_tensor(w_value, dtype=torch.float32)
    b = b_value if isinstance(b_value, Tensor) else torch.as_tensor(b_value, dtype=torch.float32)
    hidden_dim = int(nhd or u.shape[1])
    if u.shape != (4 * hidden_dim, hidden_dim):
        raise ValueError(f"Invalid U shape: {tuple(u.shape)}")
    if w.shape != (4 * hidden_dim, x.shape[2]):
        raise ValueError(f"Invalid W shape: {tuple(w.shape)}")
    if b.shape != (4 * hidden_dim,):
        raise ValueError(f"Invalid b shape: {tuple(b.shape)}")

    x = x.to(dtype=w.dtype, device=w.device)
    m = m.to(dtype=w.dtype, device=w.device)
    batch_size = x.shape[1]
    hidden = torch.zeros(batch_size, hidden_dim, dtype=x.dtype, device=x.device)
    cell = torch.zeros_like(hidden)
    hidden_states: list[Tensor] = []

    for timestep in range(x.shape[0]):
        mask_t = m[timestep].unsqueeze(1)
        preactivation = (
            torch.matmul(x[timestep], w.transpose(0, 1))
            + torch.matmul(hidden, u.transpose(0, 1))
            + b
        )
        input_gate = torch.sigmoid(preactivation[:, :hidden_dim])
        forget_gate = torch.sigmoid(preactivation[:, hidden_dim : 2 * hidden_dim])
        output_gate = torch.sigmoid(preactivation[:, 2 * hidden_dim : 3 * hidden_dim])
        cell_candidate = torch.tanh(preactivation[:, 3 * hidden_dim :])
        new_cell = forget_gate * cell + input_gate * cell_candidate
        cell = mask_t * new_cell + (1.0 - mask_t) * cell
        new_hidden = output_gate * torch.tanh(cell)
        hidden = mask_t * new_hidden + (1.0 - mask_t) * hidden
        hidden_states.append(hidden)

    return torch.stack(hidden_states, dim=0)

def getpl2(
    prevlayer: Tensor | np.ndarray,
    pre: str,
    mymask: Tensor | np.ndarray,
    used: bool,
    rrng: Any,
    size: int,
    tnewp: Mapping[str, Tensor | np.ndarray],
) -> Tensor:
    """Compatibility wrapper matching the original ``getpl2`` signature."""
    del used, rrng
    return lstm_layer2(tnewp, prevlayer, {}, prefix=pre, mask=mymask, nhd=size)


def get_layer(name: str) -> Any:
    """Return a named layer helper using the original layer API convention."""
    layers = {"lstm": lstm_layer2, "lstm_layer2": lstm_layer2}
    try:
        return layers[name]
    except KeyError as exc:
        raise KeyError(f"Unknown layer: {name}") from exc


__all__ = [
    "INPUT_DIM",
    "HIDDEN_DIM",
    "FORGET_GATE_BIAS",
    "numpy_floatX",
    "getlayerx",
    "creatrnnx",
    "zipp",
    "unzip",
    "init_tparams",
    "lstm",
]
