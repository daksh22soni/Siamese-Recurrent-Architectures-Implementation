"""Post-training local-linear regression calibration for the MaLSTM model.

The paper specifies a univariate nonparametric regression from the raw
MaLSTM similarity to the original SICK [1, 5] relatedness score, with
bandwidth selected by leave-one-out cross-validation.  The paper and the
supplied repository do not recover the kernel or the bandwidth-search
procedure, so this module makes those missing details explicit as a
best-effort approximation rather than presenting them as original facts.

No model-training code is imported here.  Calibration is strictly a
post-training operation and accepts only arrays of raw similarities and
human scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

RAW_SIMILARITY_MIN = 0.0
RAW_SIMILARITY_MAX = 1.0
TARGET_SCORE_MIN = 1.0
TARGET_SCORE_MAX = 5.0

DEFAULT_MIN_BANDWIDTH = 0.005
DEFAULT_MAX_BANDWIDTH = 0.500
DEFAULT_COARSE_STEP = 0.005
DEFAULT_REFINE_STEP = 0.001
DEFAULT_REFINE_RADIUS = 0.010
DEFAULT_TIE_TOLERANCE = 1.0e-12
DEFAULT_CHUNK_SIZE = 256


class CalibrationError(ValueError):
    """Raised when calibration input or numerical computation is invalid."""


@dataclass(frozen=True)
class CalibrationConfiguration:
    """Configuration for the approved best-effort calibration details."""

    min_bandwidth: float = DEFAULT_MIN_BANDWIDTH
    max_bandwidth: float = DEFAULT_MAX_BANDWIDTH
    coarse_step: float = DEFAULT_COARSE_STEP
    refine_step: float = DEFAULT_REFINE_STEP
    refine_radius: float = DEFAULT_REFINE_RADIUS
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        """Validate calibration configuration values."""
        values = (
            self.min_bandwidth,
            self.max_bandwidth,
            self.coarse_step,
            self.refine_step,
            self.refine_radius,
            self.tie_tolerance,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Calibration configuration must be finite")
        if self.min_bandwidth <= 0.0:
            raise ValueError("min_bandwidth must be positive")
        if self.max_bandwidth < self.min_bandwidth:
            raise ValueError("max_bandwidth must be >= min_bandwidth")
        if self.coarse_step <= 0.0 or self.refine_step <= 0.0:
            raise ValueError("Bandwidth steps must be positive")
        if self.refine_radius < 0.0:
            raise ValueError("refine_radius must be non-negative")
        if self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be non-negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class BandwidthSearchResult:
    """Selected bandwidth and its training-only LOOCV score."""

    bandwidth: float
    cv_mse: float
    coarse_bandwidth: float
    coarse_cv_mse: float


@dataclass(frozen=True)
class CalibrationMetrics:
    """MSE, Pearson, and Spearman metrics on the [1, 5] scale."""

    mse: float
    pearson: float
    spearman: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return metrics as ``(MSE, Pearson, Spearman)``."""
        return self.mse, self.pearson, self.spearman


class LocalLinearCalibrator:
    """Fit a univariate local-linear regression calibrator.

    A Gaussian kernel is used because the paper does not specify its kernel.
    The raw MaLSTM similarity is the only covariate; the target remains on
    the original SICK [1, 5] scale.  Predictions are not clipped because
    clipping is not specified by the paper.
    """

    def __init__(
        self,
        configuration: CalibrationConfiguration | None = None,
    ) -> None:
        """Initialize an unfitted calibrator."""
        self.configuration = configuration or CalibrationConfiguration()
        self._bandwidth: float | None = None
        self._raw_similarity: np.ndarray | None = None
        self._target_scores: np.ndarray | None = None

    @staticmethod
    def gaussian_kernel(u: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the standard Gaussian kernel."""
        values = np.asarray(u, dtype=np.float64)
        result = np.exp(-0.5 * np.square(values)) / np.sqrt(2.0 * np.pi)
        if np.isscalar(u):
            return float(result)
        return result

    @property
    def bandwidth(self) -> float:
        """Return the selected bandwidth."""
        if self._bandwidth is None:
            raise CalibrationError("Calibrator has not been fitted")
        return self._bandwidth

    @property
    def is_fitted(self) -> bool:
        """Return whether a bandwidth and training data have been fitted."""
        return self._bandwidth is not None

    def fit(
        self,
        raw_similarity: Sequence[float] | np.ndarray,
        target_scores: Sequence[float] | np.ndarray,
    ) -> "LocalLinearCalibrator":
        """Fit on training similarities and original SICK target scores."""
        x, y = _validate_training_arrays(raw_similarity, target_scores)
        search = self.select_bandwidth(x, y)
        self._raw_similarity = x.copy()
        self._target_scores = y.copy()
        self._bandwidth = search.bandwidth
        return self

    def predict(
        self,
        raw_similarity: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Predict calibrated [1, 5]-scale relatedness."""
        if self._bandwidth is None:
            raise CalibrationError("Call fit() before predict()")
        if self._raw_similarity is None or self._target_scores is None:
            raise CalibrationError("Fitted training data is unavailable")
        x_query = _validate_raw_similarity(raw_similarity)
        predictions = _local_linear_predict(
            self._raw_similarity,
            self._target_scores,
            x_query,
            self._bandwidth,
            self.configuration.chunk_size,
        )
        _require_finite("calibrated predictions", predictions)
        return predictions

    def fit_predict(
        self,
        raw_similarity: Sequence[float] | np.ndarray,
        target_scores: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Fit the calibrator and predict the supplied similarities."""
        self.fit(raw_similarity, target_scores)
        return self.predict(raw_similarity)

    def cross_validate_bandwidth(
        self,
        raw_similarity: Sequence[float] | np.ndarray,
        target_scores: Sequence[float] | np.ndarray,
        bandwidth: float,
    ) -> float:
        """Return exact leave-one-out cross-validation MSE for one bandwidth."""
        x, y = _validate_training_arrays(raw_similarity, target_scores)
        _validate_bandwidth(bandwidth)
        return _leave_one_out_mse(
            x,
            y,
            float(bandwidth),
            self.configuration.chunk_size,
        )

    def select_bandwidth(
        self,
        raw_similarity: Sequence[float] | np.ndarray,
        target_scores: Sequence[float] | np.ndarray,
    ) -> BandwidthSearchResult:
        """Select bandwidth by the approved coarse-to-fine LOOCV search."""
        x, y = _validate_training_arrays(raw_similarity, target_scores)
        coarse_grid = _bandwidth_grid(
            self.configuration.min_bandwidth,
            self.configuration.max_bandwidth,
            self.configuration.coarse_step,
        )
        coarse_scores = self._evaluate_grid(x, y, coarse_grid)
        coarse_index = _select_best_index(
            coarse_grid,
            coarse_scores,
            self.configuration.tie_tolerance,
        )
        coarse_bandwidth = float(coarse_grid[coarse_index])
        coarse_cv_mse = float(coarse_scores[coarse_index])

        refine_start = max(
            self.configuration.min_bandwidth,
            coarse_bandwidth - self.configuration.refine_radius,
        )
        refine_end = min(
            self.configuration.max_bandwidth,
            coarse_bandwidth + self.configuration.refine_radius,
        )
        refine_grid = _bandwidth_grid(
            refine_start,
            refine_end,
            self.configuration.refine_step,
        )
        grid = np.unique(np.concatenate((coarse_grid, refine_grid)))
        scores = self._evaluate_grid(x, y, grid)
        best_index = _select_best_index(
            grid,
            scores,
            self.configuration.tie_tolerance,
        )
        return BandwidthSearchResult(
            bandwidth=float(grid[best_index]),
            cv_mse=float(scores[best_index]),
            coarse_bandwidth=coarse_bandwidth,
            coarse_cv_mse=coarse_cv_mse,
        )

    def _evaluate_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bandwidths: np.ndarray,
    ) -> np.ndarray:
        """Evaluate all candidate bandwidths in deterministic order."""
        scores = np.empty(bandwidths.size, dtype=np.float64)
        for index, bandwidth in enumerate(bandwidths):
            scores[index] = self.cross_validate_bandwidth(x, y, float(bandwidth))
        if not np.any(np.isfinite(scores)):
            raise CalibrationError("No candidate bandwidth has finite LOOCV MSE")
        return scores

    def configuration_report(self) -> str:
        """Return the active calibration configuration as readable text."""
        selected = (
            "not selected"
            if self._bandwidth is None
            else f"{self._bandwidth:.6f}"
        )
        return (
            "Local-linear Gaussian calibration | "
            f"bandwidth={selected} | "
            f"search=[{self.configuration.min_bandwidth:.3f}, "
            f"{self.configuration.max_bandwidth:.3f}] | "
            f"coarse_step={self.configuration.coarse_step:.3f} | "
            f"refine_step={self.configuration.refine_step:.3f} | "
            "LOOCV=training-only | target=[1,5] | "
            "test labels excluded from fitting and bandwidth selection"
        )


def calibrate_sick_predictions(
    train_raw_similarity: Sequence[float] | np.ndarray,
    train_target_scores: Sequence[float] | np.ndarray,
    test_raw_similarity: Sequence[float] | np.ndarray,
    configuration: CalibrationConfiguration | None = None,
) -> tuple[LocalLinearCalibrator, np.ndarray]:
    """Fit only on SICK training data and predict SICK test similarities."""
    calibrator = LocalLinearCalibrator(configuration)
    calibrator.fit(train_raw_similarity, train_target_scores)
    return calibrator, calibrator.predict(test_raw_similarity)


def calculate_metrics(
    predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> CalibrationMetrics:
    """Calculate MSE, Pearson, and Spearman on one common score scale."""
    predicted, target = _validate_metric_arrays(predictions, targets)
    metrics = CalibrationMetrics(
        mse=float(np.mean(np.square(predicted - target))),
        pearson=float(pearsonr(predicted, target)[0]),
        spearman=float(spearmanr(predicted, target)[0]),
    )
    _require_finite(
        "calibration metrics",
        np.asarray(metrics.as_tuple(), dtype=np.float64),
    )
    return metrics


def calculate_raw_metrics(
    raw_similarity: Sequence[float] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
) -> CalibrationMetrics:
    """Evaluate raw MaLSTM similarity after the original [0,1] to [1,5] scale."""
    raw = _validate_raw_similarity(raw_similarity)
    return calculate_metrics(raw * 4.0 + 1.0, target_scores)


def calculate_calibrated_metrics(
    calibrated_predictions: Sequence[float] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
) -> CalibrationMetrics:
    """Evaluate calibrated predictions on the original [1, 5] scale."""
    return calculate_metrics(calibrated_predictions, target_scores)


def _validate_training_arrays(
    raw_similarity: Sequence[float] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate training arrays and preserve float64 precision."""
    x = _validate_raw_similarity(raw_similarity)
    y = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    if y.size == 0:
        raise CalibrationError("target_scores must be non-empty")
    if x.shape != y.shape:
        raise CalibrationError(
            f"raw_similarity and target_scores must have identical shapes; "
            f"got {x.shape} and {y.shape}"
        )
    _require_finite("target_scores", y)
    if np.any((y < TARGET_SCORE_MIN) | (y > TARGET_SCORE_MAX)):
        raise CalibrationError("target_scores must lie in the [1, 5] range")
    if x.size < 3:
        raise CalibrationError("At least three training observations are required")
    if np.unique(x).size < 2:
        raise CalibrationError("At least two distinct raw similarity values are required")
    return x, y


def _validate_raw_similarity(
    raw_similarity: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Validate raw MaLSTM similarities in the [0, 1] range."""
    values = np.asarray(raw_similarity, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise CalibrationError("raw_similarity must be non-empty")
    _require_finite("raw_similarity", values)
    if np.any((values < RAW_SIMILARITY_MIN) | (values > RAW_SIMILARITY_MAX)):
        raise CalibrationError("raw_similarity must lie in the [0, 1] range")
    return values


def _validate_metric_arrays(
    predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate same-shaped finite metric arrays."""
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.size < 2 or target.size < 2:
        raise CalibrationError("At least two observations are required for metrics")
    if predicted.shape != target.shape:
        raise CalibrationError(
            f"Metric arrays must have identical shapes; got "
            f"{predicted.shape} and {target.shape}"
        )
    _require_finite("predictions", predicted)
    _require_finite("targets", target)
    if np.all(predicted == predicted[0]) or np.all(target == target[0]):
        raise CalibrationError("Correlation is undefined for constant arrays")
    return predicted, target


def _require_finite(name: str, values: np.ndarray) -> None:
    """Raise a descriptive error when values contain NaN or infinity."""
    if not np.all(np.isfinite(values)):
        raise CalibrationError(f"{name} contains NaN or infinite values")


def _validate_bandwidth(bandwidth: float) -> None:
    """Validate a positive finite bandwidth."""
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise CalibrationError("bandwidth must be a finite positive value")


def _bandwidth_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Build an inclusive deterministic bandwidth grid."""
    if stop < start:
        raise CalibrationError("Bandwidth grid stop must be >= start")
    values = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    values = np.clip(values, start, stop)
    if values.size == 0 or not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return np.unique(np.round(values, decimals=12))


def _select_best_index(
    bandwidths: np.ndarray,
    scores: np.ndarray,
    tolerance: float,
) -> int:
    """Select minimum CV score, preferring smaller bandwidth within tolerance."""
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise CalibrationError("No candidate bandwidth has finite LOOCV MSE")
    best_score = float(np.min(scores[finite]))
    eligible = np.flatnonzero(finite & (scores <= best_score + tolerance))
    if eligible.size == 0:
        raise CalibrationError("Unable to select a bandwidth")
    return int(eligible[np.argmin(bandwidths[eligible])])


def _leave_one_out_mse(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    chunk_size: int,
) -> float:
    """Compute exact LOOCV MSE using duplicate-aware local-linear fits.

    A direct implementation would fit n regressions for every bandwidth.
    Instead, each leave-one-out fit removes one observation from the weighted
    sufficient statistics.  This is algebraically identical and keeps the
    expensive work in vectorized NumPy operations.
    """
    unique_x, inverse, counts = np.unique(
        x,
        return_inverse=True,
        return_counts=True,
    )
    counts = counts.astype(np.float64)
    members = [np.flatnonzero(inverse == index) for index in range(unique_x.size)]
    group_sum_y = np.bincount(inverse, weights=y).astype(np.float64)
    squared_error = 0.0

    for start in range(0, unique_x.size, chunk_size):
        stop = min(start + chunk_size, unique_x.size)
        query = unique_x[start:stop]
        deltas = unique_x[None, :] - query[:, None]
        weights = np.exp(-0.5 * np.square(deltas / bandwidth))
        s0 = weights @ counts
        s1 = (weights * deltas) @ counts
        s2 = (weights * np.square(deltas)) @ counts
        t0 = weights @ group_sum_y
        t1 = (weights * deltas) @ group_sum_y

        for local_index, group_index in enumerate(range(start, stop)):
            # At the query point x_g, every observation in group g has
            # distance zero and therefore kernel weight exactly one.
            s0_loo = s0[local_index] - 1.0
            s1_loo = s1[local_index]
            s2_loo = s2[local_index]
            denominator = s2_loo - (s1_loo * s1_loo) / s0_loo
            if (
                not np.isfinite(denominator)
                or denominator <= np.finfo(np.float64).tiny
            ):
                # This candidate cannot support a valid local-linear LOO fit
                # at this boundary/group.  Returning infinity excludes it
                # deterministically without manufacturing a fallback model.
                return float("inf")

            group_y = y[members[group_index]]
            t0_loo = t0[local_index] - group_y
            t1_loo = t1[local_index]
            mean_d = s1_loo / s0_loo
            mean_y = t0_loo / s0_loo
            covariance = t1_loo - s1_loo * mean_y
            slope = covariance / denominator
            prediction = mean_y - slope * mean_d
            residual = prediction - group_y
            squared_error += float(np.sum(np.square(residual)))

    if not np.isfinite(squared_error):
        return float("inf")
    return float(squared_error / y.size)


def _local_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    bandwidth: float,
    chunk_size: int,
) -> np.ndarray:
    """Predict local-linear intercepts for arbitrary query points."""
    predictions = np.empty(x_query.size, dtype=np.float64)
    for start in range(0, x_query.size, chunk_size):
        stop = min(start + chunk_size, x_query.size)
        query = x_query[start:stop]
        deltas = x_train[None, :] - query[:, None]
        weights = np.exp(-0.5 * np.square(deltas / bandwidth))
        s0 = np.sum(weights, axis=1)
        s1 = np.sum(weights * deltas, axis=1)
        s2 = np.sum(weights * np.square(deltas), axis=1)
        t0 = weights @ y_train
        t1 = (weights * deltas) @ y_train
        denominator = s2 - (s1 * s1) / s0
        if np.any(denominator <= np.finfo(np.float64).tiny):
            raise CalibrationError(
                "Local-linear weighted system is singular for prediction; "
                "use a bandwidth with sufficient local support"
            )
        mean_d = s1 / s0
        mean_y = t0 / s0
        covariance = t1 - s1 * mean_y
        slope = covariance / denominator
        predictions[start:stop] = mean_y - slope * mean_d
    _require_finite("local-linear predictions", predictions)
    return predictions


def _run_self_tests() -> None:
    """Run lightweight executable tests for the calibration module."""
    rng = np.random.default_rng(20261007)
    x_train = np.linspace(0.0, 1.0, 48)
    y_train = np.clip(
        1.0 + 4.0 * (0.15 + 0.7 * x_train + 0.12 * np.sin(2.0 * np.pi * x_train))
        + rng.normal(0.0, 0.025, x_train.size),
        1.0,
        5.0,
    )
    configuration = CalibrationConfiguration(
        min_bandwidth=0.02,
        max_bandwidth=0.20,
        coarse_step=0.02,
        refine_step=0.01,
        refine_radius=0.02,
    )
    calibrator = LocalLinearCalibrator(configuration).fit(x_train, y_train)
    predictions = calibrator.predict(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    assert np.all(np.isfinite(predictions))
    assert calibrator.bandwidth > 0.0

    duplicate_x = np.array([0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0])
    duplicate_y = np.array([1.1, 1.2, 2.0, 3.0, 3.1, 4.0, 4.8, 4.9])
    duplicate_config = CalibrationConfiguration(
        min_bandwidth=0.05,
        max_bandwidth=0.20,
        coarse_step=0.05,
        refine_step=0.02,
        refine_radius=0.02,
    )
    duplicate_model = LocalLinearCalibrator(duplicate_config).fit(
        duplicate_x,
        duplicate_y,
    )
    assert np.all(np.isfinite(duplicate_model.predict(duplicate_x)))

    boundary_model = LocalLinearCalibrator(
        CalibrationConfiguration(
            min_bandwidth=0.05,
            max_bandwidth=0.15,
            coarse_step=0.05,
            refine_step=0.01,
            refine_radius=0.01,
        )
    ).fit(np.linspace(0.05, 0.95, 30), 1.0 + 4.0 * np.linspace(0.05, 0.95, 30))
    boundary_predictions = boundary_model.predict(np.array([0.0, 0.02, 0.98, 1.0]))
    assert np.all(np.isfinite(boundary_predictions))

    # Test/train separation: test labels are never passed to fit or CV.
    test_x = np.array([0.0, 0.02, 0.98, 1.0])
    separated_prediction = boundary_model.predict(test_x)
    altered_test_labels = np.array([5.0, 5.0, 1.0, 1.0])
    del altered_test_labels
    np.testing.assert_array_equal(separated_prediction, boundary_model.predict(test_x))

    # Deliberately distorted raw predictions should improve after calibration.
    distorted_x = np.linspace(0.0, 1.0, 60)
    distorted_target = 1.0 + 4.0 * distorted_x
    distorted_raw = np.clip((distorted_target - 0.8) / 5.0, 0.0, 1.0)
    distorted_test_x = np.linspace(0.03, 0.97, 20)
    distorted_test_target = 1.0 + 4.0 * distorted_test_x
    distorted_test_raw = np.clip(
        (distorted_test_target - 0.8) / 5.0,
        0.0,
        1.0,
    )
    distorted_model = LocalLinearCalibrator(
        CalibrationConfiguration(
            min_bandwidth=0.03,
            max_bandwidth=0.18,
            coarse_step=0.03,
            refine_step=0.01,
            refine_radius=0.02,
        )
    ).fit(distorted_raw, distorted_target)
    raw_metrics = calculate_raw_metrics(distorted_test_raw, distorted_test_target)
    calibrated_metrics = calculate_calibrated_metrics(
        distorted_model.predict(distorted_test_raw),
        distorted_test_target,
    )
    assert calibrated_metrics.mse < raw_metrics.mse
    print("Calibration self-tests: PASS")
    print(distorted_model.configuration_report())


if __name__ == "__main__":
    _run_self_tests()
