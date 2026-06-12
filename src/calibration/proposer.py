"""Parameter proposers for autoresearch calibration.

A proposer suggests parameter configurations to try next, given the search
space and the history of previous (config, loss) pairs. This module provides:

  - `ParameterSpec`: declarative search-space entry with literature range
  - `SearchSpace`: collection of specs with random/validate/clip helpers
  - `RandomProposer`: uniform random search baseline (no learning)
  - `LatinHypercubeProposer`: LHS for better initial coverage (no scipy)
  - `GridProposer`: grid search over a small subset of dims
  - `BayesianProposer`: optional scikit-optimize backed (graceful fallback)

All proposers share a common interface:

    class ProposerBase:
        def propose(self, history: list[Trial]) -> dict[str, Any]: ...

where `Trial` is `(config_dict, loss_scalar)`.

Determinism: all proposers accept `seed` for reproducible sequences.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Search space primitives
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpec:
    """One dimension of the autoresearch search space.

    Fields:
      name: identifier (can be dotted path, e.g. 'adhd_inattentive.anxiety')
      lo, hi: inclusive bounds from literature
      kind: 'float' | 'int' | 'choice'
      choices: only used when kind == 'choice'
      default: starting-point value (used by proposers that need it)
      source: literature citation for documentation
    """

    name: str
    lo: float = 0.0
    hi: float = 1.0
    kind: str = "float"
    choices: list[Any] = field(default_factory=list)
    default: float | int | str | None = None
    source: str = ""

    def clip(self, value):
        if self.kind == "choice":
            return value if value in self.choices else self.choices[0]
        value = max(self.lo, min(self.hi, float(value)))
        if self.kind == "int":
            return int(round(value))
        return value

    def sample_random(self, rng: random.Random):
        if self.kind == "choice":
            return rng.choice(self.choices)
        if self.kind == "int":
            return rng.randint(int(self.lo), int(self.hi))
        return rng.uniform(self.lo, self.hi)

    def is_valid(self, value) -> bool:
        if self.kind == "choice":
            return value in self.choices
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return self.lo <= v <= self.hi


@dataclass
class SearchSpace:
    """Collection of `ParameterSpec` entries."""

    specs: list[ParameterSpec] = field(default_factory=list)

    def add(self, spec: ParameterSpec) -> None:
        self.specs.append(spec)

    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def random_config(self, rng: random.Random) -> dict[str, Any]:
        return {s.name: s.sample_random(rng) for s in self.specs}

    def clip_config(self, config: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in self.specs:
            if s.name in config:
                out[s.name] = s.clip(config[s.name])
            elif s.default is not None:
                out[s.name] = s.default
        return out

    def default_config(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in self.specs:
            if s.default is not None:
                out[s.name] = s.default
            else:
                out[s.name] = (s.lo + s.hi) / 2 if s.kind != "choice" else s.choices[0]
        return out

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        for s in self.specs:
            if s.name not in config:
                errors.append(f"missing key: {s.name}")
                continue
            if not s.is_valid(config[s.name]):
                errors.append(
                    f"{s.name}={config[s.name]} outside [{s.lo}, {s.hi}]"
                    if s.kind != "choice"
                    else f"{s.name}={config[s.name]} not in {s.choices}"
                )
        return (len(errors) == 0, errors)

    def __len__(self) -> int:
        return len(self.specs)


# ---------------------------------------------------------------------------
# Trial history
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """One evaluation of a configuration."""

    config: dict[str, Any]
    loss: float
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Proposer interface + implementations
# ---------------------------------------------------------------------------


class ProposerBase:
    """Abstract proposer. Subclasses implement `propose`."""

    def __init__(self, space: SearchSpace, seed: int | None = None) -> None:
        self.space = space
        self.rng = random.Random(seed)
        self._iter = 0

    def propose(self, history: list[Trial]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._iter = 0


class RandomProposer(ProposerBase):
    """Uniform random search over the search space. Baseline."""

    def propose(self, history: list[Trial]) -> dict[str, Any]:
        self._iter += 1
        return self.space.random_config(self.rng)


class LatinHypercubeProposer(ProposerBase):
    """Latin hypercube sampling for better coverage in early iterations.

    Uses a single LHS block of `n_initial` samples, then falls back to
    random. Dependency-free implementation.
    """

    def __init__(
        self,
        space: SearchSpace,
        n_initial: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__(space, seed)
        self.n_initial = max(1, n_initial)
        self._lhs_queue: list[dict[str, Any]] = []
        self._regenerate_lhs()

    def _regenerate_lhs(self) -> None:
        """Build a latin hypercube block sized `n_initial`."""
        n = self.n_initial
        specs = self.space.specs
        # For each dim: permuted bucket indices in [0, n)
        dim_grids: list[list[int]] = []
        for _ in specs:
            idxs = list(range(n))
            self.rng.shuffle(idxs)
            dim_grids.append(idxs)

        configs: list[dict[str, Any]] = []
        for row in range(n):
            cfg: dict[str, Any] = {}
            for j, spec in enumerate(specs):
                bucket = dim_grids[j][row]
                if spec.kind == "choice":
                    cfg[spec.name] = spec.choices[bucket % len(spec.choices)]
                    continue
                # Uniform within bucket
                frac = (bucket + self.rng.random()) / n
                val = spec.lo + frac * (spec.hi - spec.lo)
                cfg[spec.name] = int(round(val)) if spec.kind == "int" else val
            configs.append(cfg)
        self._lhs_queue = configs

    def propose(self, history: list[Trial]) -> dict[str, Any]:
        self._iter += 1
        if self._lhs_queue:
            return self._lhs_queue.pop(0)
        return self.space.random_config(self.rng)


class GridProposer(ProposerBase):
    """Grid search over a small subset of dimensions (for debugging)."""

    def __init__(
        self,
        space: SearchSpace,
        levels: int = 3,
        max_dims: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__(space, seed)
        self.levels = levels
        self.max_dims = max_dims
        self._queue: list[dict[str, Any]] = []
        self._build_grid()

    def _build_grid(self) -> None:
        # Take first max_dims dims only
        dims = self.space.specs[: self.max_dims]
        default = self.space.default_config()
        grid_points: list[list[tuple[str, Any]]] = []
        for spec in dims:
            if spec.kind == "choice":
                vals = list(spec.choices)
            elif spec.kind == "int":
                step = max(1, (int(spec.hi) - int(spec.lo)) // max(1, self.levels - 1))
                vals = list(range(int(spec.lo), int(spec.hi) + 1, step))
            else:
                step = (spec.hi - spec.lo) / max(1, self.levels - 1)
                vals = [spec.lo + i * step for i in range(self.levels)]
            grid_points.append([(spec.name, v) for v in vals])

        # Cartesian product
        def _recurse(idx: int, acc: dict[str, Any]):
            if idx == len(grid_points):
                cfg = dict(default)
                cfg.update(acc)
                self._queue.append(cfg)
                return
            for name, val in grid_points[idx]:
                acc[name] = val
                _recurse(idx + 1, acc)
                del acc[name]

        if grid_points:
            _recurse(0, {})

    def propose(self, history: list[Trial]) -> dict[str, Any]:
        self._iter += 1
        if self._queue:
            return self._queue.pop(0)
        return self.space.default_config()


class BayesianProposer(ProposerBase):
    """scikit-optimize `gp_minimize` wrapper with graceful fallback to random.

    Only activated if scikit-optimize is installed; otherwise behaves as
    `RandomProposer` and exposes `available=False` for the orchestrator
    to log.
    """

    def __init__(
        self,
        space: SearchSpace,
        seed: int | None = None,
        n_random_starts: int = 5,
    ) -> None:
        super().__init__(space, seed)
        self.n_random_starts = n_random_starts
        try:
            from skopt import Optimizer  # type: ignore
            from skopt.space import Real, Integer, Categorical  # type: ignore

            dims = []
            for spec in space.specs:
                if spec.kind == "choice":
                    dims.append(Categorical(spec.choices, name=spec.name))
                elif spec.kind == "int":
                    dims.append(Integer(int(spec.lo), int(spec.hi), name=spec.name))
                else:
                    dims.append(Real(spec.lo, spec.hi, name=spec.name))
            self._opt = Optimizer(
                dimensions=dims,
                random_state=seed,
                n_initial_points=n_random_starts,
                base_estimator="GP",
            )
            self._names = space.names()
            self.available = True
        except Exception:
            self._opt = None
            self._names = space.names()
            self.available = False

    def propose(self, history: list[Trial]) -> dict[str, Any]:
        self._iter += 1
        if not self.available or self._opt is None:
            return self.space.random_config(self.rng)
        # Tell the optimizer about the most recent trial (if any)
        if history:
            last = history[-1]
            try:
                point = [last.config[name] for name in self._names]
                self._opt.tell(point, last.loss)
            except Exception:
                pass
        try:
            next_point = self._opt.ask()
            return {name: val for name, val in zip(self._names, next_point)}
        except Exception:
            return self.space.random_config(self.rng)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_proposer(
    kind: str,
    space: SearchSpace,
    seed: int | None = None,
    **kwargs,
) -> ProposerBase:
    """Construct a proposer by name.

    kind ∈ {'random', 'lhs', 'grid', 'bayes'}
    """
    k = kind.lower()
    if k == "random":
        return RandomProposer(space, seed=seed)
    if k == "lhs":
        return LatinHypercubeProposer(
            space, n_initial=kwargs.get("n_initial", 10), seed=seed
        )
    if k == "grid":
        return GridProposer(
            space,
            levels=kwargs.get("levels", 3),
            max_dims=kwargs.get("max_dims", 3),
            seed=seed,
        )
    if k in ("bayes", "bayesian", "skopt"):
        return BayesianProposer(
            space,
            seed=seed,
            n_random_starts=kwargs.get("n_random_starts", 5),
        )
    raise ValueError(
        f"Unknown proposer kind: {kind!r}. "
        f"Valid: random, lhs, grid, bayes"
    )
