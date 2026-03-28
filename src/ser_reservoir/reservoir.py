from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig

_MODEL_FORMAT = "ser_reservoir_model_v1"
_VALIDATED_CFG_FIELDS = (
    "n_mfcc",
    "dt_ms",
    "v_thresh",
    "v_init",
    "u_init_factor",
    "stdp_a_plus",
    "stdp_a_minus",
    "stdp_tau_pre",
    "stdp_tau_post",
    "w_exc_max",
    "w_inh_min",
    "stp_u_exc",
    "stp_tau_d_exc",
    "stp_tau_f_exc",
    "stp_u_inh",
    "stp_tau_d_inh",
    "stp_tau_f_inh",
    "ip_lr",
    "ip_target_rate",
    "ip_bias_min",
    "ip_bias_max",
)


@dataclass(slots=True, frozen=True)
class ReservoirStats:
    n_neurons: int
    n_edges: int
    excitatory_neurons: int
    inhibitory_neurons: int
    excitatory_edges: int
    inhibitory_edges: int


class IzhikevichReservoir:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = cfg.n_neurons

        self._init_neuron_parameters()
        self._build_topology()
        self._build_input_projection()
        self._init_plasticity_state()
        self.reset_trial_state(keep_intrinsic_bias=True)

    def present(self, mfcc: np.ndarray) -> tuple[np.ndarray, float]:
        if mfcc.ndim != 2:
            raise ValueError(f"Expected MFCC shape [n_mfcc, frames], got {mfcc.shape}")
        if mfcc.shape[0] != self.cfg.n_mfcc:
            raise ValueError(f"Expected {self.cfg.n_mfcc} MFCC channels, got {mfcc.shape[0]}")

        self.reset_trial_state(keep_intrinsic_bias=True)

        total_steps = int(mfcc.shape[1] * self.cfg.frame_repeat)
        if total_steps <= 0:
            raise ValueError("Input MFCC has zero frames.")

        v_accum = np.zeros(self.n, dtype=np.float64)
        spike_accum = np.zeros(self.n, dtype=np.float64)

        for frame_idx in range(mfcc.shape[1]):
            frame_current = mfcc[:, frame_idx].astype(np.float32, copy=False)
            for _ in range(self.cfg.frame_repeat):
                spikes = self._step(frame_current)
                v_accum += self.v
                spike_accum += spikes

        mean_v = (v_accum / total_steps).astype(np.float32)
        firing_rate = (spike_accum / total_steps).astype(np.float32)
        feature = np.concatenate([mean_v, firing_rate], axis=0)
        global_rate = float(firing_rate.mean())
        return feature, global_rate

    def stats(self) -> ReservoirStats:
        return ReservoirStats(
            n_neurons=self.n,
            n_edges=int(self.w.size),
            excitatory_neurons=int(self.is_excitatory.sum()),
            inhibitory_neurons=int((~self.is_excitatory).sum()),
            excitatory_edges=int(self.edge_from_excit.sum()),
            inhibitory_edges=int((~self.edge_from_excit).sum()),
        )

    def save_model(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            model_metadata_json=np.array(
                json.dumps(self._build_model_metadata(), ensure_ascii=False)
            ),
            n=np.array(self.n, dtype=np.int32),
            is_excitatory=self.is_excitatory.astype(np.bool_),
            exc_idx=self.exc_idx.astype(np.int32),
            inh_idx=self.inh_idx.astype(np.int32),
            a=self.a.astype(np.float32),
            b=self.b.astype(np.float32),
            c=self.c.astype(np.float32),
            d=self.d.astype(np.float32),
            post_idx=self.post_idx.astype(np.int32),
            pre_idx=self.pre_idx.astype(np.int32),
            w=self.w.astype(np.float32),
            edge_from_excit=self.edge_from_excit.astype(np.bool_),
            input_neuron_idx=self.input_neuron_idx.astype(np.int32),
            intrinsic_bias=self.intrinsic_bias.astype(np.float32),
        )

    def save_checkpoint(self, path: Path) -> None:
        self.save_model(path)

    @classmethod
    def from_model(cls, cfg: ExperimentConfig, path: Path) -> "IzhikevichReservoir":
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with np.load(model_path, allow_pickle=False) as data:
            metadata = cls._read_model_metadata(data)
            n = int(np.asarray(data["n"]).item())
            if metadata is not None:
                cls._validate_model_metadata(cfg, metadata, model_path)
            arrays = {
                "is_excitatory": data["is_excitatory"].astype(bool, copy=True),
                "exc_idx": data["exc_idx"].astype(np.int32, copy=True),
                "inh_idx": data["inh_idx"].astype(np.int32, copy=True),
                "a": data["a"].astype(np.float32, copy=True),
                "b": data["b"].astype(np.float32, copy=True),
                "c": data["c"].astype(np.float32, copy=True),
                "d": data["d"].astype(np.float32, copy=True),
                "post_idx": data["post_idx"].astype(np.int32, copy=True),
                "pre_idx": data["pre_idx"].astype(np.int32, copy=True),
                "w": data["w"].astype(np.float32, copy=True),
                "edge_from_excit": data["edge_from_excit"].astype(bool, copy=True),
                "input_neuron_idx": data["input_neuron_idx"].astype(np.int32, copy=True),
                "intrinsic_bias": data["intrinsic_bias"].astype(np.float32, copy=True),
            }

        if n != cfg.n_neurons:
            raise ValueError(
                f"Model n={n} does not match cfg.n_neurons={cfg.n_neurons}."
            )

        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj.rng = np.random.default_rng(cfg.seed)
        obj.n = n

        obj.is_excitatory = arrays["is_excitatory"]
        obj.exc_idx = arrays["exc_idx"]
        obj.inh_idx = arrays["inh_idx"]

        obj.a = arrays["a"]
        obj.b = arrays["b"]
        obj.c = arrays["c"]
        obj.d = arrays["d"]

        obj.post_idx = arrays["post_idx"]
        obj.pre_idx = arrays["pre_idx"]
        obj.w = arrays["w"]
        obj.edge_from_excit = arrays["edge_from_excit"]
        obj.input_neuron_idx = arrays["input_neuron_idx"]
        obj.intrinsic_bias = arrays["intrinsic_bias"]

        obj.stp_U = np.where(
            obj.edge_from_excit, cfg.stp_u_exc, cfg.stp_u_inh
        ).astype(np.float32)
        obj.stp_tau_d = np.where(
            obj.edge_from_excit, cfg.stp_tau_d_exc, cfg.stp_tau_d_inh
        ).astype(np.float32)
        obj.stp_tau_f = np.where(
            obj.edge_from_excit, cfg.stp_tau_f_exc, cfg.stp_tau_f_inh
        ).astype(np.float32)

        obj.pre_trace = np.zeros(obj.n, dtype=np.float32)
        obj.post_trace = np.zeros(obj.n, dtype=np.float32)
        obj.stp_u = obj.stp_U.copy()
        obj.stp_x = np.ones(obj.w.shape[0], dtype=np.float32)
        obj.v = np.full(obj.n, cfg.v_init, dtype=np.float32)
        obj.recovery_u = (obj.b * obj.v * cfg.u_init_factor).astype(np.float32)
        obj.syn_current = np.zeros(obj.n, dtype=np.float32)

        obj._validate_loaded_state(model_path)
        return obj

    @classmethod
    def from_checkpoint(cls, cfg: ExperimentConfig, path: Path) -> "IzhikevichReservoir":
        return cls.from_model(cfg, path)

    def _validate_loaded_state(self, path: Path) -> None:
        if self.is_excitatory.shape != (self.n,):
            raise ValueError(f"Invalid is_excitatory shape in model file: {path}")
        if self.a.shape != (self.n,) or self.b.shape != (self.n,):
            raise ValueError(f"Invalid neuron parameter shapes in model file: {path}")
        if self.input_neuron_idx.shape != (self.cfg.n_mfcc,):
            raise ValueError(
                f"Invalid input_neuron_idx shape in model file: {path}; "
                f"expected ({self.cfg.n_mfcc},), got {self.input_neuron_idx.shape}."
            )
        n_edges = self.w.shape[0]
        if self.pre_idx.shape != (n_edges,) or self.post_idx.shape != (n_edges,):
            raise ValueError(f"Invalid edge index shapes in model file: {path}")
        if self.edge_from_excit.shape != (n_edges,):
            raise ValueError(f"Invalid edge_from_excit shape in model file: {path}")
        if self.intrinsic_bias.shape != (self.n,):
            raise ValueError(f"Invalid intrinsic_bias shape in model file: {path}")

    def _build_model_metadata(self) -> dict:
        return {
            "format": _MODEL_FORMAT,
            "n_neurons": self.n,
            "validated_config": {
                field: getattr(self.cfg, field) for field in _VALIDATED_CFG_FIELDS
            },
            "runtime_config": {
                "frame_repeat": self.cfg.frame_repeat,
                "input_gain": self.cfg.input_gain,
                "seed": self.cfg.seed,
            },
        }

    @staticmethod
    def _read_model_metadata(data: np.lib.npyio.NpzFile) -> dict | None:
        if "model_metadata_json" not in data.files:
            return None
        raw = np.asarray(data["model_metadata_json"]).item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))

    @classmethod
    def _validate_model_metadata(
        cls, cfg: ExperimentConfig, metadata: dict, path: Path
    ) -> None:
        if metadata.get("format") != _MODEL_FORMAT:
            raise ValueError(
                f"Unsupported model format in {path}: {metadata.get('format')!r}."
            )

        stored_n = metadata.get("n_neurons")
        if stored_n is not None and int(stored_n) != cfg.n_neurons:
            raise ValueError(
                f"Model n={stored_n} does not match cfg.n_neurons={cfg.n_neurons}."
            )

        stored_cfg = metadata.get("validated_config")
        if not isinstance(stored_cfg, dict):
            raise ValueError(f"Model metadata is missing validated_config: {path}")

        mismatches: list[str] = []
        for field in _VALIDATED_CFG_FIELDS:
            expected = stored_cfg.get(field)
            actual = getattr(cfg, field)
            if isinstance(expected, float) or isinstance(actual, float):
                if not np.isclose(float(expected), float(actual)):
                    mismatches.append(f"{field}: model={expected}, current={actual}")
            elif expected != actual:
                mismatches.append(f"{field}: model={expected}, current={actual}")

        if mismatches:
            detail = "; ".join(mismatches[:6])
            if len(mismatches) > 6:
                detail += f"; ... and {len(mismatches) - 6} more"
            raise ValueError(
                f"Model config mismatch for {path}. "
                "Use the same dynamics/plasticity settings as the saved model. "
                f"Details: {detail}"
            )

    def reset_trial_state(self, keep_intrinsic_bias: bool) -> None:
        self.v = np.full(self.n, self.cfg.v_init, dtype=np.float32)
        self.recovery_u = (self.b * self.v * self.cfg.u_init_factor).astype(np.float32)
        self.syn_current = np.zeros(self.n, dtype=np.float32)
        self.pre_trace = np.zeros(self.n, dtype=np.float32)
        self.post_trace = np.zeros(self.n, dtype=np.float32)
        self.stp_u = self.stp_U.copy()
        self.stp_x = np.ones(self.w.shape[0], dtype=np.float32)
        if not keep_intrinsic_bias:
            self.intrinsic_bias = np.zeros(self.n, dtype=np.float32)

    def _step(self, frame_input: np.ndarray) -> np.ndarray:
        input_current = np.zeros(self.n, dtype=np.float32)
        input_current[self.input_neuron_idx] = self.cfg.input_gain * frame_input

        total_current = self.syn_current + input_current + self.intrinsic_bias

        dv = (
            0.04 * self.v * self.v
            + 5.0 * self.v
            + 140.0
            - self.recovery_u
            + total_current
        )
        self.v = self.v + self.cfg.dt_ms * dv.astype(np.float32)

        du = self.a * (self.b * self.v - self.recovery_u)
        self.recovery_u = self.recovery_u + self.cfg.dt_ms * du.astype(np.float32)

        spikes = self.v >= self.cfg.v_thresh
        if spikes.any():
            self.v[spikes] = self.c[spikes]
            self.recovery_u[spikes] = self.recovery_u[spikes] + self.d[spikes]

        self._update_intrinsic_plasticity(spikes)
        self._update_stdp(spikes)
        self._update_stp_and_syn_current(spikes)
        return spikes

    def _update_intrinsic_plasticity(self, spikes: np.ndarray) -> None:
        spike_f = spikes.astype(np.float32)
        delta = self.cfg.ip_lr * (self.cfg.ip_target_rate - spike_f)
        self.intrinsic_bias = np.clip(
            self.intrinsic_bias + delta,
            self.cfg.ip_bias_min,
            self.cfg.ip_bias_max,
        )

    def _update_stdp(self, spikes: np.ndarray) -> None:
        pre_decay = np.exp(-self.cfg.dt_ms / self.cfg.stdp_tau_pre).astype(np.float32)
        post_decay = np.exp(-self.cfg.dt_ms / self.cfg.stdp_tau_post).astype(np.float32)
        self.pre_trace *= pre_decay
        self.post_trace *= post_decay

        if spikes.any():
            active_pre_edges = spikes[self.pre_idx]
            if active_pre_edges.any():
                self.w[active_pre_edges] -= (
                    self.cfg.stdp_a_minus * self.post_trace[self.post_idx[active_pre_edges]]
                ).astype(np.float32)

            active_post_edges = spikes[self.post_idx]
            if active_post_edges.any():
                self.w[active_post_edges] += (
                    self.cfg.stdp_a_plus * self.pre_trace[self.pre_idx[active_post_edges]]
                ).astype(np.float32)

            self.w[self.edge_from_excit] = np.clip(
                self.w[self.edge_from_excit], 0.0, self.cfg.w_exc_max
            )
            self.w[~self.edge_from_excit] = np.clip(
                self.w[~self.edge_from_excit], self.cfg.w_inh_min, 0.0
            )

            spike_idx = np.flatnonzero(spikes)
            self.pre_trace[spike_idx] += 1.0
            self.post_trace[spike_idx] += 1.0

    def _update_stp_and_syn_current(self, spikes: np.ndarray) -> None:
        dt = self.cfg.dt_ms
        self.stp_u += ((self.stp_U - self.stp_u) * dt / self.stp_tau_f).astype(np.float32)
        self.stp_x += ((1.0 - self.stp_x) * dt / self.stp_tau_d).astype(np.float32)

        syn_in = np.zeros(self.n, dtype=np.float32)
        active = spikes[self.pre_idx]
        if active.any():
            self.stp_u[active] += (self.stp_U[active] * (1.0 - self.stp_u[active])).astype(np.float32)
            eff = self.w[active] * self.stp_u[active] * self.stp_x[active]
            self.stp_x[active] *= (1.0 - self.stp_u[active]).astype(np.float32)
            syn_in += np.bincount(
                self.post_idx[active],
                weights=eff.astype(np.float64),
                minlength=self.n,
            ).astype(np.float32)

        # Mild low-pass on synaptic current to keep richer temporal dynamics.
        self.syn_current = 0.70 * self.syn_current + syn_in

    def _init_neuron_parameters(self) -> None:
        n_exc = int(self.n * self.cfg.exc_ratio)
        all_idx = np.arange(self.n)
        self.rng.shuffle(all_idx)
        self.exc_idx = np.sort(all_idx[:n_exc])
        self.inh_idx = np.sort(all_idx[n_exc:])

        self.is_excitatory = np.zeros(self.n, dtype=bool)
        self.is_excitatory[self.exc_idx] = True

        self.a = np.empty(self.n, dtype=np.float32)
        self.b = np.empty(self.n, dtype=np.float32)
        self.c = np.empty(self.n, dtype=np.float32)
        self.d = np.empty(self.n, dtype=np.float32)

        # RS neurons (excitatory): slight heterogeneity around canonical values.
        self.a[self.exc_idx] = np.clip(
            self.rng.normal(0.02, 0.004, size=self.exc_idx.size), 0.01, 0.05
        ).astype(np.float32)
        self.b[self.exc_idx] = np.clip(
            self.rng.normal(0.20, 0.015, size=self.exc_idx.size), 0.15, 0.25
        ).astype(np.float32)
        self.c[self.exc_idx] = np.clip(
            self.rng.normal(-65.0, 2.5, size=self.exc_idx.size), -72.0, -58.0
        ).astype(np.float32)
        self.d[self.exc_idx] = np.clip(
            self.rng.normal(8.0, 1.0, size=self.exc_idx.size), 5.0, 11.0
        ).astype(np.float32)

        # FS neurons (inhibitory): slight heterogeneity around canonical values.
        self.a[self.inh_idx] = np.clip(
            self.rng.normal(0.10, 0.01, size=self.inh_idx.size), 0.06, 0.14
        ).astype(np.float32)
        self.b[self.inh_idx] = np.clip(
            self.rng.normal(0.20, 0.02, size=self.inh_idx.size), 0.15, 0.27
        ).astype(np.float32)
        self.c[self.inh_idx] = np.clip(
            self.rng.normal(-65.0, 2.0, size=self.inh_idx.size), -72.0, -58.0
        ).astype(np.float32)
        self.d[self.inh_idx] = np.clip(
            self.rng.normal(2.0, 0.4, size=self.inh_idx.size), 1.0, 3.5
        ).astype(np.float32)

    def _build_topology(self) -> None:
        side = self.cfg.grid_side
        coords = np.stack(
            np.meshgrid(np.arange(side), np.arange(side), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)
        distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(
            np.float32
        )

        p = self.cfg.base_conn_prob * np.exp(-distances / self.cfg.distance_lambda)
        np.fill_diagonal(p, 0.0)
        conn = self.rng.random((self.n, self.n)) < p

        if self.inh_idx.size > 0:
            local_dist = distances[:, self.inh_idx]
            near = local_dist <= self.cfg.local_inh_radius
            local_random = self.rng.random(near.shape) < self.cfg.local_inh_prob
            conn[:, self.inh_idx] |= near & local_random
            conn[np.arange(self.n), np.arange(self.n)] = False

        post_idx, pre_idx = np.where(conn)
        if post_idx.size == 0:
            raise RuntimeError("Reservoir has zero synapses. Increase base_conn_prob.")

        edge_from_excit = self.is_excitatory[pre_idx]

        w = np.empty(post_idx.size, dtype=np.float32)
        n_exc_edges = int(edge_from_excit.sum())
        n_inh_edges = int((~edge_from_excit).sum())
        if n_exc_edges > 0:
            w[edge_from_excit] = np.abs(
                self.rng.normal(
                    self.cfg.init_w_exc_mean,
                    self.cfg.init_w_exc_std,
                    size=n_exc_edges,
                )
            ).astype(np.float32)
        if n_inh_edges > 0:
            w[~edge_from_excit] = -np.abs(
                self.rng.normal(
                    self.cfg.init_w_inh_mean,
                    self.cfg.init_w_inh_std,
                    size=n_inh_edges,
                )
            ).astype(np.float32)

        self.post_idx = post_idx.astype(np.int32)
        self.pre_idx = pre_idx.astype(np.int32)
        self.w = w
        self.edge_from_excit = edge_from_excit

    def _build_input_projection(self) -> None:
        if self.exc_idx.size < self.cfg.n_mfcc:
            raise RuntimeError(
                f"Need at least {self.cfg.n_mfcc} excitatory neurons for input mapping, "
                f"but got {self.exc_idx.size}."
            )
        self.input_neuron_idx = np.sort(
            self.rng.choice(self.exc_idx, size=self.cfg.n_mfcc, replace=False)
        )

    def _init_plasticity_state(self) -> None:
        self.intrinsic_bias = np.zeros(self.n, dtype=np.float32)

        self.pre_trace = np.zeros(self.n, dtype=np.float32)
        self.post_trace = np.zeros(self.n, dtype=np.float32)

        self.stp_U = np.where(
            self.edge_from_excit, self.cfg.stp_u_exc, self.cfg.stp_u_inh
        ).astype(np.float32)
        self.stp_tau_d = np.where(
            self.edge_from_excit, self.cfg.stp_tau_d_exc, self.cfg.stp_tau_d_inh
        ).astype(np.float32)
        self.stp_tau_f = np.where(
            self.edge_from_excit, self.cfg.stp_tau_f_exc, self.cfg.stp_tau_f_inh
        ).astype(np.float32)
        self.stp_u = self.stp_U.copy()
        self.stp_x = np.ones(self.w.shape[0], dtype=np.float32)
