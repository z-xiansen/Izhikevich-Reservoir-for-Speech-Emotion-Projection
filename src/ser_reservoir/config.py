from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 7
    workspace: Path = Path(".")

    # Data
    dataset_id: str = "ejlok1/toronto-emotional-speech-set-tess"
    hf_dataset_id: str = "TwinkStart/TESS"
    raw_data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    max_samples: int | None = 420
    max_samples_per_emotion: int | None = 70

    # Audio / MFCC
    sample_rate: int = 22050
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    # Reservoir topology
    grid_side: int = 20
    exc_ratio: float = 0.8
    base_conn_prob: float = 0.18
    distance_lambda: float = 3.2
    local_inh_radius: float = 2.0
    local_inh_prob: float = 0.45
    init_w_exc_mean: float = 0.85
    init_w_exc_std: float = 0.18
    init_w_inh_mean: float = 1.10
    init_w_inh_std: float = 0.20

    # Izhikevich + coding
    dt_ms: float = 1.0
    frame_repeat: int = 2
    input_gain: float = 14.0
    v_thresh: float = 30.0
    v_init: float = -65.0
    u_init_factor: float = 1.0

    # STDP
    stdp_a_plus: float = 0.0040
    stdp_a_minus: float = 0.0043
    stdp_tau_pre: float = 20.0
    stdp_tau_post: float = 20.0
    w_exc_max: float = 2.0
    w_inh_min: float = -2.5

    # STP (TM model)
    stp_u_exc: float = 0.22
    stp_tau_d_exc: float = 700.0
    stp_tau_f_exc: float = 50.0
    stp_u_inh: float = 0.06
    stp_tau_d_inh: float = 120.0
    stp_tau_f_inh: float = 760.0

    # Intrinsic plasticity (edge-of-chaos tuning)
    ip_lr: float = 0.0012
    ip_target_rate: float = 0.045
    ip_bias_min: float = -4.0
    ip_bias_max: float = 4.0

    # t-SNE
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1200

    # runtime
    verbose: bool = True
    save_model: bool = False
    model_path: Path = Path("results/reservoir_model.npz")
    load_model_path: Path | None = None

    def resolve(self) -> "ExperimentConfig":
        self.workspace = self.workspace.resolve()
        self.raw_data_dir = (self.workspace / self.raw_data_dir).resolve()
        self.processed_dir = (self.workspace / self.processed_dir).resolve()
        self.results_dir = (self.workspace / self.results_dir).resolve()
        self.model_path = (self.workspace / self.model_path).resolve()
        if self.load_model_path is not None:
            self.load_model_path = (self.workspace / self.load_model_path).resolve()
        return self

    @property
    def n_neurons(self) -> int:
        return self.grid_side * self.grid_side

    @property
    def tess_target_dir(self) -> Path:
        return self.raw_data_dir / "TESS"

    @property
    def state_feature_path(self) -> Path:
        return self.processed_dir / "reservoir_states.npz"

    @property
    def tsne_plot_path(self) -> Path:
        return self.results_dir / "tsne_emotion_clusters.png"

    @property
    def metrics_path(self) -> Path:
        return self.results_dir / "run_summary.yaml"

    @property
    def label_order(self) -> list[str]:
        return [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "pleasant_surprise",
            "sad",
        ]

    def as_dict(self) -> dict:
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in self.__dict__.items()
        }
