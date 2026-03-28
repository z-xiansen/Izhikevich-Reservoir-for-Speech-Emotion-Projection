"""SER reservoir package based on Izhikevich SNN."""

from .config import ExperimentConfig
from .pipeline import run_pipeline

__all__ = ["ExperimentConfig", "run_pipeline"]

