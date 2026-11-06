from __future__ import annotations

from pathlib import Path
from typing import Optional


class RunLogger:
    """Thin wrapper around TensorBoard's SummaryWriter that no-ops if disabled."""

    def __init__(self, log_dir: Optional[Path], enabled: bool = True, run_name: str = "run"):
        self.enabled = enabled and log_dir is not None
        self.writer = None
        if self.enabled:
            from torch.utils.tensorboard import SummaryWriter

            run_dir = Path(log_dir) / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(run_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
