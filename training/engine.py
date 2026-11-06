from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is the new best."""
        improved = self.best is None or (
            value < self.best if self.mode == "min" else value > self.best
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def save_checkpoint(model: torch.nn.Module, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path, map_location=None) -> torch.nn.Module:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


def save_training_state(path, model: torch.nn.Module, optimizer, epoch: int, stopper: EarlyStopper,
                         history: dict | None = None) -> None:
    """Per-epoch checkpoint (model + optimizer + early-stop state + history) so a
    killed/hung run can resume from the last completed epoch instead of starting over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stopper_best": stopper.best,
        "stopper_counter": stopper.counter,
        "history": history,
    }, path)


def load_training_state(path, model: torch.nn.Module, optimizer, stopper: EarlyStopper, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    stopper.best = state["stopper_best"]
    stopper.counter = state["stopper_counter"]
    return state["epoch"], state.get("history")
