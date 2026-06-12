"""Shared checkpoint/resume logic for all trainers.

Provides:
  - save_checkpoint / load_checkpoint with full state (model, optimizer, AMP scaler)
  - signal handler for graceful shutdown (SIGINT/SIGTERM)
  - auto-save of "latest.pt" after every epoch for crash recovery
  - resume() that restores full training state
"""

import os
import signal
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class CheckpointMixin:
    """Mixin providing checkpoint save/load/resume with crash recovery.

    Classes that mix this in must define:
      - self.cfg  (OSFTConfig with checkpoint_dir)
      - self.device
      - self.logger
      - self.generator
      - self.discriminator
      - self.g_optimizer
      - self.d_optimizer
      - self.scaler  (GradScaler)
      - self.global_step
    """

    _interrupted: bool = False
    _signal_registered: bool = False

    @classmethod
    def _register_signal_handlers(cls):
        """Register signal handlers for graceful shutdown.

        Only registers once per process (class-level flag).
        """
        if cls._signal_registered:
            return
        cls._signal_registered = True

        def _handler(signum, frame):
            name = signal.Signals(signum).name if hasattr(signal, "Signals") else f"SIG{signum}"
            print(f"\n[{name}] Interrupt received. Finishing current epoch then saving checkpoint...")
            CheckpointMixin._interrupted = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError, AttributeError):
                pass  # Windows or non-main thread

    def save_checkpoint(self, epoch: int, path: Optional[str] = None, best: bool = False) -> str:
        """Save a full training checkpoint.

        Args:
            epoch: Current epoch (1-indexed)
            path: Explicit path. If None, uses cfg.checkpoint_dir / f'{prefix}_epoch_{epoch}.pt'
            best: If True, also saves as 'best.pt' symlink/copy

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            prefix = getattr(self, '_ckpt_prefix', 'model')
            path = os.path.join(self.cfg.checkpoint_dir, f"{prefix}_epoch_{epoch:04d}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": self.global_step,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "g_optimizer_state_dict": self.g_optimizer.state_dict(),
            "d_optimizer_state_dict": self.d_optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.cfg,
        }

        # Save any extra state provided by subclass
        extra = getattr(self, '_extra_checkpoint_state', lambda: {})()
        state.update(extra)

        torch.save(state, path)

        # Also save as "latest.pt" for auto-resume
        latest_path = os.path.join(self.cfg.checkpoint_dir,
                                   f"{getattr(self, '_ckpt_prefix', 'model')}_latest.pt")
        torch.save(state, latest_path)

        if best:
            best_path = os.path.join(self.cfg.checkpoint_dir,
                                     f"{getattr(self, '_ckpt_prefix', 'model')}_best.pt")
            torch.save(state, best_path)

        self.logger.info(f"Checkpoint saved → {os.path.basename(path)}")
        return path

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint and restore full training state.

        Returns:
            Dict with 'epoch', 'global_step', 'best_val' from the checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)
        self.logger.info(f"Loading checkpoint from {path} (epoch {state.get('epoch', '?')})")

        # Model weights
        self.generator.load_state_dict(state["generator_state_dict"])
        self.discriminator.load_state_dict(state["discriminator_state_dict"])

        # Optimizer states
        self.g_optimizer.load_state_dict(state["g_optimizer_state_dict"])
        self.d_optimizer.load_state_dict(state["d_optimizer_state_dict"])

        # AMP scaler
        if "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.global_step = state.get("global_step", 0)

        return {
            "epoch": state.get("epoch", 0),
            "global_step": self.global_step,
            "best_val": state.get("best_val", float("inf")),
        }

    def resume(self, path: Optional[str] = None) -> int:
        """Resume training from a checkpoint.

        If path is None, automatically finds the latest checkpoint:
          1. cfg.checkpoint_dir / '{prefix}_latest.pt'
          2. The most recent '{prefix}_epoch_*.pt' file

        Returns:
            Epoch number to resume from (0-indexed: 0 = start from epoch 0)
        """
        prefix = getattr(self, '_ckpt_prefix', 'model')

        if path is None:
            # Auto-find latest
            candidates = []
            if os.path.isdir(self.cfg.checkpoint_dir):
                for fname in os.listdir(self.cfg.checkpoint_dir):
                    if fname.startswith(prefix) and fname.endswith(".pt"):
                        candidates.append(fname)
            candidates.sort()  # epoch_0001 < epoch_0002 < ... < latest < best

            # Prefer 'latest' if it exists
            latest_name = f"{prefix}_latest.pt"
            if latest_name in candidates:
                path = os.path.join(self.cfg.checkpoint_dir, latest_name)
            elif candidates:
                # Filter out 'best' and 'latest', take highest epoch
                epoch_files = [c for c in candidates if c not in (latest_name, f"{prefix}_best.pt")]
                if epoch_files:
                    path = os.path.join(self.cfg.checkpoint_dir, epoch_files[-1])
                else:
                    path = os.path.join(self.cfg.checkpoint_dir, candidates[-1])
            else:
                self.logger.info("No checkpoint found. Starting from scratch.")
                return 0

        info = self.load_checkpoint(path)
        return info["epoch"]

    def _should_stop(self) -> bool:
        """Check if training should stop (interrupt signal received)."""
        return CheckpointMixin._interrupted

    def _clear_interrupt(self):
        """Clear the interrupt flag (at start of train)."""
        CheckpointMixin._interrupted = False
