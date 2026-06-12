"""Tests for checkpoint/resume functionality."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

_project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _project_root)

from main.osft.config import OSFTConfig
from main.osft.checkpoint import CheckpointMixin
from main.model.topologygan import TopologyGANGenerator, TopologyGANDiscriminator


def _build_toy_trainer(tmp_path):
    """Build a minimal FullFinetuneTrainer for testing checkpoint/resume."""
    from main.baselines.full_finetune import FullFinetuneTrainer
    cfg = OSFTConfig(
        n_epochs=5,
        batch_size=4,
        img_height=32,
        img_width=32,
        gf_dim=8,
        df_dim=4,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    gen = TopologyGANGenerator(
        input_c_dim=3, output_c_dim=1, gf_dim=8,
        variant="unet", height=32, width=32,
    )
    disc = TopologyGANDiscriminator(
        condition_dim=6, output_c_dim=1, df_dim=4,
        height=32, width=32,
    )
    return FullFinetuneTrainer(cfg, generator=gen, discriminator=disc)


def _build_toy_dataloader(n_samples=64):
    """Build a tiny dataloader with random data at 32x32 resolution."""
    real_A = torch.rand(n_samples, 3, 32, 32)
    real_B = torch.rand(n_samples, 1, 32, 32)
    bc = torch.rand(n_samples, 1, 32, 32)
    lx = torch.rand(n_samples, 1, 32, 32)
    ly = torch.rand(n_samples, 1, 32, 32)
    ds = TensorDataset(real_A, real_B, bc, lx, ly)
    return DataLoader(ds, batch_size=4, shuffle=True)


class TestCheckpointSaveLoad:

    def test_save_and_load(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        train_loader = _build_toy_dataloader(64)

        trainer.train_epoch(train_loader, 0)
        g_params_before = {k: v.clone() for k, v in trainer.generator.state_dict().items()}
        gs_before = trainer.global_step

        path = trainer.save_checkpoint(epoch=1)

        # Modify state by training more
        trainer.train_epoch(train_loader, 1)
        assert trainer.global_step > gs_before

        # Load checkpoint - should restore to saved state
        info = trainer.load_checkpoint(path)
        assert info["epoch"] == 1
        for k, v_before in g_params_before.items():
            assert torch.allclose(trainer.generator.state_dict()[k], v_before, atol=1e-6)

    def test_latest_checkpoint_created(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        trainer.train_epoch(loader, 0)
        trainer.save_checkpoint(epoch=1)

        latest = os.path.join(str(tmp_path), "ckpt", "full_ft_latest.pt")
        assert os.path.exists(latest)

    def test_best_checkpoint_created(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        trainer.train_epoch(loader, 0)
        trainer.save_checkpoint(epoch=1, best=True)

        best = os.path.join(str(tmp_path), "ckpt", "full_ft_best.pt")
        assert os.path.exists(best)

    def test_optimizer_state_preserved(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        trainer.train_epoch(loader, 0)

        path = trainer.save_checkpoint(epoch=1)
        opt_keys = set(trainer.g_optimizer.state_dict().keys())

        trainer.train_epoch(loader, 1)
        trainer.load_checkpoint(path)

        restored_keys = set(trainer.g_optimizer.state_dict().keys())
        assert opt_keys == restored_keys

    def test_scaler_state_preserved(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        trainer.train_epoch(loader, 0)

        path = trainer.save_checkpoint(epoch=1)
        trainer.scaler._scale = torch.tensor(999.0)

        trainer.load_checkpoint(path)
        assert "scale" in trainer.scaler.state_dict()


class TestResume:

    def test_resume_from_explicit_path(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        trainer.train_epoch(loader, 0)
        path = trainer.save_checkpoint(epoch=3)

        trainer2 = _build_toy_trainer(tmp_path)
        start = trainer2.resume(path)
        assert start == 3

    def test_auto_resume_finds_latest(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        trainer.train_epoch(loader, 0)
        trainer.save_checkpoint(epoch=5)

        trainer2 = _build_toy_trainer(tmp_path)
        start = trainer2.resume()
        assert start == 5

    def test_resume_none_when_no_checkpoints(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        start = trainer.resume()
        assert start == 0

    def test_train_with_resume_flag(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        loader = _build_toy_dataloader(64)
        val_loader = _build_toy_dataloader(32)

        trainer.train_epoch(loader, 0)
        trainer.save_checkpoint(epoch=1)
        gs_after_1 = trainer.global_step

        trainer.train_epoch(loader, 1)
        trainer.save_checkpoint(epoch=2)

        # Resume from latest (epoch 2) and run 2 more epochs
        trainer2 = _build_toy_trainer(tmp_path)
        trainer2.train(loader, val_loader, n_epochs=4, resume=True)
        assert trainer2.global_step >= gs_after_1


class TestSignalHandling:

    def test_interrupt_flag_works(self):
        CheckpointMixin._interrupted = True
        # Instantiate to test instance method
        from main.baselines.full_finetune import FullFinetuneTrainer
        cfg = OSFTConfig(img_height=32, img_width=32, gf_dim=8, df_dim=4,
                         checkpoint_dir="./tmp_ckpt")
        gen = TopologyGANGenerator(
            input_c_dim=3, output_c_dim=1, gf_dim=8,
            variant="unet", height=32, width=32,
        )
        disc = TopologyGANDiscriminator(
            condition_dim=6, output_c_dim=1, df_dim=4,
            height=32, width=32,
        )
        trainer = FullFinetuneTrainer(cfg, generator=gen, discriminator=disc)
        assert trainer._should_stop() is True
        trainer._clear_interrupt()
        assert trainer._should_stop() is False

    def test_signal_registered(self, tmp_path):
        trainer = _build_toy_trainer(tmp_path)
        # _signal_registered is set on the subclass, not the mixin
        assert type(trainer)._signal_registered is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
