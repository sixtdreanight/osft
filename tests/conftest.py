"""Shared test fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
