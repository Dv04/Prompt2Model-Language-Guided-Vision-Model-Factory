from __future__ import annotations

import pytest

from prompt2model.config import TrainingConfig
from prompt2model.training import learning_rate_at_step


def test_default_schedule_warms_up_then_reaches_minimum():
    config = TrainingConfig(
        epochs=3,
        learning_rate=1e-3,
        warmup_epochs=1,
        min_learning_rate_ratio=0.1,
    )
    values = [
        learning_rate_at_step(config, step=step, total_steps=12, steps_per_epoch=4)
        for step in range(12)
    ]
    assert values[0] == pytest.approx(0.25e-3)
    assert values[3] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(0.1e-3)
    assert all(a >= b for a, b in zip(values[3:], values[4:]))


def test_constant_schedule_is_explicit_and_constant():
    config = TrainingConfig(lr_schedule="constant", learning_rate=2e-4)
    values = [
        learning_rate_at_step(config, step=step, total_steps=10, steps_per_epoch=5)
        for step in range(10)
    ]
    assert all(value == pytest.approx(2e-4) for value in values)


def test_schedule_rejects_invalid_step_domain():
    config = TrainingConfig()
    with pytest.raises(ValueError, match="positive"):
        learning_rate_at_step(config, step=0, total_steps=0, steps_per_epoch=1)
    with pytest.raises(ValueError, match="non-negative"):
        learning_rate_at_step(config, step=-1, total_steps=1, steps_per_epoch=1)
