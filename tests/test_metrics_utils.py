# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from scientist.utils import metrics_utils


METRIC_TYPES = {"acc": float, "loss": float, "epoch": int}

MOCK_LOGS = """
step: 1 acc: 0.95, loss: 0.12 epoch: 1 
step: 2 acc: 0.96, loss: 0.11 epoch: 2 
step: 3 acc: 0.98, loss: 0.09 epoch: 3 
step: 4 acc: 0.97, loss: 0.10 epoch: 4 
"""


def test_extract_single_line_metrics():
	text = "step: 1 acc: 0.95, loss: 0.12 epoch: 10 "

	metrics = metrics_utils.extract_single_line_metrics(text, METRIC_TYPES)

	assert metrics == {'acc': 0.95, 'loss': 0.12, 'epoch': 10}


def test_extract_single_line_metrics_bad_type():
	text = "step: 1 acc: 0.95, loss: 0.12 epoch: test "

	metrics = metrics_utils.extract_single_line_metrics(text, METRIC_TYPES)

	assert metrics == {}


def test_extract_best_line_metrics_higher_is_better():
	text = MOCK_LOGS

	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='acc',
	)

	assert metrics == {'acc': 0.98, 'loss': 0.09, 'epoch': 3, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True
	)

	assert metrics == {'acc': 0.98, 'loss': 0.09, 'epoch': 3, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_at_most():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
	)

	assert metrics == {'acc': 0.98, 'loss': 0.09, 'epoch': 3, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_at_least():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_least={'epoch': 4}
	)

	assert metrics == {'acc': 0.97, 'loss': 0.10, 'epoch': 4, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_at_most():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_most={'epoch': 2}
	)

	assert metrics == {'acc': 0.96, 'loss': 0.11, 'epoch': 2, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_mixed_thresholds():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_most={'epoch': 2},
		metrics_at_least={'loss': 0.12}
	)

	assert metrics == {'acc': 0.95, 'loss': 0.12, 'epoch': 1, 'is_valid': True}


def test_extract_best_line_metrics_lower_is_better_no_match():
	text = MOCK_LOGS
	metrics = metrics_utils.extract_best_line_metrics(
		text, 
		metric_types=METRIC_TYPES,
		selection_metric='loss',
		lower_is_better=True,
		metrics_at_most={'epoch': 2},
		metrics_at_least={'epoch': 3}
	)

	assert metrics == {'acc': 0.95, 'loss': 0.12, 'epoch': 1, 'is_valid': False}


SPEEDRUN_METRIC_TYPES = {"n_steps": int, "val_loss": float, "train_time": int}


@pytest.mark.parametrize("text,n_steps", [
    ("step:128/24576 val_loss:3.2000 train_time:1234ms", 128),
    ("step:128/24576 val_loss:3.2000 train_time:1234ms step_avg:9.64ms", 128),
    ("n_steps:7 step:128/24576 val_loss:3.2 train_time:1234ms", 7),
    ("step:128/24576 n_steps:7 val_loss:3.2 train_time:1234ms", 7),
])
def test_extract_native_speedrun_line(text, n_steps):
    assert metrics_utils.extract_single_line_metrics(
        text, SPEEDRUN_METRIC_TYPES
    ) == {"n_steps": n_steps, "val_loss": 3.2, "train_time": 1234}


def test_native_speedrun_selects_fastest_qualifying_checkpoint():
    assert metrics_utils.extract_best_line_metrics(
        "step:256/24576 val_loss:3.20 train_time:2345ms\n"
        "step:128/24576 val_loss:3.28 train_time:1234ms\n"
        "step:512/24576 val_loss:4.0 train_time:500ms",
        metric_types=SPEEDRUN_METRIC_TYPES,
        selection_metric="train_time",
        lower_is_better=True,
        metrics_at_most={"val_loss": 3.28},
    ) == {"n_steps": 128, "val_loss": 3.28, "train_time": 1234, "is_valid": True}


@pytest.mark.parametrize("fields", [
    "step:bad val_loss:3.2 train_time:1234ms",
    "step:128/bad val_loss:3.2 train_time:1234ms",
    "step:128/24576 val_loss:3.2 train_time:1.5ms",
    "step:128/24576 val_loss:3.2 train_time:1234s",
    "step:128/24576 val_loss:3.2 train_time:1234msjunk",
    "step:128/24576 train_time:1234ms",
    "n_steps:bad step:128/24576 val_loss:3.2 train_time:1234ms",
    "step:128/24576 val_loss:3.2 train_time:1234ms n_steps:",
    "step:128/24576 val_loss:3.2 train_time:1234ms n_steps:,",
])
def test_native_speedrun_malformed_or_incomplete_line(fields):
    assert metrics_utils.extract_single_line_metrics(fields, SPEEDRUN_METRIC_TYPES) == {}


def test_generic_step_and_string_time_are_unchanged():
    assert metrics_utils.extract_single_line_metrics(
        "step:128 train_time:1234ms", {"step": int, "train_time": str}
    ) == {"step": 128, "train_time": "1234ms"}
