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


@pytest.mark.parametrize("lower_is_better,invalid_score", [
    (True, 0), (True, 100), (False, 200), (False, 100),
])
def test_valid_result_replaces_invalid_first_result(lower_is_better, invalid_score):
    text = f"score:{invalid_score} loss:4.0\nscore:100 loss:3.28"
    assert metrics_utils.extract_best_line_metrics(
        text, metric_types={"score": int, "loss": float},
        selection_metric="score", lower_is_better=lower_is_better,
        metrics_at_most={"loss": 3.28},
    ) == {"score": 100, "loss": 3.28, "is_valid": True}


@pytest.mark.parametrize("text,expected", [
    ("n_steps:0 val_loss:4.0 train_time:0\n"
     "n_steps:10 val_loss:3.28 train_time:100\n"
     "n_steps:20 val_loss:3.2 train_time:200", (10, 100)),
    ("n_steps:20 val_loss:3.2 train_time:200\n"
     "n_steps:10 val_loss:3.28 train_time:100\n"
     "n_steps:0 val_loss:4.0 train_time:0", (10, 100)),
    ("n_steps:0 val_loss:4.0 train_time:bad\n"
     "n_steps:10 val_loss:3.28 train_time:100", (10, 100)),
    ("n_steps:0 val_loss:4.0\n"
     "n_steps:10 val_loss:3.28 train_time:100", (10, 100)),
    ("n_steps:10 val_loss:3.28 train_time:100\n"
     "n_steps:20 val_loss:3.2 train_time:100", (10, 100)),
    ("n_steps:0 val_loss:3.28 train_time:0", (0, 0)),
])
def test_speedrun_best_valid_result(text, expected):
    n_steps, train_time = expected
    assert metrics_utils.extract_best_line_metrics(
        text, metric_types={"n_steps": int, "val_loss": float, "train_time": int},
        selection_metric="train_time", lower_is_better=True,
        metrics_at_most={"val_loss": 3.28},
    ) == {"n_steps": n_steps, "val_loss": 3.28, "train_time": train_time, "is_valid": True}
