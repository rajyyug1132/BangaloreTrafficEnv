"""Tests for graders.py — scores must stay strictly inside (0.0, 1.0)."""

import pytest

from graders import grade_rush_hour, grade_off_peak, grade_sustained_flow

ALL_GRADERS = [grade_rush_hour, grade_off_peak, grade_sustained_flow]


@pytest.mark.parametrize("grader", ALL_GRADERS)
def test_empty_episode_returns_floor(grader):
    assert grader([]) == 0.001


@pytest.mark.parametrize("grader", ALL_GRADERS)
def test_score_bounds_on_extreme_episodes(grader):
    perfect = [{"state": [0, 0, 0, 0, 0, i], "reward": 0.0} for i in range(100)]
    terrible = [{"state": [99, 99, 99, 99, 1, i], "reward": -396.0} for i in range(100)]
    for episode in (perfect, terrible):
        assert 0.001 <= grader(episode) <= 0.999


def test_reward_graders_rank_better_episodes_higher():
    good = [{"state": [1, 1, 1, 1], "reward": -4.0}] * 100
    bad = [{"state": [10, 10, 10, 10], "reward": -40.0}] * 100
    assert grade_rush_hour(good) > grade_rush_hour(bad)
    assert grade_off_peak(good) > grade_off_peak(bad)


def test_sustained_flow_full_score_at_or_below_threshold():
    under = [{"state": [4, 4, 4, 4], "reward": -16.0}] * 100   # avg 4 < 5
    over = [{"state": [10, 10, 10, 10], "reward": -40.0}] * 100  # avg 10 > 5
    assert grade_sustained_flow(under) == 0.999
    assert grade_sustained_flow(over) == pytest.approx(0.5)  # 5/10 smooth decay
