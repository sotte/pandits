#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pandits
----------------------------------

Tests for `pandits` module.
"""
##############################################################################
from scipy import stats

from pandits import strategies
from pandits.bandit import Bandit

import pytest


##############################################################################
@pytest.mark.parametrize("StrategyCls,params", [
    (strategies.RoundRobin, None),
    (strategies.Random, None),
    (strategies.EpsilonGreedy, {"epsilon": .1}),
    (strategies.EpsilonGreedy, {"epsilon": .5}),
    (strategies.MaxMean, None),
    (strategies.UCB1, None),
    (strategies.UCB1Tuned, None),
])
def test_run_all_strategies(StrategyCls, params):
    # met = [
    #     ("regret", metrics.regret_over_time),
    #     ("regret avg.", metrics.regret_avg_over_time),
    #     ("frac. of best arm", metrics.frac_of_best_arm_over_time),
    # ]
    bandit = Bandit([stats.norm(1., 1.), stats.norm(1.5, 1.)])

    if params is None:
        strategy = StrategyCls(bandit)
    else:
        strategy = StrategyCls(bandit, **params)

    N = 100
    for _ in range(N):
        strategy.next()
    assert len(strategy.hist) == N
