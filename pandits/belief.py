# -*- coding: utf-8 -*-
"""
This module assumes that `history` is a list of (arm_id, reward) tuples.

To use one of the functions in this module on a subset of the history
use the squre bracket notation::

    rewards(hist[:t], arm_id=1)

"""
###############################################################################
import numpy as np


###############################################################################
def selected_arms(hist):
    """
    Return a list of the arms as they were selected
    (in chronological order).
    """
    return [id_ for id_, _ in hist]


def n_played(hist, arm_id):
    """
    Return how often the arm with the given `arm_id` was played.
    """
    return _calc_reward_stats(len, hist, arm_id)


def rewards(hist, arm_id=None):
    """
    Return a list of rewards (chronological ordered).

    If `arm_id` is specified only return the reward for it.
    """ + _calc_reward_stats.__doc__
    return _calc_reward_stats(lambda x: x, hist, arm_id)


def min_reward(hist, arm_id=None):
    """Return the min reward.""" + _calc_reward_stats.__doc__
    return _calc_reward_stats(np.min, hist, arm_id)


def max_reward(hist, arm_id=None):
    """Return the max reward.""" + _calc_reward_stats.__doc__
    return _calc_reward_stats(np.max, hist, arm_id)


def sum_reward(hist, arm_id=None):
    """Return the total reward.""" + _calc_reward_stats.__doc__
    return _calc_reward_stats(np.sum, hist, arm_id)


def mean_reward(hist, arm_id=None):
    """Return the mean reward.""" + _calc_reward_stats.__doc__
    return _calc_reward_stats(np.mean, hist, arm_id)


def var_reward(hist, arm_id=None):
    """Return the variance of the reward.""" + _calc_reward_stats.__doc__
    return _calc_reward_stats(np.var, hist, arm_id)


def std_reward(hist, arm_id=None):
    """Return the standard deviation of the reward.
    """ + _calc_reward_stats.__doc__
    return _calc_reward_stats(np.std, hist, arm_id)


###############################################################################
# HELPERS

# Execute `fn` with the given parameters.
# Simplifies adding statistics functions.
def _calc_reward_stats(fn, hist, arm_id):
    """

    Args:
        arm_id (int): if specified only use the bandit with the given `arm_id`.
    """
    if arm_id is None:
        return fn([reward for _, reward in hist])
    else:
        return fn([reward for id_, reward in hist if id_ == arm_id])
