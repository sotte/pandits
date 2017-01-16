# -*- coding: utf-8 -*-
"""
Metrics ...

"""
from pandits import belief


##############################################################################
def regret(hist, y_best, t=None):
    """Return the regret `r`.

    .. math ::

        r = T * y^* - \sum_{t=1}^{T} y_t

    where

      - T is the total rounds played
      - y^* is the highest mean of the underlying reward distributions
      - y_t is the sampled reward at time t
    """
    T = len(hist)
    return T * y_best - belief.sum_reward(hist)


def regret_avg(hist, y_best):
    """Return the average regret.

    .. math ::

        r / T

    Ref:
        see `regret`

    """
    T = len(hist)
    r = regret(hist, y_best)

    return r / T


def frac_of_best_arm(hist, best_arm_id):
    """
    Return the fraction the best arm played.
    """
    n = len(hist)
    n_best = belief.n_played(hist, best_arm_id)

    return n_best / n
