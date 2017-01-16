from collections import namedtuple
from math import sqrt, log
import random

from pandits import belief

###############################################################################
Observation = namedtuple("Observation", ["arm_id", "reward"])


###############################################################################
# Strategy Base Clase
class StrategyABC(object):
    """Every strategy must inherit from StrategyABC and override
    `_select_bandit()`.

    Each strategy is iterable. Calling `next()` selects the next bandis and
    update the belief.

    Override the `_select_bandit()` function.

    """
    def __init__(self, bandit):
        self.bandit = bandit
        self.hist = []
        self.step = 0
        self.name = "OVERRIDE ME"
        self.params = None

    def next(self):
        # select bandit, play bandit and update belief
        arm_id = self._select_bandit()
        reward = self.bandit.play(arm_id)

        obs = Observation(arm_id, reward=reward)
        self.hist.append(obs)

        self.step += 1
        return obs

    def _select_bandit(self):
        """`_select_bandit` contains the logic for selecting the bandit.

        Overwrite this function and return the index of the bandit you want to
        play.
        """
        raise NotImplementedError()


###############################################################################
# Strategies
class RoundRobin(StrategyABC):
    """
    This baseline strategy selects one bandit after the other in
    a sequential round robin fashion.

    """
    def __init__(self, bandit):
        super(RoundRobin, self).__init__(bandit)
        self.name = "Round Robin"

    def _select_bandit(self):
        arm_id = self.step % self.bandit.n
        return arm_id


class Random(StrategyABC):
    """
    This baseline strategy randomly selects one of the bandits.

    """
    def __init__(self, bandit):
        super(Random, self).__init__(bandit)
        self.name = "Random"

    def _select_bandit(self):
        return random.randint(0, self.bandit.n - 1)


class EpsilonGreedy(StrategyABC):
    """
    After playing each bandit once, play a random bandit with
    probability :math:`\epsilon` (explore),
    otherwise play the current best bandit (exploit).

    Think: exploration with p=epsilon, exploitation with p=1-epsilon.

    """
    def __init__(self, bandit, *, epsilon=.1):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.name = "Eps. Greedy"
        self.params = {"epsilon": epsilon}

    def _select_random_bandit(self):
        return random.randint(0, self.bandit.n - 1)

    def _select_bandit(self):
        if self.step < self.bandit.n:  # play each arm once
            arm_id = self.step

        elif random.random() < self.epsilon:  # explore
            arm_id = self._select_random_bandit()

        else:  # exploit
            mean_rewards = [belief.mean_reward(self.hist, arm_id)
                            for arm_id in range(self.bandit.n)]
            arm_id = mean_rewards.index(max(mean_rewards))

        return arm_id


class MaxMean(EpsilonGreedy):
    """
    A greedy strategy that plays the bandit with the max mean after having
    played each bandit once.

    This is a special version of the epsilon-greedy strategy where
    :math:`\epsilon = 0`.

    """
    def __init__(self, bandit):
        super(MaxMean, self).__init__(bandit, epsilon=0)
        self.name = "Greedy Max Mean"


class UCB1(StrategyABC):
    """
    Upper confidence bound strategy.

    .. math ::

            i^* = \arg \max_{i=1..k} \left(
                    \hat\mu_i + \sqrt{ \frac{2 \ln n}{n_i} }
                  \right)

    See:
        Auer et al, Finite-time Analysis of the Multiarmed Bandit Problem, 2002

    """
    def __init__(self, bandit):
        super(UCB1, self).__init__(bandit)
        self.name = "UCB1"

    def _select_bandit(self):
        if self.step < self.bandit.n:  # play each bandit once
            arm_id = self.step
        else:
            arm_id = self._select_bandit_ucb()
        return arm_id

    def _select_bandit_ucb(self):
        # play the machine that maximizes:
        #       avg_reward + srqt(2 * ln * n_total / n_played)

        ucb_rewards = []
        for i in range(self.bandit.n):
            # avg reward of machine i
            r_i = belief.mean_reward(self.hist, i)
            # how often machine i was played
            n_i = belief.n_played(self.hist, i)
            # the total number of rounds played
            n = len(self.hist)

            # The ucb criteria
            ucb_reward = r_i + sqrt((2 * log(n)) / n_i)

            ucb_rewards.append(ucb_reward)

        arm_id = ucb_rewards.index(max(ucb_rewards))

        return arm_id


class UCB1Tuned(UCB1):
    """
    Tuned UCB0

    .. math ::

            i^* = \arg \max_{i=1..k} \left(
                    \hat\mu_i +
                    \sqrt\left(
                      \frac{\ln t}{n_i} \min( \frac{1}{4}, V_i(n_i))
                    \right)
                  \right)

    where

    .. math ::

        V_i(t) = \hat\sigma_i^2(t) + \sqrt{\frac{2 \ln t}{n_i(t)}}

    See:
        Auer et al, Finite-time Analysis of the Multiarmed Bandit Problem, 2002

    """
    def __init__(self, bandit):
        super(UCB1Tuned, self).__init__(bandit)
        self.name = "UCB1-Tuned"

    def _select_bandit_ucb(self):
        ucb_rewards = []
        for i in range(self.bandit.n):
            n = len(self.hist)
            n_i = belief.n_played(self.hist, i)
            r_i = belief.mean_reward(self.hist, i)
            V_i = belief.var_reward(self.hist, i) + sqrt((2 * log(n)) / n_i)

            # The ucb tuned criteria
            ucb_reward = r_i + sqrt((log(n) / n_i) * min(.25, V_i))

            ucb_rewards.append(ucb_reward)

        arm_id = ucb_rewards.index(max(ucb_rewards))
        return arm_id
