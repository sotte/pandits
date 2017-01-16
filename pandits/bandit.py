class Bandit:
    """
    Bandit is an abstraction of a multi-armed bandit.

    Examples:

        >>> import numpy as np
        >>> from scipy import stats
        >>> np.random.seed(42)  # make example repeatable
        >>> bandit = Bandit([stats.norm(1, 1), stats.norm(2, 1)])
        >>> bandit.means()
        [1.0, 2.0]
        >>> round(bandit.play(0), 1)
        1.5

    """
    def __init__(self, arms):
        self.arms = arms
        self.n = len(arms)

    def play(self, arm_id):
        """Play the arm with the given ``arm_id`` and return a reward."""
        return self.arms[arm_id].rvs()

    def means(self):
        """Return the means of the bandits."""
        return [arm.mean() for arm in self.arms]

    def best_arm(self):
        """Return the id of the best arm according to the mean."""
        means = self.means()
        return means.index(max(means))

    def best_reward(self):
        """Return the reward of the best arm."""
        return max(self.means())
