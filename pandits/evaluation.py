# -*- coding: utf-8 -*-
"""
Helpers to evaluate strategies.
"""
from pelper.pelper import pipe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandits import metrics
from pandits import belief
from pandits.bandit import Bandit


##############################################################################
def run_experiments(
        *,
        arms: "list of distributions define the arms",
        strategies,
        n_plays: "how often the multi-armed bandit is played",
        n_repeats: "how often the one strategy is repeated"):
    """
    Run the specified experiment and return a lits of the runs.

    """
    run_results = []

    for run_i in range(n_repeats):

        for StrategyCls, params in strategies:
            np.random.seed(run_i)  # same seed for every strat in run_i
            bandit = Bandit(arms)

            # Init strategy
            if params is None:
                strategy = StrategyCls(bandit)
            else:
                strategy = StrategyCls(bandit, **params)

            for _ in range(n_plays):
                strategy.next()
            assert len(strategy.hist) == n_plays

            run_results.append((run_i, strategy))

    return run_results


def collect_statistics(run_results):
    """
    Collect statistics from the experiments and return it as a DataFrame.

    """
    tidy_data = []
    for run_i, strat in run_results:
        for step in range(1, len(strat.hist)):
            # collect data that is used later
            best_reward = strat.bandit.best_reward()
            best_arm = strat.bandit.best_arm()
            full_name = ("{} | {}".format(strat.name, str(strat.params))
                         if strat.params
                         else strat.name)

            data_point = (
                strat.name,
                strat.params,
                full_name,
                run_i,
                step,
                metrics.regret(strat.hist[:step], best_reward),
                metrics.regret_avg(strat.hist[:step], best_reward),
                metrics.frac_of_best_arm(strat.hist[:step], best_arm),
                belief.sum_reward(strat.hist[:step]),
                belief.mean_reward(strat.hist[:step]),
            )
            tidy_data.append(data_point)

    return pd.DataFrame(
        data=tidy_data,
        columns=[
            "strategy",
            "params",
            "name",
            "run_i",
            "step",
            "regret",
            "regret_avg",
            "best_arm",
            "reward",
            "reward_avg",
        ],
    )


def plot_statistics(data, title, store=False):
    """
    Create nice plots from the given dataframe `data`.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = sns.tsplot(data=data, ax=axes[0, 0], legend=True,
                    value="regret",
                    unit="run_i", time="step", condition="name")
    ax.set_title("Regret over time")

    ax = sns.tsplot(data=data, ax=axes[0, 1], legend=False,
                    value="regret_avg",
                    unit="run_i", time="step", condition="name")
    ax.set_title("Average regret over time")

    ax = sns.tsplot(data=data, ax=axes[0, 2], legend=False,
                    value="best_arm",
                    unit="run_i", time="step", condition="name")
    ax.set_title("Fraction of best arm played")

    ax = sns.tsplot(data=data, ax=axes[1, 0], legend=False,
                    value="reward",
                    unit="run_i", time="step", condition="name")
    ax.set_title("Cumulative reward")

    ax = sns.tsplot(data=data, ax=axes[1, 1], legend=False,
                    value="reward_avg",
                    unit="run_i", time="step", condition="name")
    ax.set_title("Average reward")

    plt.suptitle(title)

    # Save
    if store:
        pipe(title.lower().replace(" ", "_"),
             "images/{}.png".format,
             plt.savefig)

    return (fig, axes), data
