import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

plt.style.use('seaborn-whitegrid')
sns.set_context("paper")

figsize = (10, 6)

def plot_heatmap(square_matrix, title, show=False, save=False):
    if show or save:
        plt.figure(figsize=figsize)
        ax = sns.heatmap(square_matrix, linewidth=0.5)
        plt.title(title)
        plt.gca().invert_yaxis()

        if save:
            plt.savefig(f"{title}.png", dpi=500)

        if show:
            plt.show()

        plt.close()

def plot_accumulated_regret(data, title, show=False, save=False):
    if show or save:
        plt.figure(figsize=figsize)
        data = np.array(data)
        mean = np.mean(data, axis=0)

        ci = 1.96 * sem(data, axis=0)
        hi = mean + ci
        lo = mean - ci

        xdata = np.arange(data.shape[1])

        plt.plot(xdata, mean, '-', color='k', lw=0.75)
        plt.fill_between(xdata, hi, lo, color="k", alpha=0.1)

        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Accumulated Regret")

        if save:
            plt.savefig(f"{title}.png", dpi=500)

        if show:
            plt.show()

        plt.close()


def plot_accumulated_reward(data, title, show=False, save=False):
    if show or save:
        plt.figure(figsize=figsize)
        data = np.array(data)
        mean = np.mean(data, axis=0)

        ci = 1.96 * sem(data, axis=0)
        hi = mean + ci
        lo = mean - ci

        xdata = np.arange(data.shape[1])

        plt.plot(xdata, mean, '-', color='k', lw=0.75)
        plt.fill_between(xdata, hi, lo, color="k", alpha=0.1)

        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("Accumulated reward")

        if save:
            plt.savefig(f"{title}.png", dpi=500)

        if show:
            plt.show()

        plt.close()

def plot_option_visitation_counts(overall_visitation_counts, title, n_states, show=False, save=False):
    if show or save:
        plt.figure(figsize=figsize)
        visits_df = pd.DataFrame.from_dict(overall_visitation_counts)

        sns.histplot(data=visits_df, x="state", hue="option", multiple="dodge", binwidth=0.5)

        plt.title(title)
        plt.xlabel("States")
        plt.xticks(range(n_states))
        plt.ylabel("Visitation Counts")

        if save:
            plt.savefig(f"{title}.png", dpi=500)

        if show:
            plt.show()

        plt.close()

def plot_policy(policy, env, title, show=False, save=False):
    if show or save:
        plt.figure(figsize=figsize)
        delta = 0.1
        policy_list = []
        for p in policy:
            policy_list.append(env.options[int(p)].name)

        for ns, p in enumerate(policy_list):
            y, x = np.unravel_index(ns, (env.d, env.d))

            if ns == env.target:
                plt.text(x + delta, y + 2 * delta, f"{ns}", weight="bold", color="g")
            else:
                plt.text(x + delta, y + 2 * delta, f"{ns}", weight="bold")

            if p == "U":
                text = "$\\Downarrow$"
            elif p == "D":
                text = "$\\Uparrow$"
            elif p == "L":
                text = "$\\Leftarrow$"
            elif p == "R":
                text = "$\\Rightarrow$"
            elif p == "X":
                text = "$\\bigstar$"
            else:
                raise ValueError("Invalid policy for grid world domain.")

            plt.text(x + 0.25, y + 0.75, text, fontsize=50)

        plt.xlim((0, env.d))
        plt.ylim((0, env.d))

        plt.xticks(range(env.d))
        plt.yticks(range(env.d))

        frame = plt.gca()
        frame.axes.xaxis.set_ticklabels([])
        frame.axes.yaxis.set_ticklabels([])

        frame.invert_yaxis()

        plt.grid(True)

        plt.title(title)

        if save:
            plt.savefig(f"{title}.png", dpi=500)

        if show:
            plt.show()

        plt.close()