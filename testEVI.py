#! /usr/bin/env python3

import numpy as np
from copy import deepcopy as dc
from tqdm.auto import trange, tqdm

from GridWorldOptions import GridWorldOptions
from ExtendedValueIteration import EVI
from util import plot_option_visitation_counts, plot_policy, plot_accumulated_reward, plot_accumulated_regret

MAX_ITERS = 200_000
NUM_EXP = 50
GRID_DIM = 5
T_MAX = 2

# Keeps track of the accumulated regret across experiments.
data = list([] for _ in range(NUM_EXP))

overall_visitation_counts = {"state": [], "option": []}
accumulated_regrets = []

for exp in trange(NUM_EXP):
    env = GridWorldOptions(GRID_DIM, r_max=1, teleport_on_reaching_goal=True, m=T_MAX)
    evi = EVI()


    pi = [0] * env.n_states
    state = env.state
    previous_experience = dc(env.N)

    true_reward = env.compute_true_reward()

    tqdm.write("Computing optimal policy...", end="")
    optimal_policy, asymptotic_average_reward = env.solve(true_reward)
    tqdm.write(f"Done! {asymptotic_average_reward = :.5f}")
    plot_policy(optimal_policy, env, f"Optimal policy in {GRID_DIM}x{GRID_DIM} grid with target {env.target}", show=False, save=True)

    accumulated_reward = 0
    accumulated_holding_time = 0
    accumulated_regret = []

    for t in trange(MAX_ITERS):
        valid_options = env.get_options(state)
        option = pi[state]

        if option not in valid_options:
            option = np.random.choice(valid_options)

        option = int(option)
        opt = env.options[option]

        prev_state = dc(env.state)
        state, reward, holding_time = env.act(option)

        overall_visitation_counts["state"].append(prev_state)
        overall_visitation_counts["option"].append(opt.name)

        accumulated_reward += reward
        accumulated_holding_time += holding_time

        accumulated_regret.append(accumulated_holding_time * asymptotic_average_reward - accumulated_reward)

        data[exp].append(accumulated_reward)

        if env.has_experience_doubled(previous_experience) or t == MAX_ITERS - 1:
            # It doesn't matter which option's method is called as the counts are used from the SemiMDP.
            _, r_ub = env.options[0].estimate_reward_bound(MDP=env)

            p_hat = env.estimate_transition_probability()
            p_lower, p_upper = env.estimate_transition_bound()

            pi = evi.computepolicy(env, r_ub, p_lower, p_hat, p_upper, t+1)
            previous_experience = dc(env.N)

    accumulated_regrets.append(accumulated_regret)
    plot_policy(pi, env, f"Learned policy in {GRID_DIM}x{GRID_DIM} grid with target {env.target}", show=False, save=True)


plot_accumulated_regret(accumulated_regrets, f"Accumulated regret in {GRID_DIM}x{GRID_DIM} grid ({NUM_EXP} experiments)", show=False, save=True)
plot_accumulated_reward(data, f"Accumulated reward in {GRID_DIM}x{GRID_DIM} grid ({NUM_EXP} experiments)", save=True)
plot_option_visitation_counts(overall_visitation_counts, f"Visitation counts (teleport) in {GRID_DIM}x{GRID_DIM} grid ({NUM_EXP} experiments)", GRID_DIM * GRID_DIM, save=True)
