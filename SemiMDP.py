from collections import defaultdict
import numpy as np
from ConfidenceBounds import EmpBernsteinPeeling
from MDP import MDP
from tqdm import tqdm


class SemiMDP(MDP):
    def __init__(self, n_states, n_actions, options, r_max):
        super().__init__(n_states, n_actions)

        self.n_states = n_states
        self.n_actions = n_actions

        self.n_options = len(options)
        self.options = options
        self.transition_probabilities = None

        self.target = None
        self.r_max = r_max

        # Visitation counts of S x O.
        self.N = np.zeros((n_states, self.n_options))

        # Transition counts of S x O x S.
        self.P = np.zeros((n_states, self.n_options, n_states))

        # Accumulated rewards S x O.
        self.R = np.zeros((n_states, self.n_options))

        # Options lookup table.
        self.options_lut = defaultdict(list)

        for o in options:
            for i in o.initiation_states:
                self.options_lut[i].append(o.o_index)

    def get_next_states(self, state, option):
        next_states = []

        for s, p in enumerate(self.transition_probabilities[option][state]):
            if p > 0:
                next_states.append((s, p))

        return next_states

    def state_option_to_index(self, state, option):
        return np.ravel_multi_index((state, option), (self.n_states, self.n_options))

    def index_to_state_option(self, index):
        return np.unravel_index(index, (self.n_states, self.n_options))

    def has_experience_doubled(self, previous_experience):
        has_doubled = False

        for s in range(self.n_states):
            for o in range(self.n_options):
                if self.N[s][o] > 2 * previous_experience[s][o]:
                    has_doubled = True

        return has_doubled

    def estimate_transition_probability(self, alpha=0.1):
        p_hat = np.sum(self.P, axis=1)

        # for s in range(self.n_states):
        #     for o in range(self.n_options):
        #         for s_prime in range(self.n_states):
        #             p_hat[s][s_prime] += self.P[s][o][s_prime]

        for s in range(self.n_states):
            # Smooth estimate.
            row_sum = np.sum(p_hat[s])
            p_hat[s] = (p_hat[s] + alpha)/max(1, (row_sum + alpha * self.n_states))

        return p_hat

    def estimate_transition_bound(self, delta=0.05, t=1):
        lbs, ubs = [], []

        for o in range(self.n_options):
            p = np.zeros((self.n_states, self.n_states))
            n_plus = np.zeros((self.n_states))
            n = np.zeros((self.n_states))

            for s in range(self.n_states):
                for s_prime in range(self.n_states):
                    p[s][s_prime] += self.P[s][o][s_prime]
                    n[s] += self.P[s][o][s_prime]

            for s in range(self.n_states):
                p[s] /= max(1, np.sum(p[s]))
                n_plus[s] = max(1, n[s])


            # Using EmpBernsteinPeeling.
            CB = EmpBernsteinPeeling()
            lb, ub = CB.confidencebound(self.n_states,
                                        p,
                                        n_plus,
                                        n,
                                        delta,
                                        t)

            # assert lb.shape == p.shape, f"{lb.shape} =/= {p.shape}"
            # assert ub.shape == p.shape, f"{ub.shape} =/= {p.shape}"

            lbs.append(lb)
            ubs.append(ub)

        return np.array(lbs), np.array(ubs)

    def compute_true_transition_probability(self):
        P = np.zeros((self.n_options, self.n_states, self.n_states))

        for no, o in enumerate(self.options):
            for s in range(self.n_states):
                if s in o.initiation_states:
                    N = len(o.termination_condition)
                    states, termination_probs = zip(*o.termination_condition)
                    probs = []

                    for i in range(N):
                        p = termination_probs[i]
                        for j in range(i):
                            p *= (1 - termination_probs[j])

                        probs.append(p)

                    # assert np.isclose(sum(probs), 1.0), f"Sum of probabilities should be 1; but isn't!"

                    for i, s_prime in enumerate(states):
                        P[no][s][s_prime] = probs[i]

        return P

    def compute_true_reward(self):
        R = np.zeros((self.n_states, self.n_options))

        for s in range(self.n_states):
            if s == self.target:
                for no in self.get_options(s):
                    R[s][no] = self.r_max

        return R

    def solve(self, true_rewards):
        v_old = np.zeros(self.n_states)
        v_diff = np.arange(self.n_states)
        pi = np.zeros(self.n_states)

        iter = 0

        while iter < 1000 and max(v_diff) - min(v_diff) >= 10**-6:
            iter += 1

            v = np.zeros(self.n_states)

            for s in range(self.n_states):
                v_max_over_options = 0
                maximizing_option = None

                # Max over options.
                for o in self.get_options(s):
                    q, inner_states = self.options[o].trueP0(self)

                    v_tmp = 0
                    mu_hat = self.options[o].estimate_stationary_distribution(q)

                    # Handle terminal option.
                    if s == self.target:
                        b_o = np.ones(self.n_states)
                        b_o /= np.sum(b_o)
                    else:
                        b_o = self.compute_true_transition_probability()[o][s]

                    for s_prime_idx, s_prime in enumerate(inner_states):
                        v_tmp += true_rewards[s_prime][o] * mu_hat[s_prime_idx]

                    mu_hat_padded = np.zeros(self.n_states)

                    for s_idx, state in enumerate(inner_states):
                        mu_hat_padded[state] = mu_hat[s_idx]

                    v_tmp += mu_hat_padded[s] * (np.dot(b_o.T, v_old) - v_old[s]) + v_old[s]

                    if v_tmp >= v_max_over_options:
                        v_max_over_options = v_tmp
                        maximizing_option = o

                v[s] = v_max_over_options
                pi[s] = maximizing_option

            # print(f"{v = }, {self.target = }")

            v_diff = v - v_old
            v_old = v

        gain = v_diff[0]

        return pi, gain

    def act(self, option_id):
        raise NotImplementedError("Implement in inheriting class.")

    def get_options(self, state=None):
        state = self.state if state is None else state
        return self.options_lut[state]
