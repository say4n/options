from SemiMDP import SemiMDP
import numpy as np

MAX_ITERS = 50
EPSILON = 0.6

class EVI:
    def computepolicy(self, MDP: SemiMDP, r_upper, p_lower, p_hat, p_upper, t):
        v_old = np.zeros(MDP.n_states)
        v_diff = np.arange(MDP.n_states)
        pi = np.zeros(MDP.n_states)
        iter = 0

        while iter < 1000 and max(v_diff) - min(v_diff) > 1/np.sqrt(t):
            iter += 1

            q = self.normalizeprobs(p_lower, p_hat, p_upper, v_old)
            v = np.zeros(MDP.n_states)

            for s in range(MDP.n_states):
                v_max_over_options = 0
                maximizing_option = None

                # Max over options.
                for o in MDP.get_options(s):
                    v_tmp = 0
                    mu_hat = MDP.options[o].estimate_stationary_distribution(q[o])

                    for s_prime in range(MDP.n_states):
                        v_tmp += r_upper[s_prime] * mu_hat[s_prime]

                    v_tmp += mu_hat[s] * (np.dot(q[o][s].T, v_old) - v_old[s]) + v_old[s]

                    if v_tmp >= v_max_over_options:
                        v_max_over_options = v_tmp
                        maximizing_option = o

                v[s] = v_max_over_options
                pi[s] = maximizing_option

            v_diff = v - v_old
            v_old = v

        # print(f"{v = }, {iter = }, {pi = }")

        return pi

    def normalizeprobs(self, p_lower, p_hat, p_upper, V):
        no, ns, _ = p_upper.shape
        p = np.copy(p_upper)

        for o in range(no):
            for s in range(ns):
                p[o][s] /= np.sum(p[o][s])

        return p
