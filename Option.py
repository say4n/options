import numpy as np
from collections import defaultdict
from ConfidenceBounds import EmpBernsteinPeeling

class Option:
    def __init__(self, n_states, n_actions, initiation_states, termination_condition, policy, reward=0, *, name="", max_steps=float('inf'), o_index=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.initiation_states = initiation_states
        self.termination_condition = termination_condition
        self.reward = reward
        self.policy = policy

        self.__has_terminated = False
        self.__chosen_option = None

        self.o_index = o_index

        # Visitation counts of S x O inside the option.
        self.N_option = defaultdict(lambda: defaultdict(int))

        # Transition counts of S x O x S inside the option.
        self.P_option = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Option's copy of accumulated rewards S x O.
        self.R_option = defaultdict(int)

        # Epsilon for numerical comparison.
        self.eps = 10**-6
        self.name = name
        self.max_steps = max_steps

    def estimate_reward_bound(self, delta=0.05, t=1, MDP=None):
        """
        MDP: if MDP is none, options copy of counts for N and R are used.
        Otherwise, the MDP provides the counts.
        """
        reward = np.zeros((self.n_states))

        if MDP is None:
            for s in self.R_option:
                reward[s] = self.R_option[s]

            n_plus = np.zeros((self.n_states))
            n = np.zeros((self.n_states))

            for s in self.N_option:
                for a in self.N_option[s]:
                    n_plus[s] += self.N_option[s][a]
                    n[s] += self.N_option[s][a]

            for s in self.N_option:
                n_plus[s] = max(1, n_plus[s])

            # Using EmpBernsteinPeeling.
            CB = EmpBernsteinPeeling()
            lb, ub = CB.confidencebound(self.n_states,
                                        reward,
                                        n_plus,
                                        n,
                                        delta,
                                        t)
        else:
            for s in range(MDP.n_states):
                reward[s] = np.sum(MDP.R[s])

            # Normalize rewards.
            sum_of_rewards = sum(reward)
            sum_of_rewards = sum_of_rewards if sum_of_rewards != 0 else 1
            for s in range(MDP.n_states):
                reward[s] /= sum_of_rewards

            n_plus = np.zeros((self.n_states))
            n = np.zeros((self.n_states))

            for s in range(MDP.n_states):
                n[s] += np.sum(MDP.N[s])
                n_plus[s] = np.sum(MDP.N[s])

            for s in range(MDP.n_states):
                n_plus[s] = max(1, n_plus[s])

            # reward and n_plus work, update as expected.
            # print(f"{n_plus = }\n{n = }\n{reward = }")

            # Using EmpBernsteinPeeling.
            CB = EmpBernsteinPeeling()
            lb, ub = CB.confidencebound(self.n_states,
                                        reward,
                                        n_plus,
                                        n,
                                        delta,
                                        t)

        # assert reward.shape == lb.shape, f"{reward.shape = } =/= {lb.shape = }"
        # assert reward.shape == ub.shape, f"{reward.shape = } =/= {ub.shape = }"

        lb /= np.max(lb)
        ub /= np.max(ub)

        return lb, ub

    def estimate_stationary_distribution(self, transition_probability):
        # See https://stephens999.github.io/fiveMinuteStats/markov_chains_discrete_stationary_dist.html
        eiv, r_evec = np.linalg.eig(transition_probability)
        l_evec = np.linalg.pinv(r_evec)

        # Find index of eigenvector with eigenvalue of unity.
        pick_index = np.where(np.abs(eiv - 1) <= self.eps)
        pick_index = pick_index[0]

        t_estimate = np.zeros(transition_probability.shape[0])

        for idx in pick_index:
            t_estimate += np.abs(l_evec[idx])

        estimate = t_estimate / np.sum(t_estimate)

        return estimate

        # # See https://stephens999.github.io/fiveMinuteStats/stationary_distribution.html

        # n_rows, n_cols = transition_probability.shape
        # A = np.vstack([transition_probability.T - np.eye(n_rows), np.ones((n_cols))])
        # Z = np.vstack([np.zeros((n_rows, 1)), [1]])

        # # SVD decomposition
        # u, s, vh = np.linalg.svd(A, full_matrices=False)
        # sigma = np.eye(s.shape[0]) / s

        # estimate = (vh.T @ sigma @ u.T) @ Z

        # assert np.isclose(np.sum(estimate), 1), "Sum of stationary distribution =/= 1!"

        # return estimate

    def estimate_transition_probability(self, alpha=0):
        p_hat = np.zeros((self.n_states, self.n_states))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_prime in range(self.n_states):
                    p_hat[s][s_prime] += self.P_option[s][a][s_prime]

        for s in range(self.n_states):
            # Smooth estimate.
            p_hat[s] = (p_hat[s] + alpha)/(np.sum(p_hat[s]) + alpha * self.n_states)

        return p_hat

    def estimate_condition_number(self, p_hat_prime, mu_hat):
        k_hat_min = float('-inf')
        z_o_hat = np.linalg.inv(np.eye(p_hat_prime.shape()) - p_hat_prime + np.ones(mu_hat.shape()).T @ mu_hat)
        n_rows = z_o_hat.shape(0)

        for i in range(n_rows):
            for j in range(n_rows):
                k_hat_min = max(k_hat_min, 0.5 * np.linalg.norm(z_o_hat[i] - z_o_hat[j]))

        return k_hat_min

    def is_valid(self, state):
        """
        :return True if option is valid in current state.
        """
        if state in self.initiation_states:
            return True

        return False

    def get_visitation_counts(self):
        return self.N

    def trueP0(self, mdp):
        # terminal states
        absorbing = [(s, p) for (s, p) in self.termination_condition if p > 0.0]
        # inner: all states that are not terminal
        inner = [(s, p) for (s, p) in self.termination_condition if p < 1.0]
        # internal: internal states with initial state removed
        internal =  [(s, p) for (s, p) in inner if s not in self.initiation_states]

        # compute true matrix P0
        p0 = np.zeros((len(inner) + len(absorbing), len(inner) + len(absorbing)))
        q0 = np.zeros((len(inner), len(inner)))
        v0 = np.zeros((len(inner), len(absorbing)))

        for i, (s, _) in enumerate(inner):
            for j, (s_prime, p_s_prime) in enumerate(inner):
                q0[i][j] = (1 - p_s_prime) * mdp.transition_probabilities[self.o_index][s][s_prime]

        for i, (s, _) in enumerate(inner):
            for k, (s_prime, p_s_prime) in enumerate(absorbing):
                v0[i][j] = p_s_prime * mdp.transition_probabilities[self.o_index][s][s_prime]

        p0[:len(inner), :len(inner)] = q0
        p0[:len(inner), len(inner):] = v0
        p0[len(inner):, len(inner):] = np.eye(len(absorbing), len(absorbing))

        states = []
        states.extend(map(lambda el: int(el[0]), inner))
        states.extend(map(lambda el: int(el[0]), absorbing))

        return p0, states

    def act(self, mdp):
        holding_time = 1
        init_state = mdp.state

        self.__has_terminated = False

        if init_state == mdp.target:
            mdp.state = np.random.randint(self.n_states)
            self.__has_terminated == True
            return mdp.state, self.reward, holding_time

        state = mdp.state

        i = 0
        while i < len(self.termination_condition):
            holding_time += 1
            if np.random.random() < self.termination_condition[i][1]:
                mdp.state = self.termination_condition[i][0]
                next_state = mdp.state
                break
            else:
                i += 1

        self.N_option[state][self.o_index] += 1
        self.P_option[state][self.o_index][next_state] +=1

        self.R_option[init_state] += self.reward

        return mdp.state, self.reward, holding_time

    # def random_policy(self, mdp, state):
    #     action = np.random.randint(mdp.n_options)
    #     self.__chosen_option = action
    #     successors = []
    #     probs = []

    #     for s_prime in mdp.get_next_states(state, action):
    #         successors.append(s_prime[0])
    #         probs.append(s_prime[1])

    #     return np.random.choice(successors, p=probs)

    # def optimal_policy(self, mdp, state):
    #     successors, probs = zip(*self.termination_condition)

    #     # FIXME: Needs fixing to take into account the termination probabilities.
    #     # FIXME: NOT A SUM!
    #     # FIXME: holding time needs fixing too.
    #     probs = np.array(probs)
    #     probs /= sum(probs)

    #     choice = np.random.choice(successors, p=probs)

    #     return choice, successors.index(choice) + 1

    def has_terminated(self):
        return self.__has_terminated

    def __repr__(self):
        return f"Option({self.name = }, {self.initiation_states = }, {self.termination_condition = }, {self.reward = })"
