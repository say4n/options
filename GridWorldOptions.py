import numpy as np
from copy import deepcopy as dc
from SemiMDP import SemiMDP
from Option import Option

class GridWorldOptions(SemiMDP):
    action_lut = {
        "U": 0,
        "D": 1,
        "L": 2,
        "R": 3,
        "X": 4
    }

    option_lut = action_lut

    def __init__(self, d, r_max=1.0, teleport_on_reaching_goal=True, m=2):
        self.d = d
        self.r = None
        self.c = None

        self.teleport_on_reaching_goal = teleport_on_reaching_goal

        self.n_states = d * d               # Total number of positions on the grid is d^2.
        self.n_actions = 4                  # 4 actions - U,D,L,R.
        self.n_options = 4 * self.n_states  # Each state has 4 options - U,D,L,R.

        self.m = m
        self.r_max = r_max

        self.option_counter = 0

        options = []

        self.has_terminated = False

        self.r = np.random.randint(self.d)
        self.c = np.random.randint(self.d)

        state = self.row_col_to_state(self.r, self.c)
        target = np.random.randint(self.n_states)

        for rr in range(self.d):
            for cc in range(self.d):
                if target == self.row_col_to_state(rr, cc):
                    # Target state has option to trap.
                    options.append(self.__get_terminal_state_option(rr, cc))
                else:
                    # All other states have other options.
                    options.append(self.__get_up_option(rr, cc))
                    options.append(self.__get_down_option(rr, cc))
                    options.append(self.__get_right_option(rr, cc))
                    options.append(self.__get_left_option(rr, cc))

        super().__init__(self.n_states, self.n_actions, options, r_max)

        self.transition_probabilities = self.compute_true_transition_probability()
        self.state = state
        self.target = target


    def __get_up_state(self, row, col):
        new_r, new_c = min(row + 1, self.d - 1), col
        return self.row_col_to_state(new_r, new_c)

    def __get_down_state(self, row, col):
        new_r, new_c = max(row - 1, 0), col
        return self.row_col_to_state(new_r, new_c)

    def __get_right_state(self, row, col):
        new_r, new_c = row, min(col + 1, self.d - 1)
        return self.row_col_to_state(new_r, new_c)

    def __get_left_state(self, row, col):
        new_r, new_c = row, max(col - 1, 0)
        return self.row_col_to_state(new_r, new_c)

    def __get_terminal_state_option(self, row, col):
        init_state = self.row_col_to_state(row, col)
        next_states = [(init_state, 1.0)]

        option = Option(n_states=self.n_states,
                        n_actions=1,
                        initiation_states=[init_state],
                        termination_condition=next_states,
                        policy=[self.option_counter] * self.n_states,
                        reward=self.r_max,
                        name="X",
                        o_index=self.option_counter)

        self.option_counter += 1

        return option

    def __get_up_option(self, row, col):
        next_states = [(x, col) for x in range(row + 1, min(self.d, row + self.m + 1))]
        # if len(next_states) == 0: # Hit top wall
        #     next_states = [(row, col)]
        next_states = [(row, col)] + next_states

        t_states = list(map(lambda tuple: self.row_col_to_state(tuple[0], tuple[1]),
                            next_states))

        # print(f"U({row, col} [{self.row_col_to_state(row, col)}]) -> {list(zip(next_states, t_states))}")

        t_probs = np.zeros(len(t_states))
        for k in range(len(t_probs)):
            t_probs[k] = 1/(len(t_probs) - k)

        # print(f"{t_probs = }")

        init_state = self.row_col_to_state(row, col)

        option = Option(n_states=self.n_states,
                        n_actions=1,
                        initiation_states=[init_state],
                        termination_condition=list(zip(t_states, t_probs)),
                        policy=[self.option_counter] * self.n_states,
                        reward=0,
                        name="U",
                        o_index=self.option_counter)

        self.option_counter += 1

        return option

    def __get_down_option(self, row, col):
        next_states = [(x, col) for x in range(max(0, row - self.m - 1), row)]
        # if len(next_states) == 0: # Hit bottom wall
        #     next_states = [(row, col)]
        next_states = [(row, col)] + next_states

        t_states = list(map(lambda tuple: self.row_col_to_state(tuple[0], tuple[1]),
                            next_states))

        # print(f"D({row, col} [{self.row_col_to_state(row, col)}]) -> {list(zip(next_states, t_states))}")

        t_probs = np.zeros(len(t_states))
        for k in range(len(t_probs)):
            t_probs[k] = 1/(len(t_probs) - k)

        # print(f"{t_probs = }")

        init_state = self.row_col_to_state(row, col)

        option = Option(n_states=self.n_states,
                        n_actions=1,
                        initiation_states=[init_state],
                        termination_condition=list(zip(t_states, t_probs)),
                        policy=[self.option_counter] * self.n_states,
                        reward=0,
                        name="D",
                        o_index=self.option_counter)

        self.option_counter += 1

        return option

    def __get_right_option(self, row, col):
        next_states = [(row, x) for x in range(col + 1, min(self.d, col + self.m + 1))]
        # if len(next_states) == 0: # Hit right wall
        #     next_states = [(row, col)]
        next_states = [(row, col)] + next_states

        t_states = list(map(lambda tuple: self.row_col_to_state(tuple[0], tuple[1]),
                            next_states))

        # print(f"R({row, col} [{self.row_col_to_state(row, col)}]) -> {list(zip(next_states, t_states))}")

        t_probs = np.zeros(len(t_states))
        for k in range(len(t_probs)):
            t_probs[k] = 1/(len(t_probs) - k)

        # print(f"{t_probs = }")

        init_state = self.row_col_to_state(row, col)

        option = Option(n_states=self.n_states,
                        n_actions=1,
                        initiation_states=[init_state],
                        termination_condition=list(zip(t_states, t_probs)),
                        policy=[self.option_counter] * self.n_states,
                        reward=0,
                        name="R",
                        o_index=self.option_counter)

        self.option_counter += 1

        return option

    def __get_left_option(self, row, col):
        next_states = [(row, x) for x in range(max(0, col - self.m - 1), col)]
        # if len(next_states) == 0: # Hit left wall
        #     next_states = [(row, col)]
        next_states = [(row, col)] + next_states

        t_states = list(map(lambda tuple: self.row_col_to_state(tuple[0], tuple[1]),
                            next_states))

        # print(f"L({row, col} [{self.row_col_to_state(row, col)}]) -> {list(zip(next_states, t_states))}")

        t_probs = np.zeros(len(t_states))
        for k in range(len(t_probs)):
            t_probs[k] = 1/(len(t_probs) - k)

        # print(f"{t_probs = }")

        init_state = self.row_col_to_state(row, col)

        option = Option(n_states=self.n_states,
                        n_actions=1,
                        initiation_states=[init_state],
                        termination_condition=list(zip(t_states, t_probs)),
                        policy=[self.option_counter] * self.n_states,
                        reward=0,
                        name="L",
                        o_index=self.option_counter)

        self.option_counter += 1

        return option

    def row_col_to_state(self, row, col):
        return np.ravel_multi_index((row, col), (self.d, self.d))

    def state_to_row_col(self, state):
        return np.unravel_index(state, (self.d, self.d))

    def get_state(self):
        return self.row_col_to_state(self.r, self.c)

    def act(self, option_id):
        option = self.options[option_id]
        t_state = dc(self.state)
        reward = 0

        if t_state == self.target:
            # print("In goal state.")
            reward = self.r_max

        if option.is_valid(t_state):
            # print(f"Acting with option: {option.name}")
            next_state, _, holding_time = option.act(self)
            self.state = next_state

            self.R[t_state][option_id] += reward
            self.P[t_state][option_id][next_state] += 1
            self.N[t_state][option_id] += 1

            self.r, self.c = self.state_to_row_col(next_state)
            self.state = self.row_col_to_state(self.r, self.c)

            return next_state, reward, holding_time
        else:
            # Sanity check. This should never happen.
            raise ValueError(f"Can't initiate {option} from state `{t_state}`")

    def teleport(self):
        self.r = np.random.randint(self.d)
        self.c = np.random.randint(self.d)
        self.state = self.row_col_to_state(self.r, self.c)

    def has_terminated(self):
        return self.has_terminated

    def __repr__(self):
        to_print = f"GridWorldOptions({self.n_states = }, {self.n_options = }, {self.d = }, {self.state = }, {self.target = })\n"

        return to_print

if __name__ == "__main__":
    env = GridWorldOptions(3)
    print(env)
