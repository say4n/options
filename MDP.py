"""
MDP base class.
"""
class MDP:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        self.__state = None

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        # print(f"Setting state to {state}")
        assert 0 <= state < self.n_states, f"0 </= {state} </ {self.n_states}"

        if 0 <= state < self.n_states:
            self.__state = state
            # print(f"Set state to {self.__state}")
