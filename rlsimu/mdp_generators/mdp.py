import numpy as np

# assume finite state, action space
class MDP:
    def __init__(self, transition_matrices, reward_functions):
        # verify transition_matrices, reward_functions?
            # len(transition_matrices) = # of actions
            # transition_matrices[i] = transition matrix corresponding to action i
            # len(transition_matrices[0][0]) = len(reward_functions) = number of states

        self.n_states = len(reward_functions)
        self.n_actions = len(transition_matrices)
        self.transition_matrices = transition_matrices
        self.reward_functions = reward_functions  # for now, the reward functions are specific to a state
        # in the future, should add reward functions that change based on the action chosen

    def step(self, state, action, time):
        transition_probs = self.transition_matrices[action][state]
        next_state = np.random.choice(self.n_states, p=transition_probs)

        reward = self.reward_functions[next_state](time)

        return reward, next_state

    def get_reward(self, state, time):
        return self.reward_functions[state](time)


# might be unnecessary
class TerminalMDP(MDP):
    def __init__(self, transition_matrices, reward_functions, terminal_states):
        self.terminal_states = terminal_states

        super().__init__(transition_matrices, reward_functions)

    def step(self, state, action, time):
        if state in self.terminal_states:
            raise Exception()

        return super().step(state, action, time)
