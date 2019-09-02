import numpy as np
from agent import BaseAgent

# TODO: fix q values of terminal state?
# uses backwards-trace TD
# off-policy Q learning
class QLearningAgent(BaseAgent):
    def __init__(self, mdp, discount_rate=0.1, learning_rate=0.1, lambda_val=0, init_q_vals=None, epsilon_function=None):
        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.n_steps_trained = 0
        self.lambda_val = lambda_val
        self.eligibility_traces = np.zeros((self.n_states, self.n_actions))

        if init_q_vals is not None:
            self.init_q_vals = np.array(init_q_vals, copy=True)  # create a deep copy
            self.q_vals = np.array(self.init_q_vals, copy=True)
        else:
            self.init_q_vals = np.zeros((self.n_states, self.n_actions))
            self.q_vals = np.array(self.init_q_vals, copy=True)

        if epsilon_function is not None:
            self.epsilon_function = epsilon_function
        else:
            self.epsilon_function = lambda x: min((100/x, 1))

        super().__init__()

    def reset(self):
        self.q_vals = np.array(self.init_q_vals, copy=True)
        self.eligibility_traces = np.zeros((self.n_states, self.n_actions))

    # return initial action
    def begin_episode(self, initial_state, num_states, num_actions):
        self.eligibility_traces = np.zeros((self.n_states, self.n_actions))
        self.n_steps_trained += 1
        epsilon = self.epsilon_function(self.n_steps_trained)
        action = epsilon_random(epsilon, self.q_vals[initial_state])

        self.last_action = action
        self.last_state = initial_state
        self.eligibility_traces[self.last_state][self.last_action] += 1
        return action

    def end_episode(self):
        pass

    def step(self, reward, state, time):
        self.n_steps_trained += 1

        epsilon = self.epsilon_function(self.n_steps_trained)
        action = epsilon_random(epsilon, self.q_vals[state])

        best_action = np.argmax(self.q_vals[state])
        td_target = reward + self.discount_rate * self.q_vals[state][best_action]
        td_signals = self.learning_rate * (td_target - self.q_vals)
        self.q_vals += self.eligibility_traces * td_signals

        self.last_action = action
        self.last_state = state
        self.eligibility_traces *= self.discount_rate * self.lambda_val
        self.eligibility_traces[self.last_state][self.last_action] += 1
        return action

    def step_eval(self, reward, state, time):
        return np.argmax(self.q_vals[state])


def epsilon_random(epsilon, state_q_vals):
    n_actions = len(state_q_vals)
    best_action = np.argmax(state_q_vals)
    rand_val = np.random.uniform()
    if rand_val > epsilon:
        return best_action
    else:
        return np.random.choice(n_actions)
