import numpy as np


# this module contains various dynamic programming methods to estimate state values in an MDP
# TODO: implement prioritized sweeping
def value_iteration(mdp, n_iterations=10000, discount_rate=0.9, mean_rewards=None):
    value_estimates = np.zeros(mdp.n_states)
    if mean_rewards is None:
        mean_rewards = get_mean_rewards(mdp.reward_functions)

    expected_rewards = np.zeros((mdp.n_states, mdp.n_actions))
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            transition_probs = mdp.transition_matrices[action][state]
            expected_rewards[state][action] = np.dot(transition_probs, mean_rewards)

    non_terminal_states = set(range(mdp.n_states)) - set(mdp.terminal_states)
    for i in range(n_iterations):
        for state in non_terminal_states:
                estimated_action_values = [expected_rewards[state][action] +
                                       discount_rate * np.dot(mdp.transition_matrices[action][state], value_estimates)
                                       for action in range(mdp.n_actions)]
                value_estimates[state] = np.max(estimated_action_values)

    return value_estimates


def get_mean_rewards(reward_functions):
    n = len(reward_functions)
    mean_rewards = np.zeros(n)
    for i in range(n):
        # average the reward function over 1000 time points
        mean_rewards[i] = np.mean([reward_functions[i](t) for t in range(1000)])
    return mean_rewards


def action_value_estimates(mdp, discount_rate, state_value_estimates, mean_rewards=None):
    if mean_rewards is None:
        mean_rewards = get_mean_rewards(mdp.reward_functions)

    expected_rewards = np.zeros((mdp.n_states, mdp.n_actions))
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            transition_probs = mdp.transition_matrices[action][state]
            expected_rewards[state][action] = np.dot(transition_probs, mean_rewards)

    q_estimates = np.zeros((mdp.n_states, mdp.n_actions))
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            next_state_val = np.dot(mdp.transition_matrices[action][state], state_value_estimates)
            q_estimates[state][action] = expected_rewards[state][action] + discount_rate * next_state_val
    
    return q_estimates
