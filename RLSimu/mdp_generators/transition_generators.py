import numpy as np
import numbers


# i.i.d.
def dirichlet_transitions(alpha=1.0, n_states=1, n_actions=1):
    if isinstance(alpha, numbers.Real):
        alpha = np.array([alpha] * n_states)
    assert(len(alpha) == n_states)

    transition_matrices = [np.random.dirichlet(alpha, n_states) for _ in range(n_actions)]

    return transition_matrices


# TODO: allow n_dest_states to be a function rather than a constant
# n_dest_per_state is the number of states reachable in one step by each state
# n_dest_per_action is the number of states reachable in one step by each state-action pair
def simple_transitions(n_states=3, n_dest_per_action=None, n_dest_per_state=None, alpha=1.0, n_actions=1, terminal_states=None):
    if terminal_states is None:
        if n_states > 2:
            terminal_states = [n_states - 1, n_states - 2]

    if isinstance(n_dest_per_action, numbers.Real):
        n_dest_per_action = np.array([n_dest_per_action] * n_states)

    if isinstance(n_dest_per_state, numbers.Real):
        n_dest_per_state = np.array([n_dest_per_state] * n_states)

    if isinstance(alpha, numbers.Real):
        alpha_arr = np.array([alpha] * n_states)
    else:
        alpha_arr = alpha

    transition_matrices = [np.zeros((n_states, n_states)) for _ in range(n_actions)]
    non_terminal_states = set(range(n_states)) - set(terminal_states)
    for s in non_terminal_states:
        reachable_states = np.random.choice(n_states, size=n_dest_per_state[s], replace=False)
        for a in range(n_actions):
            dest_states = np.random.choice(reachable_states, size=n_dest_per_action[s], replace=False)
            alph_array = np.array([alpha_arr[s]] * n_dest_per_action[s])
            dest_probs = np.random.dirichlet(alph_array, 1)[0]
            all_probs = np.zeros(n_states)
            for i in range(n_dest_per_action[s]):
                all_probs[dest_states[i]] = dest_probs[i]
            transition_matrices[a][s] = all_probs

    return transition_matrices


def get_reachable_states(transition_matrices, origin_state=0):
    n_actions = len(transition_matrices)
    n_states = len(transition_matrices[0])
    all_reachable_states = {origin_state}
    adjacent_states = dict()
    for state in range(n_states):
        adjacent = set()
        for action in range(n_actions):
            transition_probs = transition_matrices[action][state]
            adjacent.update(np.nonzero(transition_probs)[0])
        adjacent_states[state] = adjacent

    while True:
        new_reachable_states = all_reachable_states.copy()
        for state in all_reachable_states:
            new_reachable_states.update(adjacent_states[state])

        if new_reachable_states == all_reachable_states:
            return all_reachable_states
        else:
            all_reachable_states = new_reachable_states


# n_dest_per_state, n_recurrent_dest_per_state, etc. are real numbers or functions which take the state number as input
# terminal_prob, recurrent_prob, n_dest_per_action, n_recurrent_dest_per_action, etc are real numbers or functions which
# take the state AND action number as inputs
# the terminal states are the last n_terminal_states states (i.e. the highest-numbered ones)
def iterative_transitions_w_terminals(
        n_states,
        n_actions,
        n_terminal_states,
        recurrent_prob,
        terminal_prob,
        n_dest_per_state=None,
        n_dest_per_action=None,
        n_recurrent_dest_per_state=None,
        n_recurrent_dest_per_action=None,
        n_terminal_dest_per_state=None,
        n_terminal_dest_per_action=None,
        alpha=1.0,
):
    n_dest_per_state, n_recurrent_dest_per_state, n_terminal_dest_per_state, n_dest_per_action, \
        n_recurrent_dest_per_action, n_terminal_dest_per_action, recurrent_prob, terminal_prob = \
        apply_defaults_and_lambdas(n_states,
                                   n_dest_per_state,
                                   n_recurrent_dest_per_state,
                                   n_terminal_dest_per_state,
                                   n_dest_per_action,
                                   n_recurrent_dest_per_action,
                                   n_terminal_dest_per_action,
                                   recurrent_prob,
                                   terminal_prob)

    validate_n_dests(n_states, n_actions, n_dest_per_state, n_recurrent_dest_per_state, n_terminal_dest_per_state,
                     n_dest_per_action, n_recurrent_dest_per_action, n_terminal_dest_per_action)

    if isinstance(alpha, numbers.Real):
        alpha_per_state = np.array([alpha] * n_states)
    else:
        alpha_per_state = alpha

    terminal_states = set(range(n_states - n_terminal_states, n_states))

    transition_matrices = [np.zeros((n_states, n_states)) for _ in range(n_actions)]
    for s in range(n_states - n_terminal_states):
        n_remaining_states = n_states - n_terminal_states - s - 1

        n_state_dests = n_dest_per_state(s)
        # TODO: double check that this math is reasonable

        n_state_recurrent_dests = n_recurrent_dest_per_state(s)
        if n_state_recurrent_dests > s:
            print('WARNING: for state {0}, n_state_recurrent_dests ({1}) is greater than the number '
                  'of previous states, so it will be capped at {0}'.format(s, n_state_recurrent_dests))
            n_state_recurrent_dests = s

        n_state_terminal_dests = n_terminal_dest_per_state(s)

        if n_state_terminal_dests > n_terminal_states:
            print('WARNING: for state {0}, n_state_terminal_dests ({1}) is greater than the number '
                  'of terminal states, so it will be capped at {0}'.format(n_terminal_states, n_state_terminal_dests))
            n_state_terminal_dests = n_terminal_states
        n_state_remaining_dests = n_state_dests - n_state_recurrent_dests - n_state_terminal_dests
        if n_state_remaining_dests > n_remaining_states:
            print('WARNING: for state {0}, n_state_terminal_dests ({1}) is greater than the number '
                  'of terminal states, so it will be capped at {0}'.format(n_remaining_states, n_state_terminal_dests))
            n_state_remaining_dests = n_remaining_states

        assert n_state_recurrent_dests >= 0; assert n_state_terminal_dests >= 0; assert n_state_remaining_dests >= 0

        recurrent_dests = np.random.choice(s, size=n_state_recurrent_dests, replace=False)
        terminal_dests = np.random.choice(list(terminal_states), size=n_state_terminal_dests, replace=False)
        remaining_dests = np.random.choice(range(s, n_states - n_terminal_states), size=n_state_remaining_dests, replace=False)
        for a in range(n_actions):
            action_recurrent_prob = recurrent_prob(s, a)
            action_terminal_prob = terminal_prob(s, a)
            action_remaining_prob = 1 - action_recurrent_prob - action_terminal_prob
            if s == n_states - n_terminal_states - 1:  # there are no remaining "remaining" states
                action_recurrent_prob /= 1 - action_remaining_prob
                action_terminal_prob /= 1 - action_remaining_prob
                action_remaining_prob = 0
            assert action_recurrent_prob >= 0
            assert action_terminal_prob >= 0
            assert action_remaining_prob >= 0
            assert np.isclose([action_recurrent_prob + action_terminal_prob + action_remaining_prob], [1])

            n_action_dests = n_dest_per_action(s, a)
            n_action_recurrent_dests = n_recurrent_dest_per_action(s, a)
            if n_action_recurrent_dests > n_state_recurrent_dests:
                print('WARNING: for state {2}, n_action_recurrent_dests ({1}) is greater than n_state_recurrent_dests'
                      ' ({0}), so it will be capped at {0}' .format(n_state_recurrent_dests, n_action_recurrent_dests, s))
                n_action_recurrent_dests = n_state_recurrent_dests

            n_action_terminal_dests = n_terminal_dest_per_action(s, a)
            if n_action_terminal_dests > n_state_terminal_dests:
                print('WARNING: for state {0}, n_state_terminal_dests ({1}) is greater than the number '
                      'of terminal states, so it will be capped at {0}'.format(n_state_terminal_dests, n_action_terminal_dests))
                n_action_terminal_dests = n_state_terminal_dests

            n_action_remaining_dests = n_action_dests - n_recurrent_dest_per_action(s, a) - n_terminal_dest_per_action(s, a)
            if n_action_remaining_dests > n_state_remaining_dests:
                print('WARNING: for state {0}, n_state_terminal_dests ({1}) is greater than the number '
                      'of terminal states, so it will be capped at {0}'.format(n_state_remaining_dests, n_action_remaining_dests))
                n_action_remaining_dests = n_state_remaining_dests

            action_recurrent_dests = np.random.choice(recurrent_dests, size=n_action_recurrent_dests, replace=False)
            action_terminal_dests = np.random.choice(terminal_dests, size=n_action_terminal_dests, replace=False)
            action_remaining_dests = np.random.choice(remaining_dests, size=n_action_remaining_dests, replace=False)

            recurrent_alph_array = np.array([alpha_per_state[s]] * n_action_recurrent_dests)
            recurrent_dest_probs = np.random.dirichlet(recurrent_alph_array, 1)[0]

            terminal_alph_array = np.array([alpha_per_state[s]] * n_action_terminal_dests)
            terminal_dest_probs = np.random.dirichlet(terminal_alph_array, 1)[0]

            remaining_alph_array = np.array([alpha_per_state[s]] * n_action_remaining_dests)
            remaining_dest_probs = np.random.dirichlet(remaining_alph_array, 1)[0]
            print('\n')
            print('recurrent_dest_probs: ', recurrent_dest_probs)
            print('terminal_dest_probs: ', terminal_dest_probs)
            print('remaining_dest_probs: ', remaining_dest_probs)
            all_probs = np.zeros(n_states)
            if n_action_recurrent_dests == 0:
                action_recurrent_prob = 0
            if n_action_terminal_dests == 0:
                action_terminal_prob = 0
            if n_action_remaining_dests == 0:
                action_remaining_prob = 0
            prob_sum = action_recurrent_prob + action_terminal_prob + action_remaining_prob
            assert prob_sum > 0
            for i in range(n_action_recurrent_dests):
                all_probs[action_recurrent_dests[i]] = recurrent_dest_probs[i] * action_recurrent_prob / prob_sum
            for i in range(n_action_terminal_dests):
                all_probs[action_terminal_dests[i]] = terminal_dest_probs[i] * action_terminal_prob / prob_sum
            for i in range(n_action_remaining_dests):
                all_probs[action_remaining_dests[i]] = remaining_dest_probs[i] * action_remaining_prob / prob_sum
            transition_matrices[a][s] = all_probs

    return transition_matrices

def apply_defaults_and_lambdas(n_states,
                               n_dest_per_state,
                               n_recurrent_dest_per_state,
                               n_terminal_dest_per_state,
                               n_dest_per_action,
                               n_recurrent_dest_per_action,
                               n_terminal_dest_per_action,
                               recurrent_prob,
                               terminal_prob):
    if n_dest_per_state is None:
        n_dest_per_state = n_states
    if n_dest_per_action is None:
        n_dest_per_action = n_dest_per_state

    if isinstance(n_dest_per_state, numbers.Real):
        tmp1 = n_dest_per_state
        n_dest_per_state = lambda state: tmp1
    if isinstance(n_recurrent_dest_per_state, numbers.Real):
        tmp2 = n_recurrent_dest_per_state
        n_recurrent_dest_per_state = lambda state: tmp2
    if isinstance(n_terminal_dest_per_state, numbers.Real):
        tmp3 = n_terminal_dest_per_state
        n_terminal_dest_per_state = lambda state: tmp3
    if isinstance(n_dest_per_action, numbers.Real):
        tmp4 = n_dest_per_action
        n_dest_per_action = lambda state, action: tmp4
    if isinstance(n_recurrent_dest_per_action, numbers.Real):
        tmp5 = n_recurrent_dest_per_action
        n_recurrent_dest_per_action = lambda state, action: tmp5
    if isinstance(n_terminal_dest_per_action, numbers.Real):
        tmp6 = n_terminal_dest_per_action
        n_terminal_dest_per_action = lambda state, action: tmp6

    if isinstance(recurrent_prob, numbers.Real):
        tmp7 = recurrent_prob
        recurrent_prob = lambda state, action: tmp7
    if isinstance(terminal_prob, numbers.Real):
        tmp8 = terminal_prob
        terminal_prob = lambda state, action: tmp8

    # TODO: add more of these if statements, for n_recurrent_dest_per_state etc.

    return n_dest_per_state, n_recurrent_dest_per_state, n_terminal_dest_per_state, n_dest_per_action,\
        n_recurrent_dest_per_action, n_terminal_dest_per_action, recurrent_prob, terminal_prob

# TODO: validate this
def validate_n_dests(n_states, n_actions, n_dest_per_state, n_recurrent_dest_per_state, n_terminal_dest_per_state,
                     n_dest_per_action, n_recurrent_dest_per_action, n_terminal_dest_per_action):
    pass

import networkx as nx
from networkx.drawing.nx_agraph import write_dot

COLORS = ['black', 'firebrick', 'blue', 'darkgreen', 'darkorange', 'goldenrod', 'aquamarine4', 'indigo', 'magenta',
          'olivedrab', 'pink', 'gray35', 'chocolate', 'burlywood', 'green', 'red', 'cadetblue1', 'palegreen', 'brown', 'bisque']
# def write_to_dot(transition_matrix, path='mc.dot', cutoff=0, terminal_states=None):
#     if terminal_states is None:
#         terminal_states = []
#     g = nx.DiGraph()
#     edge_labels = {}
#     n_states = len(transition_matrix[0])
#
#     for i in range(n_states):
#         for j in range(n_states):
#             rate = transition_matrix[i][j]
#             if rate > cutoff and i not in terminal_states:
#                 g.add_edge(i, j, weight=rate, label="{:.02f}".format(rate), color=)
#                 edge_labels[(i, j)] = "{:.02f}".format(rate)
#
#     write_dot(g, path)

def write_to_dot(transition_matrices, path='mc.dot', cutoff=0, terminal_states=None):
    if terminal_states is None:
        terminal_states = []
    g = nx.DiGraph()
    edge_labels = {}
    n_states = len(transition_matrices[0][0])

    for index, transition_matrix in enumerate(transition_matrices):
        if index >= len(COLORS):
            edge_color = "gray"
        else:
            edge_color = COLORS[index]
        for i in range(n_states):
            print('------\nstate: ', i, '\n------')
            for j in range(n_states):
                rate = transition_matrix[i][j]
                print('dest state ', j, ' has rate ', rate)
                if rate > cutoff and i not in terminal_states:
                    g.add_edge(i, j, weight=rate, label="{:.02f}".format(rate), color=edge_color)
                    edge_labels[(i, j)] = "{:.02f}".format(rate)

    write_dot(g, path)
