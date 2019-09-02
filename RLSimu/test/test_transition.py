import transition_generators as tg
import numpy as np

a = tg.iterative_transitions_w_terminals(n_states=10, n_actions=2, n_terminal_states=2,
                                         recurrent_prob=lambda state, action: (state + 1) / 30,
                                         terminal_prob=lambda state, action: (state + 1) / 30,
                                         n_dest_per_state=4,
                                         n_dest_per_action=3,
                                         n_recurrent_dest_per_state=2,
                                         n_recurrent_dest_per_action=2,
                                         n_terminal_dest_per_state=2,
                                         n_terminal_dest_per_action=1)

b = tg.iterative_transitions_w_terminals(n_states=10, n_actions=2, n_terminal_states=2,
                                         recurrent_prob=lambda state, action: (state + 1) / 30,
                                         terminal_prob=lambda state, action: (state + 1) / 30,
                                         n_dest_per_state=4,
                                         n_dest_per_action=3,
                                         n_recurrent_dest_per_state=2,
                                         n_recurrent_dest_per_action=2,
                                         n_terminal_dest_per_state=2,
                                         n_terminal_dest_per_action=1)

tg.write_to_dot(a, path='a.dot', cutoff=0.01, terminal_states=[3,4])
# tg.write_to_dot(b[0], path='b.dot', cutoff=0.01, terminal_states=[3,4])
print(a[0])
print(a[1])
print(b[0])
print(b[1])

print(tg.get_reachable_states(a, 0))
print(tg.get_reachable_states(b, 0))