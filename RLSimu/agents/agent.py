class BaseAgent:
    def __init__(self):
        pass

    def reset(self):
        pass

    # return initial action
    def begin_episode(self, initial_state, num_states, num_actions):
        return 0

    def end_episode(self):
        pass

    # do I need to specify previous_state and action? Or should I expect the Agent to keep track of that itself?
    def step(self, reward, state, time):
        return 0

    def step_eval(self, reward, state, time):
        pass
