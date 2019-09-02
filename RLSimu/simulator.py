import numpy as np


class TerminalSimulator:
    def __init__(self, terminal_mdp, agent):
        self.mdp = terminal_mdp
        self.agent = agent

    # return list of cumulative reward for each episode
    def simulate(self, n_episodes, initial_state, no_learning=False):
        episode_rewards = [0] * n_episodes
        mdp = self.mdp
        for i in range(n_episodes):
            current_time = 0
            current_state = initial_state
            if no_learning:
                action = self.agent.step_eval(None, current_state, 0)
            else:
                action = self.agent.begin_episode(initial_state, mdp.n_states, mdp.n_actions)
            while current_state not in mdp.terminal_states:
                current_time += 1
                reward, current_state = mdp.step(current_state, action, current_time)
                if no_learning:
                    action = self.agent.step_eval(reward, current_state, current_time)
                else:
                    action = self.agent.step(reward, current_state, current_time)
                episode_rewards[i] += reward
            self.agent.end_episode()

        return episode_rewards

    def simulate_with_eval(self, n_episodes, initial_state, n_eval_episodes = 1000):
        episode_rewards = [0] * n_episodes
        evaluated_rewards = [0] * n_episodes
        for i in range(n_episodes):
            episode_rewards[i] = self.simulate(1, initial_state, no_learning=False)[0]
            simulated_rewards = self.simulate(n_eval_episodes, initial_state, no_learning=True)
            evaluated_rewards[i] = np.mean(simulated_rewards)
        return episode_rewards, evaluated_rewards