import numpy as np


# Define the Agent
class Agent:
    def __init__(self, env, num_actions, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.env = env
        # num_actions: indicated the actions that the agent can take
        # 4: up, down, left, right
        # 8: includes diagonal moves
        # 9: includes no movement action
        self.num_actions = num_actions
        self.Q = np.zeros((env.rows, env.cols, self.num_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state, greedy=False):
        if np.random.uniform() >= self.epsilon or greedy:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(self.num_actions)

    def update_Q(self, state, action, reward, next_state,):
        next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta
