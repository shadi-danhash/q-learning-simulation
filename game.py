from agent import Agent
from windy_grid_world import WindyGridworld
import matplotlib.pyplot as plt

class Runner:
    def __init__(self, num_actions, start, goal, rows, cols, episodes, wind=None, traps=None,
                 render_interval=None, gamma=1.0, alpha=0.5, epsilon=0.1, verbose=10):
        self.env = WindyGridworld(start=start, goal=goal, rows=rows, cols=cols, wind=wind, traps=traps)
        self.agent = Agent(env=self.env, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
        self.episodes = episodes
        self.render_interval = render_interval
        self.verbose = verbose
        self.ax1, self.ax2, self.text_box = None, None, None
        self.hist = {}

    def reset(self):
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.ax1.set_title("Gridworld")
        self.ax2.set_title("Episodic Time steps")
        self.ax2.set_xlabel("Time steps")
        self.ax2.set_ylabel("Episodes")
        self.text_box = self.ax2.text(0.05, 0.95, "", transform=self.ax2.transAxes, va="top")
        self.hist = {'episodes': [], 'times': []}

    def train(self):
        self.reset()
        t = 0

        for episode in range(self.episodes):
            state = self.env.start
            total_reward = 0
            done = False
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(state, action)
                self.agent.update_Q(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                self.hist['episodes'].append(episode)
                self.hist['times'].append(t)

                t += 1
                s = f"Episode: {episode}, Total Reward: {total_reward}"

                if self.render_interval and episode % self.render_interval[0] == self.render_interval[1]:
                    self.env.render(self.ax1, state)
                    self.ax2.plot(self.hist['times'], self.hist['episodes'], 'b')
                    self.text_box.set_text(s)
                    plt.pause(0.01)

            if episode % self.verbose == 0:
                s = f"Episode: {episode}, Total Reward: {total_reward}"
                print(s)

    def test(self, pause=3):
        print('-------------------')
        print('Evaluating agent...')
        print('Number of actions', self.agent.num_actions)
        state = self.env.start
        done = False
        total_reward = 0
        path = [self.env.start]
        while not done:
            action = self.agent.select_action(state, greedy=True)
            next_state, reward, done = self.env.step(state, action)
            state = next_state
            path.append(next_state)
            total_reward += reward

        self.env.render(self.ax1, path)
        s = f"Evaluation - Total Reward: {total_reward}"
        print(s)
        print('-------------------')

        self.text_box.set_text(s)
        plt.pause(pause)
        plt.close()