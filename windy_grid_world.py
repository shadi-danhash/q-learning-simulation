import numpy as np
import matplotlib.pyplot as plt


# Define the Windy Gridworld environment
class WindyGridworld:
    def __init__(self, start, goal, rows, cols, wind=None, traps=None,):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.traps = traps or []
        self.wind = wind or [0] * cols

    def step(self, state, action):
        x, y = state
        dx, dy = self._get_action_direction(action)

        # Update position with wind effect
        x += self.wind[y]
        x += dx
        y += dy

        # Apply boundaries
        x = max(0, min(x, self.rows - 1))
        y = max(0, min(y, self.cols - 1))

        # Determine the reward and next state
        if (x, y) == self.goal:
            reward = 0
            done = True
        elif (x, y) in self.traps:
            reward = -5
            done = False
        else:
            reward = -1
            done = False

        next_state = (x, y)
        return next_state, reward, done

    @staticmethod
    def _get_action_direction(action):
        # The first four actions are for (up, down, right, left) directions
        # The second four directions are for the diagonal directions
        # The last actions is for no movement
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1), (-1, -1), (1, -1), (1, 1), (-1, 1), (0, 0)]
        return directions[action]

    def render(self, ax, agent_position):
        ax.clear()
        grid = np.zeros((self.rows, self.cols))
        if type(agent_position) == list:
            for p in agent_position:
                grid[p] = 5
        else:
            grid[agent_position] = 5

        # Mark the goal position
        grid[self.goal] = 10

        for trap in self.traps:
            grid[trap] = 3

        # Plot the grid
        ax.imshow(grid, cmap='Blues', origin='lower', vmin=0, vmax=10)

        # Add gridlines
        ax.set_xticks(np.arange(self.cols + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.rows + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linewidth=1)

        # Remove tick labels
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        plt.draw()
