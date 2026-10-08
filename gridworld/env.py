import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

GRID_SIZE = 100

EMPTY = 0
AGENT = 1
GOAL = 2
HAZARD = 3

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class GridWorldEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        self.state = np.zeros(
            GRID_SIZE * GRID_SIZE,
            dtype=np.float32
        )

        self.agent_position = 6
        self.goal_position = 34
        self.hazard_position = 21

    def _update_state(self):
        self.state.fill(EMPTY)

        self.state[self.agent_position] = AGENT
        self.state[self.goal_position] = GOAL
        self.state[self.hazard_position] = HAZARD

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        position = self.np_random.choice(
            GRID_SIZE * GRID_SIZE,
            size = 3,
            replace = False
        )
        self.agent_position = int(position[0])
        self.goal_position = int(position[1])
        self.hazard_position = int(position[2])

        self._update_state()

        return self._get_observation(), {}

    def step(self, action):

        row, col = divmod(
            self.agent_position,
            GRID_SIZE
        )

        if action == UP:
            row -= 1

        elif action == DOWN:
            row += 1

        elif action == LEFT:
            col -= 1

        elif action == RIGHT:
            col += 1

        row = max(0, min(row, GRID_SIZE - 1))
        col = max(0, min(col, GRID_SIZE - 1))

        self.agent_position = row * GRID_SIZE + col

        self._update_state()

        reward = -0.01
        terminated = False
        truncated = False

        if self.agent_position == self.goal_position:
            reward = 1.0
            terminated = True

        elif self.agent_position == self.hazard_position:
            reward = -1.0
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self):

        CELL_SIZE = 8
        WINDOW_SIZE = CELL_SIZE * GRID_SIZE

        # Create window only once
        if not hasattr(self, "window"):
            pygame.init()

            self.window = pygame.display.set_mode(
                (WINDOW_SIZE, WINDOW_SIZE)
            )

            pygame.display.set_caption("RL GridWorld")

        # Colors
        WHITE = (255, 255, 255)
        BLUE = (50, 100, 255)
        GREEN = (50, 200, 50)
        RED = (220, 50, 50)

        self.window.fill(WHITE)

        for row in range(GRID_SIZE):

            for col in range(GRID_SIZE):

                position = row * GRID_SIZE + col

                x = col * CELL_SIZE
                y = row * CELL_SIZE

                if self.state[position] == AGENT:

                    pygame.draw.circle(
                        self.window,
                        BLUE,
                        (
                            x + CELL_SIZE // 2,
                            y + CELL_SIZE // 2
                        ),
                        CELL_SIZE // 2
                    )

                elif self.state[position] == GOAL:

                    pygame.draw.rect(
                        self.window,
                        GREEN,
                        (
                            x + 1,
                            y + 1,
                            CELL_SIZE - 2,
                            CELL_SIZE - 2
                        )
                    )

                elif self.state[position] == HAZARD:

                    pygame.draw.rect(
                        self.window,
                        RED,
                        (
                            x + 1,
                            y + 1,
                            CELL_SIZE - 2,
                            CELL_SIZE - 2
                        )
                    )

        pygame.display.flip()
    
    def _get_observation(self):
        observation = np.zeros(
            (3,GRID_SIZE,GRID_SIZE),
            dtype = np.float32
        )

        agent_row,agent_col = divmod(
            self.agent_position,
            GRID_SIZE
        )
        goal_row,goal_col = divmod(
            self.goal_position,
            GRID_SIZE
        )
        hazard_row,hazard_col = divmod(
            self.hazard_position,
            GRID_SIZE
        )
        observation[0,agent_row,agent_col] = 1.0
        observation[1,goal_row,goal_col] = 1.0
        observation[2,hazard_row,hazard_col] = 1.0

        return observation

if __name__ == "__main__":
    pygame.init()

    env = GridWorldEnv()
    env.reset()

    clock = pygame.time.Clock()

    running = True

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:
                    action = UP

                elif event.key == pygame.K_DOWN:
                    action = DOWN

                elif event.key == pygame.K_LEFT:
                    action = LEFT

                elif event.key == pygame.K_RIGHT:
                    action = RIGHT

                else:
                    continue

                observation, reward, terminated, truncated, info = env.step(action)

                print("Reward:", reward)

                if terminated or truncated:
                    print("Episode finished!")
                    env.reset()

        env.render()

        clock.tick(60)

    pygame.quit()