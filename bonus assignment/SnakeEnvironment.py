import gym
from gym import spaces
import numpy as np
import pygame
import time

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        
        self.grid_size = grid_size
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size, 1), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.snake_dir = 3 
        
        self.food = self._place_food()
        
        self.done = False
        self.score = 0
        
        return self._get_obs()

    def _place_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if food not in self.snake:
                return food

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for x, y in self.snake:
            obs[x, y, 0] = 1
        obs[self.food[0], self.food[1], 0] = -1
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_dir = dirs[action]
        head = self.snake[0]
        new_head = (head[0] + new_dir[0], head[1] + new_dir[1])

        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            return self._get_obs(), -10, self.done, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -1

        return self._get_obs(), reward, self.done, {}

    def render(self, mode="human"):
        cell_size = 20
        screen_size = self.grid_size * cell_size
        
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption('Snake Game')
            self.clock = pygame.time.Clock()
        
        self.screen.fill((0, 0, 0))
        
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (y * cell_size, x * cell_size, cell_size, cell_size))
        
        fx, fy = self.food
        pygame.draw.rect(self.screen, (255, 0, 0), (fy * cell_size, fx * cell_size, cell_size, cell_size))
        
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.screen = None

    def play_human(self):
        self.reset()
        running = True
        pygame.init()
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP and self.snake_dir != 1:
                            self.snake_dir = 0
                        elif event.key == pygame.K_DOWN and self.snake_dir != 0:
                            self.snake_dir = 1
                        elif event.key == pygame.K_LEFT and self.snake_dir != 3:
                            self.snake_dir = 2
                        elif event.key == pygame.K_RIGHT and self.snake_dir != 2:
                            self.snake_dir = 3

                if not self.done:
                    _, reward, self.done, _ = self.step(self.snake_dir)
                    time.sleep(0.5)
                    self.render()

                if self.done:
                    print(f"Game Over! Your score: {self.score}")
                    running = False
        finally:
            self.close()

if __name__ == "__main__":
    env = SnakeEnv(grid_size=10)
    env.snake_dir = 3  
    env.play_human()
