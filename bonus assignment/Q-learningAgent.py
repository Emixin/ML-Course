import numpy as np
import random
import matplotlib.pyplot as plt
from SnakeEnvironment import SnakeEnv


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table as a dictionary
        self.q_table = {}
        self.visited = set()  # Set to track visited positions

    def get_state(self):
        """
        The state is a tuple that includes the snake's head, food position,
        direction, and a hashable version of visited positions.
        """
        head = self.env.snake[0]
        food = self.env.food
        direction = self.env.snake_dir
        state = (head, food, direction, tuple(self.visited))
        return state

    def get_q_values(self, state):
        """
        Retrieve Q-values for the given state, initializing if not already present.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space.n)
        return self.q_table[state]

    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy strategy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        q_values = self.get_q_values(state)
        return np.argmax(q_values)  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for the given state-action pair using the Bellman equation.
        """
        current_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def train(self, episodes=1000, render_last=10):
        """
        Train the agent over a number of episodes and render the last few.
        """
        rewards = []  # Track total reward per episode
        scores = []   # Track game score per episode

        for episode in range(episodes):
            state = self.env.reset()
            self.visited = set()  # Reset visited positions at the start of each episode
            self.visited.add(self.env.snake[0])  # Add initial position to visited
            state = self.get_state()
            total_reward = 0

            while True:
                action = self.choose_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.get_state()

                # Add the new head position to visited positions
                self.visited.add(self.env.snake[0])

                # Penalize revisiting positions
                if self.env.snake[0] in self.visited:
                    reward -= 5

                self.update_q_value(state, action, reward, next_state)

                # Render the last 10 episodes
                if episode >= episodes - render_last:
                    self.env.render()

                state = next_state
                total_reward += reward

                if done:
                    break

            # Record the total reward and score for the episode
            rewards.append(total_reward)
            scores.append(len(self.env.snake) - 1)

            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Score: {len(self.env.snake) - 1}, Epsilon: {self.epsilon:.2f}")

        return rewards, scores


def plot_training_progress(rewards, scores):
    """
    Plot the training rewards and scores over episodes.
    """
    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid()

    # Plot scores
    plt.subplot(1, 2, 2)
    plt.plot(scores, label='Score per Episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Score (Snake Length)')
    plt.title('Training Scores')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = SnakeEnv(grid_size=10)
    agent = QLearningAgent(env)

    print("Training the agent...")
    rewards, scores = agent.train(episodes=1000)

    env.close()

    # Plot the training progress
    print("Plotting training progress...")
    plot_training_progress(rewards, scores)