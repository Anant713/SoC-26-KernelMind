"""
train.py

Train the Q-Learning agent for the Adrian Descent assignment.

Responsibilities
----------------
1. Create environment and agent
2. Run training episodes
3. Update Q-table
4. Record training statistics
5. Plot learning curves
6. Save trained Q-table
"""

import numpy as np
import matplotlib.pyplot as plt

from env import ProbeEnv
from agent import ProbeAgent


# ==========================================================
# Training Parameters
# ==========================================================

NUM_EPISODES = 15000

PRINT_EVERY = 100

MOVING_AVERAGE_WINDOW = 100


# Helper Function
def moving_average(data, window):
    if len(data) < window:
        return np.array(data)
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode="valid")


# Create Environment and Agent
env = ProbeEnv()
agent = ProbeAgent()

# Statistics
episode_rewards = []
episode_lengths = []
landing_success = []
epsilon_history = []


# Training Loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    state = agent.discretize_state(state)
    done = False
    total_reward = 0.0
    steps = 0
    success = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        discrete_next_state = agent.discretize_state(next_state)
        agent.update(
            state,
            action,
            reward,
            discrete_next_state,
            done
        )
        state = discrete_next_state
        total_reward += reward
        steps += 1

    
    # Determine success
    if info["termination"] == "safe_landing":
        success = 1

    
    # Store statistics
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    landing_success.append(success)
    epsilon_history.append(agent.epsilon)

    
    # Reduce exploration
    agent.decay_epsilon()

    
    # Progress print
    if (episode + 1) % PRINT_EVERY == 0:

        average_reward = np.mean(
            episode_rewards[-PRINT_EVERY:]
        )

        average_success = (
            np.mean(
                landing_success[-PRINT_EVERY:]
            )
            * 100
        )

        print(
            f"Episode {episode + 1:5d} | "
            f"Average Reward = {average_reward:8.2f} | "
            f"Landing Success = {average_success:6.2f}% | "
            f"Epsilon = {agent.epsilon:.4f}"
        )


# Save Learned Q-table
agent.save("q_table.npz")
print("\nTraining Complete")

# Plot Reward Curve
reward_curve = moving_average(
    episode_rewards,
    MOVING_AVERAGE_WINDOW
)

plt.figure(figsize=(10,5))
plt.plot(reward_curve)
plt.title("Moving Average Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("reward_curve.png")

# Plot Landing Success Rate

success_curve = moving_average(
    landing_success,
    MOVING_AVERAGE_WINDOW
)

plt.figure(figsize=(10,5))
plt.plot(success_curve)
plt.title("Landing Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success")
plt.grid(True)
plt.savefig("success_curve.png")

# Plot Episode Length
length_curve = moving_average(
    episode_lengths,
    MOVING_AVERAGE_WINDOW
)

plt.figure(figsize=(10,5))
plt.plot(length_curve)
plt.title("Episode Length")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.grid(True)
plt.savefig("length_curve.png")

# Plot Epsilon Decay
plt.figure(figsize=(10,5))
plt.plot(epsilon_history)
plt.title("Exploration Rate")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)
plt.savefig("epsilon_decay.png")

plt.show()