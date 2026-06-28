"""
run.py

Evaluate a trained Q-Learning agent.
"""

import time

from env import ProbeEnv
from agent import ProbeAgent
from renderer import render


# Create Environment

env = ProbeEnv()

agent = ProbeAgent()

agent.load("q_table.npz.npy")

# Disable exploration

agent.epsilon = 0.0


# Start Episode

state = env.reset()

state = agent.discretize_state(state)

done = False

step = 0

total_reward = 0.0


# Evaluation Loop

while not done:

    action = agent.greedy_action(state)

    next_state, reward, done, info = env.step(action)

    render(

        altitude=env.altitude,

        velocity=env.velocity,

        wind=env.wind,

        action=action,

        reward=reward,

        step=step

    )

    time.sleep(0.04)

    total_reward += reward

    state = agent.discretize_state(next_state)

    step += 1

    time.sleep(0.05)


# Episode Summary

print("\n==============================")

print("Evaluation Complete")

print("==============================")

print(f"Steps           : {step}")

print(f"Final Altitude  : {env.altitude:.2f} m")

print(f"Final Velocity  : {env.velocity:.2f} m/s")

print(f"Total Reward    : {total_reward:.2f}")

print()


if info["safe_landing"]:

    print("Mission Result : SAFE LANDING")

elif info["crash"]:

    print("Mission Result : CRASH")

elif info["runaway"]:

    print("Mission Result : RUNAWAY")

elif info["timeout"]:

    print("Mission Result : TIMEOUT")

print("==============================")