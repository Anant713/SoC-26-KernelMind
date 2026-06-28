import numpy as np


class ProbeAgent:

    def __init__(self):

        # Learning parameters
        self.alpha = 0.10
        self.gamma = 0.99

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01

        # State discretization
        self.num_altitude_bins = 50
        self.num_velocity_bins = 50
        self.num_wind_states = 3
        self.num_actions = 2

        # Discretization ranges
        self.altitude_bins = np.linspace(
            0,
            1200,
            self.num_altitude_bins
        )

        self.velocity_bins = np.linspace(
            -100,
            20,
            self.num_velocity_bins
        )

        # Q-table dimensions:
        # Altitude x Velocity x Wind x Action
        self.q_table = np.zeros(
            (
                self.num_altitude_bins,
                self.num_velocity_bins,
                self.num_wind_states,
                self.num_actions
            )
        )

    # Convert continuous state into discrete indices
    def discretize_state(self, state):

        altitude, velocity, wind = state

        altitude_index = np.digitize(
            altitude,
            self.altitude_bins
        )

        velocity_index = np.digitize(
            velocity,
            self.velocity_bins
        )

        altitude_index = np.clip(
            altitude_index,
            0,
            self.num_altitude_bins - 1
        )

        velocity_index = np.clip(
            velocity_index,
            0,
            self.num_velocity_bins - 1
        )

        return (
            altitude_index,
            velocity_index,
            int(wind)
        )

    # Choose action using epsilon-greedy policy
    def choose_action(self, state):

        if np.random.random() < self.epsilon:

            # Explore
            return np.random.randint(
                self.num_actions
            )

        # Exploit
        return np.argmax(
            self.q_table[state]
        )

    # Bellman Update (Q-Learning)
    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        current_q = self.q_table[
            state + (action,)
        ]

        if done:

            target = reward

        else:

            target = (
                reward
                + self.gamma
                * np.max(
                    self.q_table[next_state]
                )
            )

        td_error = target - current_q

        self.q_table[
            state + (action,)
        ] += (
            self.alpha
            * td_error
        )

    # Reduce exploration after every episode
    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # Return greedy action (used during evaluation)
    def greedy_action(self, state):

        return np.argmax(
            self.q_table[state]
        )

    # Save trained Q-table
    def save(self, filename="q_table.npy"):

        np.save(
            filename,
            self.q_table
        )

    # Load trained Q-table
    def load(self, filename="q_table.npy"):

        self.q_table = np.load(
            filename
        )