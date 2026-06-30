import numpy as np


class ProbeEnv:

    def __init__(self):

        self.mass = 1000.0                     # kg
        self.gravity = 13.7                    # m/s^2
        self.drag_coefficient = 2.0
        self.max_thrust = 25000.0              # Newton

        self.planet_radius = 10_700_000.0      # m

        self.dt = 0.1                          # seconds

        self.start_altitude = 1000.0           # m

        self.max_altitude = 1200.0             # runaway condition

        self.safe_landing_velocity = -3.0      # m/s

        self.max_steps = 5000

        # Wind Model
        # Wind States
        # 0 = Calm
        # 1 = Gusty
        # 2 = Gale

        self.wind_drag_multiplier = {
            0: 1.0,
            1: 1.5,
            2: 2.5
        }

        # Transition probabilities

        self.wind_transition = np.array(

            [
                [0.80, 0.15, 0.05],
                [0.20, 0.60, 0.20],
                [0.10, 0.25, 0.65]
            ]

        )

        # Initialise episode

        self.reset()


    def reset(self):
        """
        Starts a new episode.
        Returns
        -------
        state : tuple
            (altitude, velocity, wind)
        """

        self.altitude = self.start_altitude

        self.velocity = 0.0

        self.wind = 0

        self.steps = 0

        return self.get_state()

    # Return Current State
    def get_state(self):

        return (
            self.altitude,
            self.velocity,
            self.wind
        )

    # Gravity Force
    def gravity_force(self):
        """
        F = -mg(1-h/R)
        """
        return (
            -self.mass
            * self.gravity
            * (1.0 - self.altitude / self.planet_radius)
        )

    # Aerodynamic Drag
    def drag_force(self):
        """
        F = k*v^2
        Direction determined using sign(-velocity).
        """
        if self.velocity == 0:
            return 0.0
        multiplier = self.wind_drag_multiplier[self.wind]
        return (
            multiplier
            * self.drag_coefficient
            * self.velocity ** 2
            * np.sign(-self.velocity)
        )

    # Engine Thrust
    def thrust_force(self, action):
        """
        Action
        0 -> Engine OFF
        1 -> Engine ON
        """
        if action == 1:
            return self.max_thrust
        return 0.0

    # Wind Update
    def update_wind(self):
        """
        Samples next wind state according to the transition matrix.
        """
        probabilities = self.wind_transition[self.wind]
        self.wind = np.random.choice(
            [0, 1, 2],
            p=probabilities
        )

    # Reward Function
    def compute_reward(self, previous_altitude, action):

        reward = 0.0
        # Fuel penalty
        if action == 1:
            reward -= 2.0

        # Encourage descent
        altitude_change = previous_altitude - self.altitude
        if altitude_change > 0:
            reward += 0.1 * altitude_change
        else:
            reward += 0.2 * altitude_change

        # Successful landing

        if self.altitude <= 0:
            if self.velocity >= self.safe_landing_velocity:
                reward += 1000.0
            else:
                # Harder crash -> larger penalty
                reward -= (
                    300.0
                    +
                    15.0 * abs(self.velocity)
                )

        # Runaway probe
        if self.altitude > self.max_altitude:
            reward -= 1000.0

        # Timeout
        if self.steps >= self.max_steps:
            reward -= 500.0
        return reward

    # One Environment Step
    def step(self, action):

        self.steps += 1

        previous_altitude = self.altitude

        # Wind evolves first
        self.update_wind()

        # Forces
        gravity = self.gravity_force()
        drag = self.drag_force()
        thrust = self.thrust_force(action)
        net_force = gravity + drag + thrust

        # Newton's Second Law
        acceleration = net_force / self.mass
        # Euler Integration
        self.velocity += acceleration * self.dt
        self.altitude += self.velocity * self.dt

        # Compute reward
        reward = self.compute_reward(
            previous_altitude,
            action
        )

        # Episode termination
        done = False
        if self.altitude <= 0:
            done = True
        elif self.altitude > self.max_altitude:
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # Extra information for debugging/evaluation

        info = {

            "termination": None,

            "safe_landing": False,

            "crash": False,

            "runaway": False,

            "timeout": False

        }

        if self.altitude <= 0:

            if self.velocity >= self.safe_landing_velocity:

                info["termination"] = "safe_landing"
                info["safe_landing"] = True

            else:

                info["termination"] = "crash"
                info["crash"] = True

        elif self.altitude > self.max_altitude:

            info["termination"] = "runaway"
            info["runaway"] = True

        elif self.steps >= self.max_steps:

            info["termination"] = "timeout"
            info["timeout"] = True

        return (
            self.get_state(),
            reward,
            done,
            info
        )
