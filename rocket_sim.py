import numpy as np

import config


def runge_kutta4(fun, x0, y0, h):
    """Standard RK4 single-step integrator."""
    k_0 = fun(x0, y0)
    k_1 = fun(x0 + h/2, y0 + h/2 * k_0)
    k_2 = fun(x0 + h/2, y0 + h/2 * k_1)
    k_3 = fun(x0 + h, y0 + h * k_2)

    k = 1/6 * (k_0 + 2.0*k_1 + 2.0*k_2 + k_3)
    return y0 + h * k


class Rocket1DSim:
    """Minimal 1D vertical rocket-landing physics (point mass).

    State is (h, v, m) = altitude [m], velocity [m/s], total mass [kg].
    Positive up: descending rocket has v < 0. Thrust up, gravity down,
    drag opposes motion. Integrated with RK4.
    """

    def __init__(
        self,
        dry_mass=config.DRY_MASS,
        fuel_mass=config.FUEL_MASS,
        max_thrust=config.MAX_THRUST,
        isp=config.ISP,
        g=config.GRAVITY,
        rho=config.AIR_DENSITY,
        cd_a=config.DRAG_AREA,
        init_altitude=config.INIT_ALTITUDE,
        init_velocity=config.INIT_VELOCITY,
        dt=config.DT,
    ):
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.max_thrust = max_thrust
        self.isp = isp
        self.v_e = isp * config.G0   # exhaust velocity [m/s], from Isp
        self.g = g
        self.rho = rho
        self.cd_a = cd_a
        self.init_altitude = init_altitude
        self.init_velocity = init_velocity
        self.dt = dt
        self.reset()

    def reset(self):
        self.h = self.init_altitude
        self.v = self.init_velocity
        self.m = self.dry_mass + self.fuel_mass
        return self.state

    @property
    def state(self):
        return np.array([self.h, self.v, self.m])

    def step(self, throttle):
        """Advance one dt at the given throttle in [0, 1].

        Returns ``terminated`` (bool), True once the rocket reaches the ground.
        """
        throttle = float(np.clip(throttle, 0.0, 1.0))

        def derivatives(t, state):
            h, v, m = state
            fuel = m - self.dry_mass
            if fuel <= 0.0:
                thrust = 0.0
                mdot = 0.0
            else:
                thrust = throttle * self.max_thrust
                mdot = thrust / self.v_e
            drag = -0.5 * self.rho * self.cd_a * v * abs(v)
            a = (thrust - m * self.g + drag) / m
            return np.array([v, a, -mdot])

        state = runge_kutta4(derivatives, 0.0, np.array([self.h, self.v, self.m]), self.dt)
        self.h, self.v, self.m = state
        self.m = max(self.m, self.dry_mass)

        terminated = self.h <= 0.0
        return terminated
