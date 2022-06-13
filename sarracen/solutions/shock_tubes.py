import numpy as np
import pandas as pd

from sarracen import SarracenDataFrame


def exact_shock(time, gamma, x_shock, left_rho, right_rho, left_pressure, right_pressure, left_velocity, right_velocity,
                dust_gas_ratio, left_plot, right_plot, pixels):
    if left_rho <= 0 or right_rho <= 0:
        raise ValueError("rho cannot be less than zero!")
    if left_pressure <= 0 or right_pressure <= 0:
        raise ValueError("pressure cannot be less than zero!")
    if gamma < 1:
        raise ValueError("gamma cannot be less than one!")

    x_zero = x_shock

    left_sound_speed = np.sqrt(gamma * left_pressure / left_rho)
    right_sound_speed = np.sqrt(gamma * right_pressure / right_rho)
    if not dust_gas_ratio == 0:
        left_sound_speed = left_sound_speed * np.sqrt(1 / (1 + dust_gas_ratio))
        right_sound_speed = right_sound_speed * np.sqrt(1 / (1 + dust_gas_ratio))
    gamma_factor = (gamma - 1) / (gamma + 1)

    post_pressure, post_velocity = get_pstar(gamma, left_pressure, right_pressure, left_velocity, right_velocity,
                                             left_sound_speed, right_sound_speed)

    left_is_shock = post_pressure > left_pressure
    right_is_shock = post_pressure > right_pressure

    if right_is_shock:
        post_right_velocity = right_velocity + right_sound_speed ** 2\
                              * (post_pressure / right_pressure - 1) / (gamma * (post_velocity - right_velocity))
    else:
        post_right_velocity = right_sound_speed + 0.5 * (gamma + 1) * post_velocity - 0.5 * (gamma - 1) * right_velocity

    if left_is_shock:
        post_left_velocity = -(left_velocity + left_sound_speed ** 2
                             * (post_pressure / left_pressure - 1) / (gamma * (post_velocity - left_velocity)))
    else:
        post_left_velocity = left_sound_speed - 0.5 * (gamma + 1) * post_velocity + 0.5 * (gamma - 1) * left_velocity

    x_left = x_zero - post_left_velocity * time
    if left_is_shock:
        x_left_going = x_left
    else:
        x_left_going = x_zero - (left_sound_speed - left_velocity) * time

    x_contact = x_zero + post_velocity * time

    x_right = x_zero + post_right_velocity * time
    if right_is_shock:
        x_right_going = x_right
    else:
        x_right_going = x_zero + (right_sound_speed + right_velocity) * time

    pressure, density, velocity = np.zeros(pixels), np.zeros(pixels), np.zeros(pixels)
    x_plot = np.linspace(left_plot, right_plot, pixels)

    # undisturbed medium on left
    pressure[x_plot <= x_left_going] = left_pressure
    density[x_plot <= x_left_going] = left_rho
    velocity[x_plot <= x_left_going] = left_velocity

    # inside expansion fan
    density[(x_plot > x_left_going) & (x_plot < x_left)] = \
        left_rho * (gamma_factor * (x_zero - x_plot[(x_plot > x_left_going) & (x_plot < x_left)]) / (left_sound_speed * time) + (1 - gamma_factor))\
        ** (2 / (gamma - 1))
    pressure[(x_plot > x_left_going) & (x_plot < x_left)] = left_pressure * (density[(x_plot > x_left_going) & (x_plot < x_left)] / left_rho) ** gamma
    velocity[(x_plot > x_left_going) & (x_plot < x_left)] = \
        (1 - gamma_factor) * (left_sound_speed - (x_zero - x_plot[(x_plot > x_left_going) & (x_plot < x_left)]) / time) + gamma_factor * left_velocity

    # between expansion fan and contact discontinuity
    pressure[(x_plot >= x_left) & (x_plot < x_contact)] = post_pressure
    density[(x_plot >= x_left) & (x_plot < x_contact)] = left_rho * (post_pressure / left_pressure) ** (1 / gamma)
    velocity[(x_plot >= x_left) & (x_plot < x_contact)] = post_velocity

    # post-shock, ahead of contact discontinuity
    pressure[(x_plot >= x_contact) & (x_plot < x_right)] = post_pressure
    density[(x_plot >= x_contact) & (x_plot < x_right)] = right_rho * (gamma_factor + post_pressure / right_pressure) / (1 + gamma_factor * post_pressure / right_pressure)
    velocity[(x_plot >= x_contact) & (x_plot < x_right)] = post_velocity

    # undisturbed medium to the right
    pressure[x_plot >= x_right] = right_pressure
    density[x_plot >= x_right] = right_rho
    velocity[x_plot >= x_right] = right_velocity

    df = pd.DataFrame({'x': x_plot, 'P': pressure, 'rho': density, 'v': velocity, 'u': pressure / ((gamma - 1) * density)})
    return SarracenDataFrame(df, params=dict())


def get_pstar(gamma, left_pressure, right_pressure, left_velocity, right_velocity, left_sound_speed, right_sound_speed):
    power = (gamma - 1) / (2 * gamma)
    denominator = left_sound_speed / left_pressure ** power + right_sound_speed / right_pressure ** power
    new_pressure = ((left_sound_speed + right_sound_speed +
                     (left_velocity - right_velocity) * 0.5 * (gamma - 1)) / denominator) ** (1 / power)
    pressure = left_pressure
    iterations = 0

    while not np.isclose(new_pressure, pressure, atol=1.5e-12) and iterations < 30:
        iterations += 1
        pressure = new_pressure

        left_function = pressure_function(pressure, left_pressure, left_sound_speed, gamma)
        left_derivative = pressure_derivative(pressure, left_pressure, left_sound_speed, gamma)
        right_function = pressure_function(pressure, right_pressure, right_sound_speed, gamma)
        right_derivative = pressure_derivative(pressure, right_pressure, right_sound_speed, gamma)

        function = left_function + right_function + (right_velocity - left_velocity)
        derivative = left_derivative + right_derivative

        pressure_change = -function / derivative
        new_pressure = pressure + pressure_change

    return new_pressure, left_velocity - left_function


def pressure_function(pressure_star, pressure, sound_speed, gamma):
    H = pressure_star / pressure

    if H > 1:
        denominator = gamma * ((gamma + 1) * H + (gamma - 1))
        term = np.sqrt(2 / denominator)
        return (H - 1) * sound_speed * term
    else:
        power = (gamma - 1) / (2 * gamma)
        return (H ** power - 1) * (2 * sound_speed / (gamma - 1))


def pressure_derivative(pressure_star, pressure, sound_speed, gamma):
    H = pressure_star / pressure

    if H > 1:
        denominator = gamma * ((gamma + 1) * H + (gamma - 1))
        term = np.sqrt(2 / denominator)
        return sound_speed * term / pressure + (H - 1) * sound_speed / term * (-1 / denominator ** 2) * gamma *\
               (gamma + 1) / pressure
    else:
        power = (gamma - 1) / (2 * gamma)
        return 2 * sound_speed / (gamma - 1) * power * H ** (power - 1) / pressure
    pass