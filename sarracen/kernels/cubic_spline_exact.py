import numpy as np
from numba import njit


@njit
def pint(sep, d1, d2, h):
    if sep <= 0:
        return 0
    elif sep > 0:
        result = 1
        asep = sep
    else:
        result = -1
        asep = -sep

    q0 = asep * h

    if d1 * d2 >= 0:
        result = result * full_2d_mod(np.abs(d1) / asep, q0) + full_2d_mod(np.abs(d2) / asep, q0)
    else:
        if np.abs(d1) < np.abs(d2):
            result = result * full_2d_mod(np.abs(d2) / asep, q0) - full_2d_mod(np.abs(d1) / asep, q0)
        else:
            result = result * full_2d_mod(np.abs(d1) / asep, q0) - full_2d_mod(np.abs(d2) / asep, q0)

    return result


@njit
def full_2d_mod(tphi, q0):
    phi = np.arctan(tphi)

    if q0 <= 1.0:
        q = q0 / np.cos(phi)

        if q <= 1:
            return F1_2d(phi, q0)
        elif q <= 2.0:
            phi1 = np.arccos(q0)
            return F2_2d(phi, q0) - F2_2d(phi1, q0) + F1_2d(phi1, q0)
        else:
            phi1 = np.arccos(q0)
            phi2 = np.arccos(0.5 * q0)
            return F3_2d(phi) - F3_2d(phi2) + F2_2d(phi2, q0) - F2_2d(phi1, q0) + F1_2d(phi1, q0)

    elif q0 <= 2.0:
        q = q0 / np.cos(phi)

        if q <= 2:
            return F2_2d(phi, q0)
        else:
            phi2 = np.arccos(0.5 * q0)
            return F3_2d(phi) - F3_2d(phi2) + F2_2d(phi2, q0)

    else:
        return F3_2d(phi)


@njit
def F1_2d(phi, q0):
    logs = np.log(np.tan(phi / 2 + np.pi / 4))

    I_2 = np.tan(phi)
    I_4 = (1 / 3) * np.tan(phi) * (2 + 1 / np.cos(phi) ** 2)
    I_5 = (1 / 16) * (0.5 * (11 * np.sin(phi) + 3 * np.sin(3 * phi)) / np.cos(phi) ** 4 + 6 * logs)

    return (5 / 7) * q0 ** 2 / np.pi * (I_2 - (3 / 4) * q0 ** 2 * I_4 + 0.3 * q0 ** 3 * I_5)


@njit
def F2_2d(phi, q0):
    logs = np.log(np.tan(phi / 2 + np.pi / 4))

    I_0 = phi
    I_2 = np.tan(phi)
    I_4 = (1 / 3) * np.tan(phi) * (2 + 1 / np.cos(phi) ** 2)

    I_3 = (1 / 2) * (np.tan(phi) / np.cos(phi) + logs)
    I_5 = (1 / 16) * (0.5 * (11 * np.sin(phi) + 3 * np.sin(3 * phi)) / np.cos(phi) ** 4 + 6 * logs)

    return 5 / 7 * q0 ** 2 / np.pi * (2 * I_2 - 2 * q0 * I_3 + 3 / 4 * q0 ** 2 * I_4 - 1 / 10 * q0 ** 3 * I_5 - 1 / 10 / q0 ** 2 * I_0)


@njit
def F3_2d(phi):
    return 0.5 / np.pi * phi
