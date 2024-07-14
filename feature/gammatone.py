import math
from scipy.signal import lfilter
import numpy as np
import cmath
from typing import Tuple
from constant import *

"""
    Implemented by: NGUYEN Quoc Huy
    The IIR Wavelet gammatone filterbank
    This implementation is an IIR version of the Wavelet gammatone filterbank described in [1].

    References:
    [1] Unoki, M., Akagi, M. (1999). A method of signal extraction from noisy signal based on auditory scene analysis. 
    Speech Communication, 27(3-4), 261-279. https://doi.org/10.1016/S0167-6393(98)00077-6
"""


EarQ = 9.26449
minBW = 24.7
order = 1
EarQ_minBW = EarQ * minBW
GTord = 4
b_coefficient = 1.019


def ERB_N(f_c: float):
    return math.pow(
        math.pow(f_c / EarQ, order) + math.pow(minBW, order), (1. / order))

def transfer_function_analog_to_digital(gain: complex, pole: complex, fs: int) \
        -> Tuple[complex, complex]:
    """ Returns: g and p
        The difference function is y[n] = gx[n] + py[n-1] """
    return gain / fs, cmath.exp(pole / fs)


# noinspection DuplicatedCode
def iir_gammatone_filter(f_c: float, fs: int,
                         scale_factor: float = 1.0,
                         b: float = b_coefficient):
    """
    Mother wavelet g(t)
    g(t) = t^{n-1} * exp(-2 * pi * b * ERB(f_c) * t) * exp(i * 2 * pi * f_c * t)
    Try to return IIR filter corresponding to g_a(t) = g(t/a)
        a is scale factor, a > 0
    Remind that
        g(t)  <->  H(s)
        g(t/a) <-> aH(a*s)
    """
    omega = 2 * math.pi * f_c
    B = 2 * math.pi * b * ERB_N(f_c)
    Amp = scale_factor * (B ** GTord) / (GTord - 1)
    Amp = (1 / scale_factor) * Amp  # Coefficient for analysis-synthesis wavelet
    a_c = 6 * Amp
    a_s = omega * 24 * Amp
    a_z_c_1 = -B + (math.sqrt(2.) + 1.) * omega
    a_z_c_2 = -B + (math.sqrt(2.) - 1.) * omega
    a_z_c_3 = -B - (math.sqrt(2.) + 1.) * omega
    a_z_c_4 = -B - (math.sqrt(2.) - 1.) * omega
    a_z_s_1 = -B
    a_z_s_2 = -B + omega
    a_z_s_3 = -B - omega
    a_p_1 = -B + 1j * omega
    a_p_2 = -B - 1j * omega

    g_c = math.pow(a_c, 1./4.) / scale_factor
    H_c_11 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_1 - a_z_c_1) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_c_12 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_2 - a_z_c_1) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)
    H_c_21 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_1 - a_z_c_2) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_c_22 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_2 - a_z_c_2) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)
    H_c_31 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_1 - a_z_c_3) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_c_32 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_2 - a_z_c_3) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)
    H_c_41 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_1 - a_z_c_4) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_c_42 = transfer_function_analog_to_digital(
        gain=g_c * (a_p_2 - a_z_c_4) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)

    g_s = math.pow(a_s, 1./4.) / scale_factor
    H_s_11 = transfer_function_analog_to_digital(
        gain=g_s * (a_p_1 - a_z_s_1) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_s_12 = transfer_function_analog_to_digital(
        gain=g_s * (a_p_2 - a_z_s_1) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)
    H_s_21 = transfer_function_analog_to_digital(
        gain=g_s * (a_p_1 - a_z_s_2) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_s_22 = transfer_function_analog_to_digital(
        gain=g_s * (a_p_2 - a_z_s_2) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)
    H_s_31 = transfer_function_analog_to_digital(
        gain=g_s * (a_p_1 - a_z_s_3) / (a_p_1 - a_p_2),
        pole=a_p_1 / scale_factor, fs=fs)
    H_s_32 = transfer_function_analog_to_digital(
        gain=g_s * (a_p_2 - a_z_s_3) / (a_p_2 - a_p_1),
        pole=a_p_2 / scale_factor, fs=fs)
    H_s_41 = transfer_function_analog_to_digital(
        gain=g_s / (a_p_1 - a_p_2), pole=a_p_1 / scale_factor, fs=fs)
    H_s_42 = transfer_function_analog_to_digital(
        gain=g_s / (a_p_2 - a_p_1), pole=a_p_2 / scale_factor, fs=fs)

    return [
        (H_c_11, H_c_12), (H_c_21, H_c_22), (H_c_31, H_c_32), (H_c_41, H_c_42)
    ], [
        (H_s_11, H_s_12), (H_s_21, H_s_22), (H_s_31, H_s_32), (H_s_41, H_s_42)
    ]

def plus_filters(H_1, H_2):
    g_1, p_1 = H_1
    g_2, p_2 = H_2
    return [(g_1 + g_2).real, -(p_1 * g_2 + p_2 * g_1).real], \
           [1., -(p_1 + p_2).real, (p_1 * p_2).real]

# Gammatone filter constants
K = NUM_BANDS - 1
f_min = F_MIN
f_max = F_MAX
fs = SAMPLE_RATE
segment_width = 512
alpha_0 = f_max / f_min
f_c = math.sqrt(f_max * f_min)
H_c_k, H_s_k = [], []
Ha = []


for alpha in [alpha_0 ** (k / K) for k in range(-K // 2, K // 2 + 1)]:
    aH_c, aH_s = iir_gammatone_filter(
        f_c, fs, scale_factor=alpha)
    aH_c = [plus_filters(aH_c[d][0], aH_c[d][1]) for d in range(4)]
    aH_s = [plus_filters(aH_s[d][0], aH_s[d][1]) for d in range(4)]
    H_c_k.append(aH_c)
    H_s_k.append(aH_s)
H_c_k = H_c_k[::-1]
H_s_k = H_s_k[::-1]

def calcGammatone(x_t: np.ndarray):
    x_t = x_t.astype(np.float32)
    total_pad = (segment_width - (x_t.shape[-1] % segment_width)) % segment_width
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    x_t = np.pad(x_t, pad_width=((left_pad, right_pad),))
    

    Xr_kt = np.tile(x_t, reps=(K + 1, 1))
    Xi_kt = np.tile(x_t, reps=(K + 1, 1))
    for k in range(K + 1):
        for d in range(4):
            Xr_kt[k, :] = lfilter(
                b=H_c_k[k][d][0], a=H_c_k[k][d][1], x=Xr_kt[k, :])
            Xi_kt[k, :] = lfilter(
                b=H_s_k[k][d][0], a=H_s_k[k][d][1], x=Xi_kt[k, :])
    
    A_kt = np.sqrt(np.square(Xr_kt) + np.square(Xi_kt))

    return A_kt