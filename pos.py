# sys.path.append('./SkinDetector/')
import os
import sys
import warnings

import cv2
# import dlib
# import insightface
import matplotlib.pyplot as plt
# import mediapipe as mp
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks, savgol_filter, welch


def get_pos_rppg(mean_rgb, fps=30, l_factor=1.6, bp=None, savgol=False, order=5):
    l = int(fps * l_factor)
    rPPG = np.zeros(mean_rgb.shape[0])
    
    for t in range(0, mean_rgb.shape[0]-l+1):
        C = mean_rgb[t:t+l,:].T
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.pinv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv, C)
        proj_mat = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(proj_mat, Cn)
        std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        P = np.matmul(std, S)
        rPPG[t:t+l] = rPPG[t:t+l] + (P-np.mean(P))/np.std(P)
        
    # if bandpass is not none
    if bp is not None and len(bp) == 2:
        b, a = signal.butter(order, [bp[0], bp[1]], btype='bandpass', fs=fps)
        rPPG = signal.filtfilt(b, a, rPPG)
        
    if savgol:
        rPPG = savgol_filter(rPPG, 23, 2)
    return rPPG

def smooth_signal(data, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth the data.

    :param data: numpy array of 1D signal data.
    :param window_length: Length of the filter window (must be an odd integer).
    :param polyorder: Order of the polynomial used in the filtering. Must be less than window_length.
    :return: Smoothed data.
    """
    # Ensure the window_length is odd
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")

    # Apply the filter
    smoothed_data = savgol_filter(data, window_length, polyorder)
    
    return smoothed_data
    
def welch_bpm(signal, n_seg = 6, fs = 30):
    # seg_len = (2 * signal.shape[0]) // (n_seg + 1)
    seg_len = len(signal)
    f, Pxx = welch(signal, fs=fs, nperseg=seg_len, window='flattop')
    f_max = f[np.argmax(Pxx)]
    bpm = 60 * f_max
    return bpm


