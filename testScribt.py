
import wfdb
from scipy.signal import butter, filtfilt ,find_peaks
import os
import statsmodels.api as sm
import scipy
import numpy as np
import pywt
import pickle

from main import fs


def load_signal(signal_path):

    signal, fields = wfdb.rdsamp(signal_path, channels=[1])
    return signal


def butter_Banpass_filter(data, Low_Cutoff=1.0, High_Cutoff=40.0 , SamplingRate=1000, order=2):
    # nyq=500HZ
    nyq = 0.5 * SamplingRate
    low = Low_Cutoff / nyq
    high = High_Cutoff / nyq
    b, a = butter(order, [low, high],btype='band')
    filtered_signals = []
    filtered = filtfilt(b, a, data[:, 0])
    filtered_signals.append(filtered)

    # Filtered_Data = filtfilt(b, a, data[:, 0])
    return filtered_signals



def segment_signal(Filtered_signal):
    # Define the segment length and overlap (in samples)
    segment_len = 2000
    overlap = 1000

    # Create an empty list to store the segments
    segment_list = []


    # Loop over each ECG signal in the 'ecg_signals' list
    for i in range(len(Filtered_signal)):

        # Calculate the total number of segments in the signal
        num_segments = int(np.ceil((len(Filtered_signal[i]) - segment_len) / overlap)) + 1

        # Loop over each segment in the signal
        for j in range(num_segments):

            # Calculate the start and end indices of the current segment
            start = j * overlap
            end = start + segment_len

            # Make sure the segment doesn't exceed the length of the signal
            if end > len(Filtered_signal[i]):
                end = len(Filtered_signal[i])
                start = end - segment_len

            # Extract the current segment from the signal
            segment = Filtered_signal[i][start:end]

            # Append the segment to the segment list
            segment_list.append(segment)

    return segment_list




def extract_r_peaks(filtered_signal):
    r_peaks, _ = find_peaks(filtered_signal, distance=550)
    return r_peaks


def extract_s_indices(filtered_signal, r_peaks):
    s_point_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.04 * fs)
        local_min_index = np.argmin(filtered_signal[i:i + window_size])
        s_point_indices.append(i + local_min_index)
    return s_point_indices


def extract_q_indices(filtered_signal, r_peaks):
    q_point_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.1 * fs)
        start = i - window_size
        if start < 0:
            start = 0
        local_min_index = np.argmin(filtered_signal[start:i - 1])
        q_point_indices.append(start + local_min_index)
    return q_point_indices


def extract_t_peak(filtered_signal, r_peaks):
    t_peaks_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.4 * fs)
        local_max_index = np.argmax(filtered_signal[i + 40:i + 40 + window_size])
        t_peaks_indices.append(i + 40 + local_max_index)
    return t_peaks_indices


def extract_p_peak(filtered_signal, r_peaks):
    p_peaks_indices = []
    for i in r_peaks:
        # Find the minimum value in a small window after the R-peak
        window_size = int(0.2 * fs)
        start = i - window_size - 40
        if start < 0:
            start = 0
        local_max_index = np.argmax(filtered_signal[start:i - 40])
        p_peaks_indices.append(start + local_max_index)
    return p_peaks_indices


def get_fiducial_points(signal):
    R_indices = extract_r_peaks(signal)
    return R_indices, extract_s_indices(signal, R_indices), extract_q_indices(signal, R_indices), \
        extract_t_peak(signal, R_indices), extract_p_peak(signal, R_indices)


def extract_fiducial_features(signal):
    R_indices, S_indices, Q_indices, T_indices, P_indices = get_fiducial_points(signal)
    features = []
    for j in range(len(P_indices)):
        QT_duration = (T_indices[j] - Q_indices[j]) / fs
        PQ_duration = ((Q_indices[j] - P_indices[j]) / fs) / QT_duration
        PR_duration = ((R_indices[j] - P_indices[j]) / fs) / QT_duration
        PS_duration = ((S_indices[j] - P_indices[j]) / fs) / QT_duration
        PT_duration = ((T_indices[j] - P_indices[j]) / fs) / QT_duration
        QS_duration = ((S_indices[j] - Q_indices[j]) / fs) / QT_duration
        QR_duration = ((R_indices[j] - Q_indices[j]) / fs) / QT_duration
        RS_duration = ((S_indices[j] - R_indices[j]) / fs) / QT_duration
        RT_duration = ((T_indices[j] - R_indices[j]) / fs) / QT_duration
        RP_freq = (signal[R_indices[j]] - signal[P_indices[j]])
        RT_freq = (signal[R_indices[j]] - signal[T_indices[j]])
        TP_freq = (signal[T_indices[j]] - signal[P_indices[j]])
        heart_beat_features = [QT_duration, PQ_duration, PR_duration, PS_duration,
                               PT_duration, QS_duration, QR_duration, RS_duration, RT_duration,
                               RP_freq, RT_freq, TP_freq]
        features.append(heart_beat_features)
    return np.array(features)




def main(signal, model_name):
    #filtring for preprocessing
    filtered_signal = butter_Banpass_filter(signal)
    #segmantation
    segments = segment_signal(filtered_signal)

    #Extraction
    features = extract_fiducial_features(segments)
    print(model_name)
    if model_name == 1:


        clf = pickle.load(open(r"rf_clf_Wavelets.pkl", "rb"))

    else:

        clf = pickle.load(open(r"svm_clf_AC_DCT.pkl","rb"))
    predictions = clf.predict(features)
    # Define subject identification threshold
    identification_threshold = 0.70
    for i in range(4):
        subject_indices = np.where(predictions == i)[0]
        print(len(subject_indices), i)

        if len(subject_indices) / len(predictions) > identification_threshold:
            return i, len(subject_indices) / len(predictions)

    return -1, -1





























