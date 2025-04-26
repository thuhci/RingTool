import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch


def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """Apply a bandpass filter to the data."""
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_rr(y, fs=100, min=6, max=30, method = 'fft'):
    """
    Calculate heart rate using the fft or peak detection
    y: 0.1-0.5 Hz filtered PPG signal, only support 1D array
    fs: sampling frequency
    min: minimum heart rate
    max: maximum heart rate
    method: 'fft' or 'peak'
    """
    if y.shape[1]!=1:
        raise ValueError("y should be a 1D array, but got a 2D array.")
    # Check if y is a 2D array and flatten it to 1D
    if len(y.shape) > 1:
        y = y.flatten()
        
    if method == 'fft':
        p, q = welch(y, fs, nfft=int(5e6/fs), nperseg=np.min((len(y)-1, 512)))
        fft_rr = p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
        return fft_rr
    elif method == 'peak':
        ppg_peaks, _ = find_peaks(y)
        if len(ppg_peaks) > 1:  # Need at least 2 peaks to calculate intervals
            peak_intervals = np.diff(ppg_peaks) / fs
            if np.mean(peak_intervals) > 0:  # Avoid division by zero
                peak_rr = 60 / np.mean(peak_intervals)
            else:
                peak_rr = 15.0  # Default if intervals are zero
        else:
            peak_rr = 15.0  # Default if insufficient peaks found
        return peak_rr
    else:
        raise ValueError("Invalid method. Choose 'fft' or 'peak'.")
