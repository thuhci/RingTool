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

def get_hr(y, fs=100, min=30, max=180, method = 'fft'):
    """
    Calculate heart rate using the fft or peak detection
    y: filtered PPG signal, only support 1D array
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
        fft_hr = p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
        return fft_hr
    elif method == 'peak':
        ppg_peaks, _ = find_peaks(y)
        # Check if we have enough peaks to calculate heart rate
        if len(ppg_peaks) <= 1:
            return 80.0  # Default heart rate if insufficient peaks found
        
        # Calculate differences between peaks and their mean
        peak_diffs = np.diff(ppg_peaks)
        mean_diff = np.mean(peak_diffs)
        
        # Check for invalid mean (zero or negative)
        if mean_diff <= 0:
            return 80.0  # Default heart rate
            
        # Calculate heart rate
        peak_hr = 60 / (mean_diff / fs)
        
        # Handle any NaN or infinite values
        if not np.isfinite(peak_hr):
            return 80.0  # Default heart rate
            
        return peak_hr
    else:
        raise ValueError("Invalid method. Choose 'fft' or 'peak'.")
