import numpy as np
from neurokit2 import ppg_peaks, ppg_quality
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, welch


def calculate_sampling_rate(timestamps):
    """Calculate the sampling rate based on the time difference between consecutive timestamps."""
    if len(timestamps) < 2:
        return None
    time_diff = np.diff(timestamps)
    # Filter out any negative or zero values that would cause division by zero
    valid_diffs = time_diff[time_diff > 0]
    if len(valid_diffs) == 0:
        return None
    return 1 / np.mean(valid_diffs)

def bandpass_filter(data, lowcut=0.5, highcut=3, fs=30, order=3):
    """Apply a bandpass filter to the data."""
    if fs is None or fs <= 0:
        return np.zeros_like(data)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def diff_normalize_label(label):
    """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
    if len(label) <= 1:
        return np.zeros_like(label)
    diff_label = np.diff(label, axis=0)
    std_val = np.std(diff_label)
    if std_val == 0:
        diffnormalized_label = np.zeros_like(diff_label)
    else:
        diffnormalized_label = diff_label / std_val
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label

def get_hr(y, fs=30, min=30, max=180):
    p, q = welch(y, fs, nfft=int(1e5/fs), nperseg=np.min((len(y)-1, 512)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60 # Using welch method to caculate PSD and find the peak of it.

def welch_spectrum(x, fs, window='hann', nperseg=None, noverlap=None, nfft=None, min=30, max=180):
    """Calculate the Welch power spectrum."""
    if fs is None or fs <= 0 or len(x) < 2:
        return np.array([]), np.array([])
    
    if nperseg is not None and len(x) < nperseg:
        nperseg = 2 ** int(np.log2(len(x)))
        if nperseg < 2:
            nperseg = len(x)
        
        # Adjust noverlap if it's too large
        if noverlap is not None and noverlap >= nperseg:
            noverlap = nperseg // 2
    
    # Use appropriate nfft
    if nfft is None:
        nfft = np.maximum(256, 2 ** int(np.log2(len(x))))
    
    try:
        f, Pxx = welch(x, fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        
        # Filter to only include 0.5-3Hz range (common PPG frequency band)
        mask = (f >= min/60) & (f <= max/60)
        f_filtered = f[mask]
        Pxx_filtered = Pxx[mask]
        
        # Interpolate to match the length of the input signal
        if len(Pxx_filtered) > 0:
            # Create interpolation function
            interp_func = interp1d(
                np.linspace(0, 1, len(Pxx_filtered)),
                Pxx_filtered,
                kind='linear',
                bounds_error=False,
                fill_value=(Pxx_filtered[0], Pxx_filtered[-1])
            )
            
            # Generate interpolated values with the same length as x
            x_points = np.linspace(0, 1, len(x))
            Pxx_interpolated = interp_func(x_points)
            
            return f_filtered, Pxx_interpolated
        else:
            # If no data in the filtered range, return zeros
            return f_filtered, np.zeros(len(x))
    
    except Exception as e:
        print(f"Error in welch spectrum calculation: {e}")
        return np.array([]), np.zeros(len(x))

def single_signal_quality_assessment(signal, fs, method_quality='templatematch', method_peaks='elgendi'):
    assert method_quality in ['templatematch', 'dissimilarity'], "method_quality must be one of ['templatematch', 'dissimilarity']"
    

    signal_filtered = signal
    
    # Check if the signal is too short or has no variation
    if len(signal_filtered) < 10 or np.all(signal_filtered == signal_filtered[0]):
        print("Warning: Signal is too short or constant. Skipping quality assessment.")
        return 0 # Return a high value indicating poor quality

    if method_quality in ['templatematch', 'dissimilarity']:
        method_quality = 'dissimilarity' if method_quality == 'dissimilarity' else method_quality
        
        try:
            # Attempt peak detection on the filtered signal
            _, peak_info = ppg_peaks(
                signal_filtered,
                sampling_rate=fs,
                method=method_peaks
            )
            
            # If no peaks were detected, return a high quality value
            if peak_info["PPG_Peaks"].size == 0:
                print("No peaks detected in the signal. Skipping quality assessment.")
                return 0

            quality = ppg_quality(
                signal_filtered,
                ppg_pw_peaks=peak_info["PPG_Peaks"],
                sampling_rate=fs,
                method=method_quality
            )
            
            # Calculate mean quality excluding NaN values
            quality = np.nanmean(quality)
        
        except ValueError as e:
            print(f"Error in ppg_quality function: {e}")
            quality = 0
        
        return quality

def compute_time_domain_hrv(ppg_signal, fs):
    """
    Calculate time domain HRV metrics from PPG signal.
    
    Args:
        ppg_signal: 1D array of PPG signal data
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary containing HRV metrics: mean_rr, sdnn, rmssd, nn50, pnn50
    """
    if fs is None or fs <= 0 or len(ppg_signal) < 2 * fs:  # Need at least 2 seconds of data
        return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}
    
    try:
        # Detect peaks with minimum distance based on max possible heart rate
        min_distance = int(fs * 60 / 200)  # Minimum distance for 200 bpm
        peaks, _ = find_peaks(ppg_signal, distance=min_distance)
        
        if len(peaks) < 2:
            return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}
        
        # Calculate RR intervals in seconds
        rr_intervals = np.diff(peaks) / fs
        
        # Filter for physiologically plausible RR intervals (250ms to 1500ms)
        valid_rr = rr_intervals[(rr_intervals >= 0.25) & (rr_intervals <= 1.5)]
        
        if len(valid_rr) < 2:
            return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}
        
        # Calculate HRV metrics
        mean_rr = np.mean(valid_rr)
        sdnn = np.std(valid_rr, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2))
        nn50 = np.sum(np.abs(np.diff(valid_rr)) > 0.05)
        pnn50 = (nn50 / len(valid_rr)) * 100 if len(valid_rr) > 0 else 0
        
        return {
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'nn50': nn50,
            'pnn50': pnn50
        }
    
    except Exception as e:
        print(f"Error computing HRV metrics: {e}")
        return {'mean_rr': None, 'sdnn': None, 'rmssd': None, 'nn50': None, 'pnn50': None}

def preprocess_data(data, fs=100):
    """
    Process raw signal data to extract standardized, filtered, and other features.
    
    Args:
        data: Input signal
        fs: Sampling frequency
    
    Returns:
        Tuple containing (raw, standardized, filtered, difference, welch_spectrum) data
    """
    if data is None or len(data) < 2 or fs is None or fs <= 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), (np.array([]), np.array([])))
    
    try:
        # Standardize data
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            standardize_data = np.zeros_like(data)
        else:
            standardize_data = (data - mean_val) / std_val
        
        # Bandpass filter
        filtered_data = bandpass_filter(standardize_data, lowcut=0.5, highcut=3, fs=fs)
        
        # Calculate difference
        difference_data = diff_normalize_label(filtered_data)
        
        # Calculate Welch spectrum
        f, welch_data = welch_spectrum(filtered_data, fs=fs, window='hann', nperseg=min(512, len(filtered_data)-1), min=6, max=180)
        
        return data, standardize_data, filtered_data, difference_data, (f, welch_data)
    
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return (np.array([]), np.array([]), np.array([]), np.array([]), (np.array([]), np.array([])))