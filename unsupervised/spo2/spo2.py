import numpy as np
from scipy.signal import find_peaks
import pandas as pd

def get_spo2(ppg_ir, ppg_red, fs=100, ring_type="ring1", method="ratio"):
    """
    Calculate SpO2 from raw PPG signals using a moving window approach.
    
    Parameters:
    -----------
    ppg_red : array-like
        Red channel PPG signal
    ppg_ir : array-like
        IR channel PPG signal
    fs : int, optional
        Sampling frequency in Hz, default is 100
    method : str, optional
        Method to calculate SpO2, default is "ratio"
    """
    
    # Convert inputs to numpy arrays if they aren't already
    ppg_red = np.array(ppg_red)
    ppg_ir = np.array(ppg_ir)
    
    # Calculate AC component (peak-to-peak)
    ac_red = np.max(ppg_red) - np.min(ppg_red)
    ac_ir = np.max(ppg_ir) - np.min(ppg_ir)
    
    # Calculate DC component (mean value)
    dc_red = np.mean(ppg_red)
    dc_ir = np.mean(ppg_ir)
    
    # Avoid division by zero
    epsilon = 1e-6
    dc_red = max(dc_red, epsilon)
    dc_ir = max(dc_ir, epsilon)
    
    # Calculate AC/DC for each signal
    acDivDcRed = ac_red / dc_red
    acDivDcIr = ac_ir / dc_ir
    
    # Calculate ratio
    ratio = acDivDcRed / acDivDcIr
    
    # Calculate SpO2 using the formula: 
    if ring_type == "ring1":
        SPO2 = 99 - 6 * ratio
    elif ring_type == "ring2":
        SPO2 = 87 + 6 * ratio
    
    # Print mean ratio for debugging
    # print(f"Ratio: {np.mean(ratio)}")
    
    # Ensure SpO2 is within valid range
    SPO2 = np.clip(SPO2, 80, 99)
    
    # Return the mean SpO2 value
    # return ratio
    return float(np.mean(SPO2))

# Example usage
ppg_red = np.random.rand(1000)  # Red channel signal
ppg_ir = np.random.rand(1000)   # IR channel signal
spo2 = get_spo2(ppg_red, ppg_ir)
print(f"SpO2: {spo2}%")
