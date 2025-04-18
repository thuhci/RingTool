import numpy as np

def get_spo2(ppg_red, ppg_ir, fs=100, method="ratio"):
    """
    Calculate SpO2 from raw PPG signals.
    """
    # Assuming ppg_red and ppg_ir are the raw PPG signals for red and infrared channels
    ppg_min = min(np.min(ppg_red), np.min(ppg_ir))
    ratio = np.mean(ppg_red-ppg_min) / np.mean(ppg_ir-ppg_min)
    print(f"Ratio: {ratio}")
    # Calibrate ratio to SpO2 (this step requires a calibration curve, here simplified)
    spo2 = 100 - 5 * ratio  # Example simplified calibration
    spo2 = np.clip(spo2, 80, 100)  # Ensure SpO2 is within valid range
    return spo2

# Example usage
ppg_red = np.random.rand(1000)  # Red channel signal
ppg_ir = np.random.rand(1000)   # IR channel signal
spo2 = get_spo2(ppg_red, ppg_ir)
print(f"SpO2: {spo2}%")
