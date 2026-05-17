import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- Configurations ---
RESULTS_FILE = "MOT17-04-results.txt"
TARGET_IDS = [395, 708, 711]
FPS = 30.0  # Hz
FC_KINEMATIC = 3.0  # Hz (cutoff frequency for smoothing)
WINDOW_SNR = 30  # frames (1 second)

def load_tracking_data(filepath, target_ids):
    """Load trajectories for specific IDs from MOT results file."""
    data = np.loadtxt(filepath, delimiter=',')
    # Format: frame, id, x, y, w, h, conf, -1, -1, -1
    trajectories = {}
    for tid in target_ids:
        mask = data[:, 1] == tid
        tid_data = data[mask]
        # Sort by frame number just in case
        tid_data = tid_data[np.argsort(tid_data[:, 0])]
        trajectories[tid] = {
            'frame': tid_data[:, 0],
            'x': tid_data[:, 2],
            'y': tid_data[:, 3],
            'w': tid_data[:, 4],
            'h': tid_data[:, 5],
            'conf': tid_data[:, 6]
        }
    return trajectories

def module1_kinematic_smoothing(x_raw, fs, fc):
    """Zero-phase Butterworth low-pass filtering."""
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
    
    # Calculate padlen safely
    padlen = min(3 * max(len(a), len(b)), len(x_raw) - 1)
    if padlen <= 0:
        return x_raw, b, a # Too short to filter
        
    x_smooth = signal.filtfilt(b, a, x_raw, padlen=padlen)
    return x_smooth, b, a

def module2_gait_analysis(h_raw, fs):
    """Spectral Gait Analysis."""
    # Detrending (remove mean and linear trend)
    h_detrend = signal.detrend(h_raw, type='linear')
    
    # FFT
    N = len(h_detrend)
    frequencies = np.fft.rfftfreq(N, d=1/fs)
    fft_values = np.fft.rfft(h_detrend)
    
    # Power Spectral Density (PSD)
    psd = np.abs(fft_values) ** 2 / N
    
    # Peak detection in 1.0Hz - 2.5Hz
    mask = (frequencies >= 1.0) & (frequencies <= 2.5)
    f_peak = None
    is_biological_gait = False
    
    if np.any(mask):
        valid_freqs = frequencies[mask]
        valid_psd = psd[mask]
        peak_idx = np.argmax(valid_psd)
        f_peak = valid_freqs[peak_idx]
        
        # 带外噪声基底比较（比 3×全谱均值更适合低振幅俯视步态）
        noise_mask = ~mask & (frequencies > 0)
        noise_floor = np.mean(psd[noise_mask]) if np.any(noise_mask) else np.mean(psd)
        if valid_psd[peak_idx] > 2.0 * noise_floor:
            is_biological_gait = True
            
    return frequencies, psd, f_peak, is_biological_gait

def module3_snr_evaluation(x_raw, x_smooth, window_size):
    """SNR Evaluation and feedback."""
    residual = x_raw - x_smooth
    
    snr_local = []
    for i in range(len(x_raw) - window_size + 1):
        window_signal = x_smooth[i:i+window_size]
        window_noise = residual[i:i+window_size]
        
        p_signal = np.mean(window_signal**2)
        p_noise = np.mean(window_noise**2)
        
        if p_noise == 0:
            snr = float('inf')
        else:
            snr = 10 * np.log10(p_signal / p_noise)
        snr_local.append(snr)
    
    return np.array(snr_local), residual

def plot_dashboard(trajectories, smooth_data, gait_data, filter_coeffs):
    """Generate the 3-subplot dashboard."""
    fig = plt.figure(figsize=(15, 10))
    
    # (a) Time domain comparison
    ax1 = plt.subplot(2, 2, (1, 2))
    colors = {395: 'blue', 708: 'green', 711: 'red'}
    for tid in TARGET_IDS:
        frames = trajectories[tid]['frame']
        x_raw = trajectories[tid]['x']
        x_smooth = smooth_data[tid]['x']
        
        c = colors.get(tid, 'black')
        ax1.plot(frames, x_raw, color=c, alpha=0.3, linestyle='--', label=f'ID {tid} Raw')
        ax1.plot(frames, x_smooth, color=c, linewidth=2, label=f'ID {tid} Smooth')
        
    ax1.set_title('(a) Time Domain: Zero-phase Kinematic Smoothing (x-coordinate)')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('X Coordinate (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # (b) Frequency domain (PSD) for ID 711
    ax2 = plt.subplot(2, 2, 3)
    tid_gait = 711
    freqs = gait_data[tid_gait]['freqs']
    psd = gait_data[tid_gait]['psd']
    f_peak = gait_data[tid_gait]['f_peak']
    
    ax2.plot(freqs, psd, color='darkorange', linewidth=1.5)
    ax2.axvspan(1.0, 2.5, color='blue', alpha=0.1, label='Biological Gait Band (1.0-2.5Hz)')
    if f_peak is not None:
        ax2.axvline(x=f_peak, color='red', linestyle='--', label=f'Peak: {f_peak:.2f}Hz')
        
    ax2.set_title(f'(b) Spectral Gait Analysis (ID {tid_gait} Height)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim(0, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # (c) Pole-Zero plot
    ax3 = plt.subplot(2, 2, 4)
    b, a = filter_coeffs
    z, p, k = signal.tf2zpk(b, a)
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Unit Circle')
    
    # Poles and Zeros
    ax3.scatter(np.real(z), np.imag(z), s=50, marker='o', facecolors='none', edgecolors='blue', label='Zeros')
    ax3.scatter(np.real(p), np.imag(p), s=50, marker='x', color='red', label='Poles')
    
    ax3.set_title(f'(c) Z-Plane: Butterworth Filter Stability\n(2nd order, fc={FC_KINEMATIC}Hz)')
    ax3.set_xlabel('Real')
    ax3.set_ylabel('Imaginary')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=0.5)
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dsp_analysis_dashboard.png', dpi=300)
    print("Dashboard saved to 'dsp_analysis_dashboard.png'")

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found!")
        return
        
    print("Loading data...")
    trajectories = load_tracking_data(RESULTS_FILE, TARGET_IDS)
    
    smooth_data = {}
    gait_data = {}
    filter_coeffs = None
    
    for tid in TARGET_IDS:
        print(f"\n--- Analyzing ID {tid} ---")
        traj = trajectories[tid]
        
        # Module 1: Smoothing
        x_smooth, b, a = module1_kinematic_smoothing(traj['x'], FPS, FC_KINEMATIC)
        y_smooth, _, _ = module1_kinematic_smoothing(traj['y'], FPS, FC_KINEMATIC)
        filter_coeffs = (b, a) # Same for all
        
        smooth_data[tid] = {'x': x_smooth, 'y': y_smooth}
        
        # Module 2: Gait
        freqs, psd, f_peak, is_biological = module2_gait_analysis(traj['h'], FPS)
        gait_data[tid] = {
            'freqs': freqs,
            'psd': psd,
            'f_peak': f_peak,
            'is_biological': is_biological
        }
        
        if tid == 711:
            print(f"Gait Analysis for ID 711:")
            if f_peak:
                print(f"  Peak frequency (1.0-2.5 Hz): {f_peak:.2f} Hz")
                print(f"  Biological gait detected: {is_biological}")
            else:
                print("  No peak found in 1.0-2.5 Hz range.")
                
        # Module 3: SNR
        snr_local, residual = module3_snr_evaluation(traj['x'], x_smooth, WINDOW_SNR)
        
        print(f"SNR Statistics:")
        print(f"  Min: {np.min(snr_local):.2f} dB")
        print(f"  Max: {np.max(snr_local):.2f} dB")
        print(f"  Mean: {np.mean(snr_local):.2f} dB")
        
        # Adaptive INSA trigger check (simple logic: if mean SNR in last 1/3 of track is lower than first 1/3)
        third = len(snr_local) // 3
        if third > 0:
            early_snr = np.mean(snr_local[:third])
            late_snr = np.mean(snr_local[-third:])
            if late_snr < early_snr - 3.0: # Dropped by 3dB
                print("  => SNR declining trend detected!")
                print("  => TRIGGER RECOMMENDATION: Apply stricter INSA (Noise Inflation) weights for Kalman Filter.")

    # Visualization
    print("\nGenerating dashboard...")
    plot_dashboard(trajectories, smooth_data, gait_data, filter_coeffs)
    print("Done.")

if __name__ == "__main__":
    main()
