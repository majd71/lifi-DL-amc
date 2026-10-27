import numpy as np
from signall import MODULATION_GEN, apply_channel

# Average over many signals per modulation for statistical robustness
N_TRIALS = 100

for name in ['OOK', '4-PAM', '8-PAM', 'CAP-16', 'CAP-64']:
    correlations = []
    effective_snrs = []
    
    for _ in range(N_TRIALS):
        signal = MODULATION_GEN[name]()
        noisy = apply_channel(signal, 2.0, 10)
        
        corr = np.corrcoef(signal, noisy)[0, 1]
        correlations.append(corr)
        
        signal_var = np.var(signal)
        noise_var = np.var(noisy - signal)
        effective_snrs.append(10 * np.log10(signal_var / noise_var))
    
    print(f"{name:10s}: corr={np.mean(correlations):+.4f} ± {np.std(correlations):.4f}, "
          f"eff SNR={np.mean(effective_snrs):+.2f} ± {np.std(effective_snrs):.2f} dB")