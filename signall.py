import numpy as np
from scipy.signal import hilbert
from configg import (SAMPLE_LENGTH, SYMBOL_LENGTH, NUM_SIGNALS_PER_COMBO, MODULATIONS, DISTANCES, PL_EXPONENT,
                    SNR_MIN, SNR_MAX, LED_SATURATION, LED_BANDWIDTH, CLOCK_JITTER_STD, PHOTODIODE_RISETIME, PHOTODIODE_RES)

# Modulation Generator
def generate_ook():
    num_symbols = SAMPLE_LENGTH // SYMBOL_LENGTH    # integer division = 64 symbols
    bits = np.random.randint(0, 2, num_symbols)     # random bits (0,1)
    signal = np.repeat(bits, SYMBOL_LENGTH)         # each bit repeated 64 times = 32 * 64 = 2048 samples
    return signal.astype(np.float32)                # less memory -> faster


def generate_vppm():
    signal = np.zeros(SAMPLE_LENGTH, dtype=np.float32)   # np.zeros(x,dtype) -> 1D array of data (0.0)
    num_symbols = SAMPLE_LENGTH // SYMBOL_LENGTH         # 64 symbols
    pulse_width = SYMBOL_LENGTH // 3                     # 1/3 design choice for good reading, more (1/5) ->noisy 
    for i in range(num_symbols):
        pos = np.random.randint(0, SYMBOL_LENGTH - pulse_width)  # random pulse position
        start = i * SYMBOL_LENGTH + pos                          
        end = start + pulse_width                                # width range (start -> end)
        if end < SAMPLE_LENGTH:
            signal[start:end] = 1.0                              # light on
    return signal                                                # add the repeated signal [0,0,0,1,1,1,0...]


def generate_csk():
    """bits = log(levels) -> CSK use 6 bits per symbol (IEE 802.15.7)"""
    num_symbols = int(np.ceil(SAMPLE_LENGTH / 3))             # 3 colors, each gets 1 sample/symbol, ceil() -> rounding
    levels = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)  # 4 intensity: off, 33%, 66%, high -> 2 bits per color
    rgb = np.random.choice(levels, size=(num_symbols, 3))        # random, 2D array [[R, G, B],[]....]
    signal = rgb.reshape(-1)[:SAMPLE_LENGTH]                     # flatten to 1D [R,G...], drop extra samples
    return signal


def generate_pam(levels):                                 # level -> 4 & 8-PAM, from mapping (later)
    num_symbols = SAMPLE_LENGTH // SYMBOL_LENGTH          # 64 symbols
    symbols = np.random.randint(0, levels, num_symbols)   # random.randint(low,high,size)   if 4: [0, (0->4), ...()]
    # distinct between 4 & 8 PAM
    amplitudes = (symbols * (1.0 / levels)) + (0.5 / levels) # np array since symbol is an np array (not regular array)

    signal = np.repeat(amplitudes, SYMBOL_LENGTH)         # concatenate amplitude
    return signal.astype(np.float32)


def raised_cosine_filter(beta, sps, span):         # for cap (qam)
    """h(t) = sinc(t/T)*cos(pi beta/T) / (1 - (2 beta t/T)^2)   [general case]
    Special cases: t = 0  t = T/(4 beta)   ((beta -> roll off, how smooth is the filter)"""
    N = span * sps
    t = (np.arange(-N /2, N / 2+1)) / sps
    h = np.zeros_like(t, dtype = np.float64)
    for i, ti in enumerate(t):
        if ti == 0:         # case 1
            h[i] = 1 - beta + 4*beta/np.pi
        elif abs(abs(4 * beta * ti) - 1) < 1e-8:     #case 2
            h[i] = (beta/np.sqrt(2)) * (
                   (1+2/np.sqrt(2)*np.sin(np.pi/4*beta))+(1-2/np.pi)*np.cos(np.pi)/(4*beta)).astype(np.float32)
        else:     # general case
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den    
    norm = np.sqrt(np.sum(h**2))      # normalize to unit 
    if norm > 1e-12:
        h = h / norm
    return h.astype(np.float32)    

def generate_cap(modulation_order):
    """np.convolve(N, M, mode={full, same, valid}, full mode -> returns an output shape of (N+M-1)"""
    sps = SYMBOL_LENGTH        # 32 samples per symbols
    num_symbols = SAMPLE_LENGTH // sps       # 64 symbols
    levels = int(np.sqrt(modulation_order))          # 4 for CAP 16
    beta = 0.5
    span = 6

    i_syms = (np.random.randint(0, levels, num_symbols) / (levels - 1)) * 2 - 1    # random I, Q
    q_syms = (np.random.randint(0, levels, num_symbols) / (levels - 1)) * 2 - 1
    # Upsample by zero-stuffing (standard pulse-shaping preprocessing)
    i_up = np.zeros(num_symbols * sps)
    q_up = np.zeros(num_symbols * sps)
    i_up[::sps] = i_syms
    q_up[::sps] = q_syms

    h = raised_cosine_filter(beta, sps, span)
    h = h / np.sqrt(np.sum(h ** 2))         # normalize

    fc = 0.25         # carrier signal
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    carrier_cos = np.cos(2 * np.pi * fc * n / sps)
    carrier_sin = np.sin(2 * np.pi * fc * n / sps)
    
    h_i = (h * carrier_cos).astype(np.float32)
    h_q = (h * carrier_sin).astype(np.float32)

    h_i = h_i / np.sqrt(np.sum(h_i ** 2))
    h_q = h_q / np.sqrt(np.sum(h_q ** 2))
    # pulse shaping -> smooth the edges, ensures minimal ISI
    i_shaped = np.convolve(i_up, h_i, mode='full')
    q_shaped = np.convolve(q_up, h_q, mode='full')
    # remove filter delay
    delay = span * sps // 2
    i_shaped = i_shaped[delay : delay + num_symbols * sps]
    q_shaped = q_shaped[delay : delay + num_symbols * sps]

    signal = i_shaped + q_shaped

    if len(signal) > SAMPLE_LENGTH:
        signal = signal[:SAMPLE_LENGTH]
    elif len(signal) < SAMPLE_LENGTH:
        signal = np.pad(signal, (0, SAMPLE_LENGTH - len(signal)))
    
    # DC bias + normalize
    dc_bias = np.abs(np.min(signal)) + 0.1
    signal = signal + dc_bias
    signal = signal / np.max(signal) if np.max(signal) > 0 else signal
    return signal.astype(np.float32)


def generate_dco_ofdm():
    num_symbols = SAMPLE_LENGTH // SYMBOL_LENGTH   # 64 symbols
    subcarriers = SYMBOL_LENGTH                    # 32, design choice (= symobl_length), even and power of 2
    signal = np.zeros(SAMPLE_LENGTH, dtype=np.float32)
    for i in range(num_symbols):
        # Hermitian symmetry for real output
        symbols = np.zeros(SYMBOL_LENGTH, dtype=complex)   
        half_sub = subcarriers // 2
        symbols[1:half_sub+1] = (np.random.randn(half_sub) + 1j * np.random.randn(half_sub)) / np.sqrt(2) # normalize
        symbols[SYMBOL_LENGTH-half_sub:] = np.conj(symbols[1:half_sub+1][::-1])       # reverse conjugate the complex
        # Inverse FAST Fourier Transform
        ofdm_symbol = np.fft.ifft(symbols)                       # apply IFFT to each symbol, for real valued 
        real_part = np.real(ofdm_symbol)                         # tiny q value, so take the real 
        dc_bias = np.abs(np.min(real_part)) + 0.1                # don't give fixed dc bias. absolute for the negatives 
        ofdm_signal = real_part + dc_bias                        # add the dc value
        ofdm_signal = ofdm_signal / np.max(ofdm_signal) if np.max(ofdm_signal) > 0 else ofdm_signal   #normalize

        start = i * SYMBOL_LENGTH
        signal[start:start+SYMBOL_LENGTH] = ofdm_signal
    return signal

# Modulation Mapping
MODULATION_GEN = {
    'OOK': generate_ook,
    'VPPM': generate_vppm,
    'CSK': generate_csk,
    '4-PAM': lambda: generate_pam(4),           # lambda is an anonymous fct
    '8-PAM': lambda: generate_pam(8),           # PAM is 1D (levels = modulation order), QAM is 2D 
    'CAP-16': lambda: generate_cap(16),         # modulation_order = 16
    'CAP-64': lambda: generate_cap(64),         # modulation_order = 64
    'DCO-OFDM': generate_dco_ofdm,
}

# Channel Impairments
def apply_led_nonlinearity(signal, saturation=LED_SATURATION):                     # led saturation fom config.py
    signal = np.tanh(signal / saturation) * saturation                     # tanh = hyperbolic tangent (smooth curve)
    ripple = 0.02 * np.sin(2 * np.pi * 10 * np.arange(len(signal)) / len(signal))  # create ripple
    signal = signal + ripple * (signal > 0.7 * saturation)                         # led imperfections near saturation
    return np.clip(signal, 0, 1)                                                   # keep the values between 0 & 1


def apply_bandwidth_limit(signal, bandwidth_factor=LED_BANDWIDTH):
    alpha = 1 - bandwidth_factor
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * filtered[i-1] + (1 - alpha) * signal[i] # combine the previous with the new filtered signal
    return filtered


def apply_clock_jitter(signal, jitter_std=CLOCK_JITTER_STD): 
    jitter = np.random.normal(0, jitter_std, len(signal))                         # create 0.02 random jitter samples
    indices = np.arange(len(signal))                                              # [0, 1, ...., 2047]
    new_indices = indices + jitter                                                # apply jitter
    new_indices = np.clip(new_indices, 0, len(signal)-1)                          # between 0 & 2047
    signal_jittered = signal[np.round(new_indices).astype(int)]                   # rounding
    return signal_jittered


def apply_path_loss(signal, distance):
    scale = 1.0 / (distance ** PL_EXPONENT)         # create attenuation values, ** is ^
    return (signal * scale)                         # apply attenuation


def apply_awgn(signal, snr_db):
    power = np.mean(signal ** 2)
    if power < 1e-12:
        power = 1.0                            # avoid 0 or small values
    snr_linear = 10 ** (snr_db / 10.0)         # SNR(dB) = 10 × log10(SNR_linear) -> SNR_linear = 10^(SNR(dB) / 10)
    noise_power = power / snr_linear           # SNR = Signal / Noise, Noise = Signal / SNR
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)  # same dimension as signal
    return (signal + noise).astype(np.float32) # apply noise


def apply_photodiode_response(signal, responsivity=PHOTODIODE_RES, rise_time_ns=PHOTODIODE_RISETIME):
    signal = signal * responsivity                       # convert light to current, W*W/A = A
    rise_time_s = rise_time_ns * 1e-9                    # ns to s
    cutoff_hz = 0.35 / rise_time_s                       # 0.35 (standard cte), rise-time-to-bandwidth relation
    rc = 1.0 / (2 * np.pi * cutoff_hz)                   # 1 / 2 pi f
    sample_period_s = 1.0 / 44100                        # 1  / sample rate in Hz

    alpha = np.exp(-sample_period_s / rc)                # ~0 -> underflow, nothing change

    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * signal[i]
    return filtered


def apply_channel(signal, distance, snr_db):
    signal = apply_path_loss(signal, distance)
    signal = apply_led_nonlinearity(signal)
    signal = apply_photodiode_response(signal)
    signal = apply_bandwidth_limit(signal)
    signal = apply_clock_jitter(signal)
    # signal = apply_photodiode_response(signal) 
    signal = apply_awgn(signal, snr_db)
    signal = np.clip(signal, 0, 1)             # light intensity is always between 0 and 1
    return signal

# Data cleaning
def clean_signal(signal):
    if np.isnan(signal).any() or np.isinf(signal).any():                  # nan: not a nb, inf: infinity
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)   # remove nan, inf
    return signal


# Dataset generator -> raw data
def generate_dataset(num_samples_per_combo=NUM_SIGNALS_PER_COMBO):
    X_list = []
    y_list = []
    snr_list = []
    total = len(MODULATIONS) * len(DISTANCES) * num_samples_per_combo
    print(f"Generating {total} signals...")                   #f string -> allow variables inside
    for mod_idx, mod_name in enumerate(MODULATIONS):          # enumerate() -> built in fct
        gen_func = MODULATION_GEN[mod_name]
        for dist in DISTANCES:
            for u in range(num_samples_per_combo):
                clean = gen_func()                            # generate a signal

                snr_db = np.random.uniform(SNR_MIN, SNR_MAX)  # apply noise
                
                noisy = apply_channel(clean, dist, snr_db)    # apply impairements
                
                noisy = clean_signal(noisy)                   # clean the signal

                X_list.append(noisy)
                y_list.append(mod_idx)
                snr_list.append(snr_db)
    X = np.array(X_list, dtype=np.float32)                     # signal, 4800 with each 2048 -> [4800,2048]
    y = np.array(y_list, dtype=np.int64)                       # modulation, 1 for each  4800 signals -> [4800,1]
    snr = np.array(snr_list, dtype=np.float32)                 # noise -> [4800,1]
    print(f"Dataset shape: {X.shape}, labels: {y.shape}")
    return X, y, snr