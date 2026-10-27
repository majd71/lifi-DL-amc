import torch

# Signal Parameters
SAMPLE_LENGTH = 2048           # samples per 1 signal
SYMBOL_LENGTH = 32             # samples per symbol -> 32 , 2048/32 = 64 symbols/signal
NUM_SIGNALS_PER_COMBO = 1000   # per modulation-SNR combination, 8mod*3dist=24combo, 1000*24=24000total signals 

# 8 Modulation Schemes
MODULATIONS = [
    'OOK',        # Binary intensity
    'VPPM',       # Variable Pulse Position
    'CSK',        # Color Shift Keying
    '4-PAM',      # 4-level Pulse Amplitude
    '8-PAM',      # 8-level Pulse Amplitude
    'CAP-16',     # Carrierless Amplitude Phase
    'CAP-64',
    'DCO-OFDM'    # DC-biased Optical OFDM
]
NUM_CLASSES = len(MODULATIONS)

# Channel Parameters
DISTANCES = [1.0, 2.0, 2.5]       # meters for path loss
PL_EXPONENT = 2                 # 2 for free space (open office), ~1.8 for hospital corridor

# Continuous SNR
SNR_MIN = 0.0
SNR_MAX = 18.0

# LED non-linearity parameters
LED_SATURATION = 0.95             # saturation point (clipping)
LED_BANDWIDTH = 0.95              # bandwidth limitation factor

# Photodiode rise time and responsitivity (e.g. BPW34)
PHOTODIODE_RISETIME = 5.0  # ns
PHOTODIODE_RES = 0.62      # A/W

# Clock jitter parameters
CLOCK_JITTER_STD = 0.005           # standard deviation of timing jitter (fraction of sample)

# Training Hyperparameters
BATCH_SIZE = 128
EPOCHS = 200                      # therre is early stopping in train.py
PATIENCE = 40                     # stop if no improvements
GRAD_CLIP = 1.0                   # gradient clipping ->  during backdrop, fix sharp spikes
LEARNING_RATE_MAX = 0.001
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.01               # for AdamW

# Label Smoothing
LABEL_SMOOTHING = 0.1

# Data Split
TRAIN_RATIO = 0.75         # 18000 train signal / 128 batch = 141 batch per set
VAL_RATIO = 0.15           # 3600 val signal -> 29 batch
TEST_RATIO = 0.10          # 24000*0.1=2400 test signal, 8mods -> 300 each

# Device detection; XPU(intel arc)
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():   #hasattr built-in fct
        print("Intel Arc GPU (XPU) detected!")
        return torch.device("xpu")
    else:
        print("No GPU detected. Using CPU.")
        return torch.device("cpu")
DEVICE = get_device()
print(f"Using device: {DEVICE}")