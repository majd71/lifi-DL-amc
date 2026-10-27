import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from configg import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE

class LiFiDataset(Dataset):                                                # modify data
    def __init__(self, X, y, snr, augment = False, normalize = True):      # __init__ built in fct, initialise
        self.X = torch.tensor(X, dtype=torch.float32)                      
        self.y = torch.tensor(y, dtype=torch.long)                         # modulation label
        self.snr = torch.tensor(snr, dtype=torch.float32)                  # continuous snr
        self.augment = augment                                             # False -> not enable
        self.normalize = normalize                                         # for z score
        
    def __len__(self):                # __len__ is built in special fct
        return len(self.X)            # returns the number of signals -> 4800

    def __getitem__(self, idx):       # DataLoader, built in fct, gives index
        x = self.X[idx]
        y = self.y[idx]
        snr = self.snr[idx]
    
        # Z-Score Normalization during training
        if self.normalize:
            mean = torch.mean(x, dim=-1, keepdim=True)   # Keepdim -> keep dimension of matrix
            std = torch.std(x, dim=-1, keepdim=True)
            eps = 1e-8                                   # small epsilon to preved division by 0
            x = (x - mean) / (std + eps)
    
        # Data augmentation (shift + scale + noise) 
        if self.augment and np.random.random() > 0.5:
            shift = np.random.randint(-16, 16)
            x = torch.roll(x, shifts=shift)          # shift
            scale = np.random.uniform(0.85, 1.15)
            x = x * scale                            # scale
            if np.random.random() > 0.5:
                x = x + torch.randn_like(x) * 0.01   # add 1% small noise

        return x, y, snr


def get_dataloaders(X, y, snr, batch_size = BATCH_SIZE, normalize = True, stratify = True):
  
    train_ratio = TRAIN_RATIO                # 0.75
    temp_ratio = VAL_RATIO + TEST_RATIO  # validation and test compined = 0.15 + 0.1 = 0.25
    
    # Step 1: Split into train and temp (val+test)
    X_train, X_temp, y_train, y_temp, snr_train, snr_temp = train_test_split(
        X, y, snr,                           # signal, modulation, noise
        test_size = temp_ratio,              # 25% to temp
        train_size = train_ratio,            # 75% to train
        random_state = 42,                   # randow value, to avoid getting random accuracy each rerun
        stratify = y                         # stratify -> for balanced modulations labels in each set, not random
    )
    
    # Step 2: Split temp into validation and test
    val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # 0.15/0.25 = 0.6 -> 60% of 25% -> validation
    X_val, X_test, y_val, y_test, snr_val, snr_test = train_test_split(
        X_temp, y_temp, snr_temp,
        test_size = 1 - val_ratio,                     # 1 - 0.6 = 0.4 -> 40% of 25% for test
        train_size = val_ratio,
        random_state = 42,
        stratify = y_temp
    )

    # Create datasets (augment only for train)
    train_ds = LiFiDataset(X_train, y_train, snr_train, augment = True, normalize = True)
    val_ds = LiFiDataset(X_val, y_val, snr_val, augment = False, normalize = True)
    test_ds = LiFiDataset(X_test, y_test, snr_test, augment = False, normalize = True)
    
    # Create DataLoaders (shuffle -> to avoid memorize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle = True, num_workers = 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle = False, num_workers = 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle = False, num_workers = 0)
    return train_loader, val_loader, test_loader


""" LiFiDataset custom class inherits from built in class (Dataset)
 torch tensor works on cpu & gpu, numpy only on cpu
 without seed: random split changes every time you run, with seed=42: same split
 random split-> choose random samples for each set
 test_size, train_size, stratify, random_state are built in parameters in train_test_split()
 train_test_split only splits into 2 parts, so we first split into Train vs (Val+Test) -> automatically."""