import torch
import torch.nn as nn
import torch.nn.functional as F
from configg import NUM_CLASSES, SAMPLE_LENGTH

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()                # super -> call the higher class (nn.Module)                              
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)    # 1 input, 64 filter
        self.bn1 = nn.BatchNorm1d(64)     # normalize output of conv to have mean = 0, std = 1 
        self.pool1 = nn.MaxPool1d(2)      # reduce size by half (reduce signal length, not filters)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)   # take the previous filters and add 32
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_avg = nn.AdaptiveAvgPool1d(1)  # take each the avg of each 128 -> output 1 value

        self.dropout = nn.Dropout(0.4)             # randomly set 30% of neurons to 0

        self.fc = nn.Linear(256, NUM_CLASSES)      # each neuron (256 values) to 8 classes
    
    def forward(self, x):
        x = x.unsqueeze(1)                         # add dimension because conv1D expect 3 dimensions
        
        x = F.relu(self.bn1(self.conv1(x)))        # relu -> put negative values(1D signal+noise) to 0   
        x = self.pool1(x)                          # pool from __init__ -> maxpool
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))         
        x = self.global_avg(x).squeeze(-1)         # remove dimension

        x = self.dropout(x)

        x = self.fc(x)
        return x


class LightweightCNN1D(nn.Module):       # same as 1D CNN, with less filters
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(24)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(24, 40, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(40)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(40, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.global_avg = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.2)         # fewer parameters -> less dropout

        self.fc = nn.Linear(64, NUM_CLASSES)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avg(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class RobustCNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv1d(1, 48, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(48)
        self.pool1 = nn.MaxPool1d(2)
        
        # Block 2 (skip connection)
        self.conv2 = nn.Conv1d(48, 96, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(96)
        self.skip_conv = nn.Conv1d(48, 96, kernel_size=1)    # stride = 1, padding = 0 values by default
        self.pool2 = nn.MaxPool1d(2)
        
        # Block 3 (multi-scale)
        self.conv3a = nn.Conv1d(96, 192, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv1d(96, 192, kernel_size=5, stride=1, padding=2)
        self.conv3c = nn.Conv1d(96, 192, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(576)                      # 192 * 3 = 576

        self.global_avg = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(576, NUM_CLASSES)
    
    def forward(self, x):
        x = x.unsqueeze(1)
    
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)                                 # we need to preserve x in residual connection
        
        residual = self.skip_conv(x)                      # risidual connection happen
        out = F.relu(self.bn2(self.conv2(x)))      
        block2_out = self.pool2(out + residual)           # we need another variable, out = out + x
        
        a = F.relu(self.conv3a(block2_out))
        b = F.relu(self.conv3b(block2_out))
        c = F.relu(self.conv3c(block2_out))
        block3_out = torch.cat([a, b, c], dim=1)          # torch.cat -> concatenate a + b + c
        block3_out = self.bn3(block3_out)
      
        block3_out = self.global_avg(block3_out).squeeze(-1)

        block3_out = self.dropout(block3_out)
        block3_out = self.fc(block3_out)
        return block3_out


def get_model(model_name):    
    models = {
        'Lightweight CNN': LightweightCNN1D,
        '1D CNN': CNN1D,
        'Robust CNN': RobustCNN1D
    }
    if model_name not in models:
        print(f" Model '{model_name}' not found. Using '1D CNN' by default.")
        print(f"   Available models: {list(models.keys())}")
        model_name = '1D CNN'    
    return models[model_name]()

"""nn.conv1d(input channel, output channels, kernel size, stride, padding, dilation...)
 conv2d has 2 inputs (x & y), conv1d has 1 input (x). out channels is the nb of filters
 kernal size: filter size. stride: moves. padding: adding 0 at both ends to keep output size same as input """