import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json    #json (JavaScript Object Notation) to save and load data in a readable format
from configg import MODULATIONS, DEVICE, SNR_MIN, SNR_MAX, DISTANCES, PL_EXPONENT
from signall import (MODULATION_GEN, apply_path_loss, apply_led_nonlinearity,
                     apply_bandwidth_limit, apply_clock_jitter, apply_awgn)

def plot_training_history(history, model_name, save_path=None):   
    """Plot training and validation loss/accuracy curves.
       history from train.py , path to be override in main.py (=false)
       ax1.plot -> plot on the left side. ax2.plot -> plot on the second side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))   # create 2 plots (1 row, 2 column)
    
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2, color='#1f77b4')  # line 1 = train loss
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2, color='#ff7f0e')      # line 2 = validation loss
    ax1.set_xlabel('Epoch')                                                              # x-axis = epoch
    ax1.set_ylabel('Loss')                                                               # y-axis = loss
    ax1.set_title(f'{model_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)    # alpa -> grid opacity
    
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2, color='#1f77b4')
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2, color='#ff7f0e')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)         # limit accuracy plot between 0 & 1.05
    
    plt.tight_layout()            # plot properly spaced
    if save_path:                                                                 # save path
        plt.savefig(f'{save_path}/{model_name.replace(" ", "_")}_history.png',    # eplace"space" by "_"
                   dpi=300, bbox_inches='tight')                                  # dot per inch(resolution), margin crop
    #plt.show()
    plt.close()
    return fig


def plot_confusion_matrix(model, test_loader, model_name, save_path=None):
    """Generate and plot confusion matrix. confusion_matrix() -> built in fct froms scikitlearn
       evaluation mode -> dropout disabled
       seaborn -> modern visualization from matplotlib
       2D data, with annotations, string format = integer, map color = blue, with colorbar
       """
    model.eval()                       
    all_preds = []          # predictions and labels from train.py
    all_labels = []
    with torch.no_grad():  # tells PyTorch to not calculate the gradients as that would affect the back propagation 
        for X, y, k in test_loader:                   # signal, modulation
            X, y = X.to(DEVICE), y.to(DEVICE)         # move to XPU -> faster processing
            outputs = model(X)                        # scores for each run
            k, preds = torch.max(outputs, 1)          # max value allond dimension 1 of outputs
            all_preds.extend(preds.cpu().numpy())     # convert tensors to cpu numpy array, matplotlib don't work with GPU
            all_labels.extend(y.cpu().numpy())        # extend -> add all predictions & labels  
    
    cm = confusion_matrix(all_labels, all_preds)      # matrix of true and predicted labels
    
    fig, ax = plt.subplots(figsize=(10, 8))           # 8x8 matrix, 10 inch wide 8 inch tall (8 modulations)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',         
                xticklabels=MODULATIONS, yticklabels=MODULATIONS,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} - Confusion Matrix')
    plt.xticks(rotation=45, ha='right')              # x modulation rotate by 45 degree, align right
    plt.yticks(rotation=0)                           # y modulations no rotate
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/{model_name.replace(" ", "_")}_cm.png', 
                   dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    return cm


def plot_modulation_accuracy(model, test_loader, model_name, save_path=None):
    """Plot per-modulation accuracy.
       zip() -> built in fct that t combines elements from multiple iterables (list)"""
    model.eval()
    class_prediction = {mod: 0 for mod in MODULATIONS}   # for loop, initialize class prediction = 0, for each modulation 
    class_samples = {mod: 0 for mod in MODULATIONS}      # initialize total sample = 0
    with torch.no_grad():                # disable gradient, and move to XPU (faster inference)
        for X, y, h in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            h, preds = torch.max(outputs, 1)             # get the prediction
            for label, pred in zip(y.cpu().numpy(), preds.cpu().numpy()):   # zip concatenates modulation label with pred
                mod = MODULATIONS[label]             # get modulation name
                class_samples[mod] += 1              # move to next sample
                if label == pred:                    # if prediction is correct
                    class_prediction[mod] += 1       # next modulation. if != , do nothing (by default)
    
    mod_names = list(class_prediction.keys())        # list all keys of modulations

    accuracies = []                                           # start with empty list
    for mod in mod_names:
        acc = class_prediction[mod] / class_samples[mod]      # example 85/100
        accuracies.append(acc)
    
    fig, ax = plt.subplots(figsize=(12, 6))                   # create bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(mod_names)))   # matplotlib set3() color palette
    bars = ax.bar(mod_names, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')                                 # y axis: accuracies
    ax.set_xlabel('Modulation Type')                          # x axis: modulations  
    ax.set_title(f'{model_name} - Accuracy per Modulation')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()                                     # get accuracy height
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)   # center the text label
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/{model_name.replace(" ", "_")}_mod_acc.png', 
                   dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    return accuracies

def plot_snr_accuracy(model, test_loader, model_name, save_path=None):
    """Plot accuracy as a function of SNR.
       evaluation (no calculation code) -> disable dropout
       digitize() -> numpy fct that return index of the bin(box) for a specified value"""
    model.eval()
    snr_bins = np.arange(SNR_MIN, SNR_MAX + 2, 2)      # 2dB-wide bins
    bin_acc = [[] for h in range(len(snr_bins) - 1)]  # initialize with empty list
    with torch.no_grad():
        for X, y, snr in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            k, preds = torch.max(outputs, 1)        
            for label, pred, snr_val in zip(y.cpu().numpy(), preds.cpu().numpy(), snr.numpy()):
                bin_idx = np.digitize(snr_val, snr_bins) - 1           # -2.5 belong to bin 5(between -4 & -2), so index = 4
                if 0 <= bin_idx < len(bin_acc):                        # out of range snr condition
                    bin_acc[bin_idx].append(1 if label == pred else 0) # store 1 if label = prediction, else 0
    snr_centers = (snr_bins[:-1] + snr_bins[1:]) / 2                   # center snr box for plot
    counts = [len(bin_acc[i]) for i in range(len(bin_acc))]

    # only plot bins that actually have data
    plot_centers = [snr_centers[i] for i in range(len(bin_acc)) if counts[i] > 0]
    plot_accuracies = [np.mean(bin_acc[i]) for i in range(len(bin_acc)) if counts[i] > 0]
    plot_counts = [counts[i] for i in range(len(bin_acc)) if counts[i] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_centers, plot_accuracies, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} - Accuracy vs SNR')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(SNR_MIN - 2, SNR_MAX + 2)
    
    # Add sample count annotations
    for center, acc, count in zip(plot_centers, plot_accuracies, plot_counts):
        ax.annotate(f'n={count}', (center, acc), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=8, alpha=0.7)
    
    # Save if path provided
    if save_path:
        plt.savefig(f'{save_path}/{model_name.replace(" ", "_")}_snr_acc.png', 
                   dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    return fig, snr_centers, plot_accuracies

def compare_models_results(results_dict, save_path = None):
    model_names = list(results_dict.keys())
    val_acc = [results_dict[name]['val_acc'] for name in model_names]
    test_acc = [results_dict[name]['test_acc'] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    barr1 = ax.bar(x - width/2, val_acc, width, label='Validation Accuracy', color='#2ca02c', alpha=0.8)
    barr2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison: Validation vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha = 0.3, axis = 'y')
    ax.set_ylim(0, 1.05)
    for bars in [barr1, barr2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    return fig

def plot_channel_impairments(distance = 2.0, snr_db = 10, n_samples = 500, save_path = None):
    stages = ['Clean', 'Path Loss', 'LED Non-Lin', 'Bandwidth', 'Jitter', 'AWGN']
    n_mods = len(MODULATIONS)
    n_stages = len(stages)

    fig, axes = plt.subplots(n_mods, n_stages, figsize=(3.5 * n_stages, 2.5 * n_mods), sharex='col')
    for row, mod_name in enumerate(MODULATIONS):
        clean_signal = MODULATION_GEN[mod_name]()
        signals = [clean_signal]
        signal = apply_path_loss(clean_signal.copy(), distance)
        signals.append(signal)
        signal = apply_led_nonlinearity(signal)
        signals.append(signal)
        signal = apply_bandwidth_limit(signal)
        signals.append(signal)
        signal = apply_clock_jitter(signal)
        signals.append(signal)
        signal = apply_awgn(signal, snr_db)
        signals.append(signal)
        for col, (stage, sig) in enumerate(zip(stages, signals)):
            ax = axes[row, col]
            ax.plot(sig[:n_samples], linewidth=0.8, color='#1f77b4')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.3, 1.3)
            if row == 0:
                ax.set_title(stage, fontsize=11)
            if col == 0:
                ax.set_ylabel(mod_name, fontsize=10, rotation=0, ha='right', va='center')
            if row == n_mods - 1:
                ax.set_xlabel('Sample Index', fontsize=8)
    # super (main) title
    fig.suptitle(f'Channel Impairment for all Modulations (distance={distance}m, SNR={snr_db}dB)', fontsize=12, y=1.005)
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/channel_impairments_all_modulations.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()
    return fig

def path_loss_analysis():
    print(f"Path Loss Exponent n = {PL_EXPONENT}")
    print(f"{'Distance':<10} {'Original Power':<18} {'Scale (1/d^n)':<18} {'Attenuation (dB)':<20} {'Remaining Power':<18}")
    print("-" * 80)
    sample = MODULATION_GEN['OOK']()        # ook for quick test
    original_power = np.mean(sample ** 2)   # multiple samples, P = X(t)^2 / N 
    for d in DISTANCES:
        scale = 1.0 / (d ** PL_EXPONENT)
        atten_db = 20 * np.log10(scale) if scale > 0 else float('-infinity')
        remaining = scale * 100
        print(f"{d:<10.1f} {original_power:<18.2f} {scale:<18.2f} {atten_db:<+20.2f} {remaining:<.1f}%")
        # < left align, 10 is column width, .1f is 1 decimal
    
def measure_inference_time(model, test_loader, num_batches=10):
    model.eval()
    times = []
    batch_size = test_loader.batch_size or 1
    with torch.no_grad():
        for i, (X, y, k) in enumerate(test_loader):
            if i >= num_batches:
                break
            X = X.to(DEVICE)
            if i == 0:
                k = model(X)
                if DEVICE.type == "xpu":
                    torch.xpu.synchronize()
            start = time.time()
            k = model(X)
            if DEVICE.type == "xpu":
                torch.xpu.synchronize()            
            end = time.time()
            times.append((end - start) * 1000)  # convert to ms
    avg_time_ms = float(np.mean(times))
    time_per_sample_ms = avg_time_ms / batch_size
    return avg_time_ms, time_per_sample_ms

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}min ({seconds:.0f}s)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h ({seconds:.0f}s)"
    
def save_results_to_json(results, filename='results/record.json'):    # save model results & time to record
    def convert(obj):
        if hasattr(obj, 'item'):
            return float(obj.item())
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    json_results = convert(results)
    for name, data in results.items():
        json_results[name] = {
            'val_acc': float(data['val_acc']),
            'test_acc': float(data['test_acc']),
            'test_loss': float(data['test_loss']),
            'params': data.get('params', 0),
            'train_time_seconds': float(data.get('train_time', 0)),
            'inference_time_ms': float(data.get('inference_time_ms', 0)),
            'time_per_sample_ms': float(data.get('time_per_sample_ms', 0)),
            'history': {
                'train_loss': [float(x) for x in data['history']['train_loss']],
                'train_acc': [float(x) for x in data['history']['train_acc']],
                'val_loss': [float(x) for x in data['history']['val_loss']],
                'val_acc': [float(x) for x in data['history']['val_acc']],
            }
        }
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Results saved to {filename}")