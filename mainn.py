import os
import torch
import numpy as np
import time
from configg import DEVICE, MODULATIONS, EPOCHS, LABEL_SMOOTHING, LEARNING_RATE_MAX, LEARNING_RATE_MIN, WEIGHT_DECAY
from signall import generate_dataset
from dattaset import get_dataloaders
from modeels import get_model
from trainn import train_model, evaluate, LabelSmoothingCrossEntropy
from resultss import (plot_training_history, plot_confusion_matrix, plot_modulation_accuracy, plot_snr_accuracy,
                      path_loss_analysis, compare_models_results, plot_channel_impairments, measure_inference_time,
                      format_time, save_results_to_json)

def main():
    # start the timer
    total_start_time = time.time()

    print("="*70)
    print("   CNN-Based Modulation Prediction in Li-Fi Systems")
    print("   Intel Arc GPU: " + ("Enabled" if DEVICE.type == "xpu" else "Not Used"))
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # make directories
    os.makedirs('fyp/models', exist_ok = True)
    os.makedirs('fyp/results', exist_ok = True)
    os.makedirs('fyp/results/figures', exist_ok = True)

    print("\n[1/6] Generating dataset...")
    data_start = time.time()
    X, y, snr = generate_dataset()
    data_time = time.time() - data_start
    print(f"Dataset: {X.shape[0]} signals, {X.shape[1]} samples each")
    print(f"Classes: {len(np.unique(y))} modulations")
    print(f"SNR range: {snr.min():.1f} dB to {snr.max():.1f} dB")
    path_loss_analysis()
    print(f"Dataset generated in {format_time(data_time)}")
    print("\n[1.5/6] Visualizing channel impairments...")
    plot_channel_impairments(distance = 2.0, snr_db = 10, save_path = 'fyp/results/figures')
    print("Channel impairment visualization saved")
    
    print("\n[1.6/6] Dataset diagnostic (if the mean/std are too similar, the CNN can't distinguish them)")
    for i, mod in enumerate(MODULATIONS):
        mask = y == i    # mask -> boolean array, mask.sum() -> count how many true
        print(f"{mod:10s}: {mask.sum():4d} samples | "
              f"mean={X[mask].mean():.4f} | std={X[mask].std():.4f} | "
              f"min={X[mask].min():.4f} | max={X[mask].max():.4f}")

    print("\n[2/6] Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(X, y, snr)
    print(f"Training:   {len(train_loader.dataset):,} signals")
    print(f"Validation: {len(val_loader.dataset):,} signals")
    print(f"Test:       {len(test_loader.dataset):,} signals")

    model_names = ['Lightweight CNN', '1D CNN', 'Robust CNN']   
    print(f"\n[3/6] Models to train: {', '.join(model_names)}")
    results = {}
    for model_name in model_names:
        model = get_model(model_name)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"Device: {DEVICE}, Epochs: {EPOCHS}")
        print(f"Optimizer: AdamW, Weight Decay: {WEIGHT_DECAY}")
        print(f"Scheduler: Cosine Annealing (LR: {LEARNING_RATE_MAX} to {LEARNING_RATE_MIN})")
        print(f"Loss: Label Smoothing (α={LABEL_SMOOTHING})")
        print(f"Model parameters: {total_params:,}")
        print(f"{'='*60}")
            
        # Train & Time
        train_start = time.time()
        history, best_val_acc = train_model(model, train_loader, val_loader, model_name)
        train_time = time.time() - train_start
        # Load best model
        model_path = f'fyp/models/best_{model_name.replace(" ", "_")}.pth'
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        # Evaluate & Time inference
        criterion = LabelSmoothingCrossEntropy()
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        inference_time_ms, time_per_sample_ms = measure_inference_time(model, test_loader)
        # Store results
        results[model_name] = {
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'history': history,
            'params': total_params,
            'train_time': train_time,
            'inference_time_ms': inference_time_ms,
            'time_per_sample_ms': time_per_sample_ms
        }
        print(f"\n {model_name} Results:")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Test Accuracy:            {test_acc:.4f}")
        print(f"Test Loss:                {test_loss:.4f}")
        print(f"Parameters:               {total_params:,}")
        print(f"Training time:            {format_time(train_time)}")
        print(f"Inference time:           {inference_time_ms:.2f} ms/batch")
        print(f"Per sample:               {time_per_sample_ms:.4f} ms") 
        
        print(f"\n Generating visualizations for {model_name}...")
        plot_training_history(
            history, 
            model_name, 
            save_path = 'fyp/results/figures'
        )
        plot_confusion_matrix(
            model, 
            test_loader, 
            model_name, 
            save_path = 'fyp/results/figures'
        )
        plot_modulation_accuracy(
            model, 
            test_loader, 
            model_name, 
            save_path = 'fyp/results/figures'
        )
        plot_snr_accuracy(
            model, 
            test_loader, 
            model_name, 
            save_path = 'fyp/results/figures'
        )

    
    print("\n[6/6] Model Comparison")
    print('='*70)
    compare_models_results(results, save_path = 'fyp/results/figures')
    print("\n   Saving results to record...")
    save_results_to_json(results, 'fyp/results/record.json')

    total_time = time.time() - total_start_time
    sorted_models = sorted(results.items(), key = lambda x: x[1]['test_acc'], reverse = True)
    
    print("\n" + "="*70)
    print("    Model Ranking")
    print("="*70)
    print(f"\n{'Rank':<6} {'Model':<20} {'Test Acc':<12} {'Train Time':<14} {'Inference (ms)':<15}")
    print("-"*70)    
    for rank, (name, data) in enumerate(sorted_models, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<4} {name:<20} {data['test_acc']:.4f}       "
              f"{format_time(data['train_time']):<14} {data['inference_time_ms']:.2f}")
    # Total time
    print("\n" + "="*70)
    print(f"TOTAL PIPELINE TIME: {format_time(total_time)}")
    print("="*70)

if __name__ == '__main__':
    main()