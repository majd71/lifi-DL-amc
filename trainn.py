import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from configg import EPOCHS, PATIENCE, GRAD_CLIP, LEARNING_RATE_MAX, LEARNING_RATE_MIN, WEIGHT_DECAY, LABEL_SMOOTHING, DEVICE


class LabelSmoothingCrossEntropy(nn.Module):
    """.size(-1) -> checks the list from right to left and pick the last value -> 8
       torch.zeros_like() -> return a tensor/list of zeros same dimentions as inputs
       .scatter(dimension, index, src) -> add src (1) of dimension (1) at index (labels.unsqeeze)
       .log_softmaxn(input, dim) -> apply log softmax on logits across the dimension 1 (across the 8 classes)"""
    
    def __init__(self, smoothing = LABEL_SMOOTHING):    # 0.1 from configg
        super().__init__()
        self.smoothing = smoothing                      # storing the value
    
    def forward(self, logits, labels):      # logits -> output before softmax, labels -> modulation class
        n_classes = logits.size(-1)         
        one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)           # 1 or 100% probability
        smooth_targets = one_hot * (1 - self.smoothing) + self.smoothing / n_classes    # 1-smoothing -> no overfitting
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -torch.sum(smooth_targets * log_probs, dim=1).mean()    
        return loss            # return positive value, to know how much we lost


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch in for loop. device -> CPU or GPU
       model.train()  ->  dropout: on, model.eval() ->  dropout: off
       for ... in tqdm(range(int)) -> show progress
       optimizer.zero_grad() -> reset the gradients (loss/weight) to 0"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X, y, snr in tqdm(loader, desc="Training", leave=False):  # show progress bar labeled "training"
        X, y = X.to(device), y.to(device)                         # moves traingg signals to GPU
        optimizer.zero_grad()
        outputs = model(X)                                        # calls CNN models to make predicted outputs
        loss = criterion(outputs, y)                              # compare predictions to true y(modulation)
        loss.backward()                                           # compute loss parameter, back to weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)  # cap gradient magnitude, prevents loss spikes
        optimizer.step()                                          # update model weight
        total_loss += loss.item() * X.size(0)                     # accumulate loss
        snr, prediction = torch.max(outputs, 1)                   # get prediction
        correct += (prediction == y).sum().item()                 # count how many are correct
        total += y.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()    # evaluate mode -> no dropout
    total_loss = 0
    correct = 0
    total = 0   
    with torch.no_grad():     # disable gradients
        for X, y, snr in tqdm(loader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y) 
            total_loss += loss.item() * X.size(0)
            snr, prediction = torch.max(outputs, 1)
            correct += (prediction == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def train_model(model, train_loader, val_loader, model_name):
    """Train model with AdamW optimizer and Cosine Annealing scheduler.
       CrossEntropyLoss() -> computes the cross entropy loss between input logits and target. """
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing = LABEL_SMOOTHING) # entropy loss with 0.1 label smoothing
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr = LEARNING_RATE_MAX,                    # 0.001 from configg
        weight_decay = WEIGHT_DECAY                # 0.01 from configg
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = EPOCHS,                           # 100
        eta_min = LEARNING_RATE_MIN               # 1e-6
    )
    
    best_val_acc = 0.0                            # store and monitor the values
    patience_counter = 0                          # stop if no improvements for 40 epochs
    history = {'train_loss': [], 'train_acc': [],'val_loss': [], 'val_acc': [],'lr': []}
    
    
    for epoch in range(1, EPOCHS + 1):      # range(epochs) -> stops at 99, so range(epochs+1)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE) #tain
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)                         # validate
        
        current_lr = scheduler.get_last_lr()[0]  # get_last_lr(): built in fct that return a list of recent lr
        scheduler.step()                         # update the learning rate
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)             # add the values
        
        if val_acc > best_val_acc:                   # save the best model weights
            best_val_acc = val_acc
            patience_counter = 0                     # reset
            torch.save(model.state_dict(), f'fyp/models/best_{model_name.replace(" ", "_")}.pth')
        else:
            patience_counter +=1          # no improvement this epoch
        if patience_counter >= PATIENCE:
            print(f"\n Early stopping at epoch {epoch} -> no improvement for {PATIENCE} epochs")
            break 
        
        if epoch % 10 == 0 or epoch == 1:            # print every 10 epochs, and the 1st epoch
            print(f"Epoch {epoch:3d}/{EPOCHS} | LR: {current_lr:.2e} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return history, best_val_acc