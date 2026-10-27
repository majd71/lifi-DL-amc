# CNN-Based Modulation Prediction in Li-Fi Systems

**Author:** Mr. Majd al Cheikh  
**Supervisor:** Dr. Anas El Husseini  
**Institution:** City University — Department of Communication and Network Engineering  
**Course:** FYP 596 Final Year Project II (Fall 2026)

## 📌 Project Overview
This repository contains the PyTorch implementation for Automatic Modulation Classification (AMC) in Visible Light Communication (Li-Fi) systems. Three 1D Convolutional Neural Networks (CNNs) were designed and evaluated to classify 8 modulation schemes under realistic optical channel impairments (LED non-linearity, bandwidth limitation, path loss, and AWGN).

## 🧠 Proposed Architectures
1. **Standard 1D CNN** (143K parameters)
2. **Lightweight CNN** (11.5K parameters)
3. **Multi-Scale Robust CNN** (302K parameters)

## 📂 Directory Structure
- `configg.py`: Global parameters and device detection (Intel Arc GPU).
- `signall.py`: Signal generation (8 modulations) and channel impairment modeling.
- `dattaset.py`: PyTorch Dataset class, Z-score normalization, and DataLoaders.
- `modeels.py`: Definitions for the three CNN architectures.
- `trainn.py`: Training loop with AdamW, Label Smoothing, and Early Stopping.
- `resultss.py`: Functions for plotting confusion matrices, SNR curves, and model comparisons.
- `mainn.py`: Main orchestration script to generate data, train, and evaluate.
- `report.pdf`: The FYP report

## 🚀 How to Run
1. Install dependencies the libraries in `1req.txt`
2. Rum `mainn.py` to get the results

for more information, please contact: majd75connect@proton.me
