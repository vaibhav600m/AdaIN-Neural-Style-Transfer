🎨 AdaIN Neural Style Transfer with Flask Web UI
An end-to-end implementation of Arbitrary Style Transfer using Adaptive Instance Normalization (AdaIN). This repository contains the complete PyTorch implementation, training scripts, single/batch inference scripts, and a modern Flask-based Web Interface.

Key Features
Real-time Arbitrary Style Transfer: Transfer any painting style onto any content image using AdaIN.
Apple Silicon (MPS) & CUDA Support: Auto-device detection (CUDA → MPS → CPU).
Custom Decoder Architecture: VGG-19 encoder as a fixed feature extractor with a custom-trained decoder.
Adjustable Style Strength (α): Dynamic control over style intensity (0.0≤α≤1.0).
Modern Web Interface: Built with Flask, Bootstrap 5, and interactive canvas visualizer.

### ⚙️ Training Details & Hyperparameters

* **Total Epochs:** 240
* **Batch Size:** 8
* **Learning Rate:** 1e-4 (with decay $5 \times 10^{-5}$)
* **Total Variation (TV) Loss Weight ($\lambda_{tv}$):** $1 \times 10^{-5}$

#### 🔄 Two-Phase Loss Configuration

| Hyperparameter | Phase 1 (Epochs 1–180) | Phase 2 (Epochs 181–240) |
| :--- | :--- | :--- |
| **Content Weight ($\lambda_c$)** | 1.0 | 1.0 |
| **Style Weight ($\lambda_s$)** | **10.0** *(High-Intensity Style Extraction)* | **1.0** *(Balanced Fine-Tuning)* |


⚙️ 📈 Convergence Summary Across Phases
Phase 1: High Style Weight (λs = 10.0, λc = 1.0)
Epoch 1/240   - Loss: 48.2100
Epoch 90/240  - Loss: 18.3120
Epoch 180/240 - Loss: 8.1045

Phase 2: Balanced Fine-Tuning (λs = 1.0, λc = 1.0)
Epoch 181/240 - Loss: 1.8950
Epoch 210/240 - Loss: 1.4140
Epoch 240/240 - Loss: 1.0421


Final-NST-Code/
├── static/
│   └── uploads/             # Stores uploaded & generated images
├── templates/
│   └── index.html           # Modern Web UI template
├── test_images/
│   ├── content.jpg          # Input content test image
│   └── style.jpg            # Input style test image
├── utils/
│   ├── __init__.py
│   ├── models.py            # VGGEncoder & Decoder architectures
│   └── utils.py             # AdaIN normalization & TV Loss functions
├── weights/
│   ├── vgg_normalised.pth   # Pre-trained VGG-19 Encoder
│   └── decoder_240.pth      # Trained Decoder weights (240 Epochs)
├── app.py                   # Flask Web Server
├── train.py                 # Training script
└── aa.py                    # Local inference script
