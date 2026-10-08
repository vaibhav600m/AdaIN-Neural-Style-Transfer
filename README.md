# 🎨 AdaIN Neural Style Transfer with Flask Web UI

An end-to-end implementation of **Arbitrary Neural Style Transfer** using **Adaptive Instance Normalization (AdaIN)**.

This project combines a **pre-trained VGG-19 encoder**, a **custom-trained decoder**, and a modern **Flask web interface** to transfer the visual style of any reference image onto any content image.

---

## ✨ Features

* 🎨 **Arbitrary Style Transfer**
  Transfer the style of any artwork onto any content image.

* ⚡ **Real-Time Inference**
  Generate stylized images directly through the web interface.

* 🧠 **AdaIN-Based Style Transfer**
  Uses Adaptive Instance Normalization to align the feature statistics of content and style images.

* 🔥 **Custom-Trained Decoder**
  A decoder network trained from scratch to reconstruct stylized images from AdaIN-transformed features.

* 🖼️ **VGG-19 Encoder**
  Uses a pre-trained and fixed VGG-19 network for extracting hierarchical image features.

* 🎚️ **Adjustable Style Strength**
  Control the intensity of style transfer using the parameter `α`.

  `0.0 ≤ α ≤ 1.0`

* 💻 **Multi-Device Support**

  ```text
  CUDA → MPS → CPU
  ```

  Automatically selects the best available device.

* 🌐 **Modern Web Interface**
  Built using Flask, Bootstrap 5, HTML/CSS, and an interactive image visualizer.

* 📦 **Single & Batch Inference**
  Supports both quick local testing and batch image processing.

---

# 🧠 How It Works

The pipeline follows the architecture introduced by **AdaIN-based arbitrary style transfer**:

```text
Content Image ──────┐
                    │
                    ▼
               VGG-19 Encoder
                    │
                    ▼
             Content Features
                    │
                    │
Style Image ────────┐
                    │
                    ▼
               VGG-19 Encoder
                    │
                    ▼
              Style Features
                    │
                    ▼
                  AdaIN
                    │
                    ▼
          Stylized Feature Map
                    │
                    ▼
            Custom Decoder
                    │
                    ▼
          Final Stylized Image
```

### 🎚️ Style Strength — α

The parameter `α` controls how strongly the style is applied:

```text
α = 0.0  → Mostly original content
α = 0.5  → Balanced style transfer
α = 1.0  → Maximum style transfer
```

The final AdaIN feature representation is interpolated as:

```text
t = α × AdaIN(content, style)
    + (1 - α) × content
```

---

# 🏋️ Decoder Training

The custom decoder was trained from scratch on **Kaggle GPU infrastructure** using PyTorch and the Adam optimizer.

### Training Configuration

| Parameter           |                    Value |
| ------------------- | -----------------------: |
| Total Epochs        |                  **240** |
| Batch Size          |                    **8** |
| Optimizer           |                 **Adam** |
| Learning Rate       |             **1 × 10⁻⁴** |
| Learning Rate Decay |             **5 × 10⁻⁵** |
| TV Loss Weight      |             **1 × 10⁻⁵** |
| Encoder             |   **Pre-trained VGG-19** |
| Decoder             | **Trained from scratch** |

---

## 🔄 Two-Phase Training Strategy

Training was divided into two phases to first emphasize strong style reconstruction and then perform balanced fine-tuning.

| Hyperparameter      | Phase 1<br>Epochs 1–180 | Phase 2<br>Epochs 181–240 |
| ------------------- | ----------------------: | ------------------------: |
| Content Weight `λc` |                     1.0 |                       1.0 |
| Style Weight `λs`   |                **10.0** |                   **1.0** |

### Phase 1 — Strong Style Learning

```text
λc = 1.0
λs = 10.0
```

A higher style-loss weight encourages the decoder to learn strong reconstruction of stylized features.

### Phase 2 — Balanced Fine-Tuning

```text
λc = 1.0
λs = 1.0
```

The style weight is reduced to balance content preservation and style reconstruction during the final training stage.

---

# 📈 Training Progress

The training loss decreased substantially throughout the two phases:

```text
Phase 1 — High Style Weight
λs = 10.0, λc = 1.0

Epoch   1   → Loss: 48.2100
Epoch  90   → Loss: 18.3120
Epoch 180   → Loss:  8.1045


Phase 2 — Balanced Fine-Tuning
λs = 1.0, λc = 1.0

Epoch 181   → Loss: 1.8950
Epoch 210   → Loss: 1.4140
Epoch 240   → Loss: 1.0421
```

### 📊 Overall Loss Reduction

```text
48.2100  ───────────────────────────────►  1.0421
Epoch 1                                      Epoch 240
```

This substantial decrease in training loss indicates that the decoder progressively improved its reconstruction objective during training.

---

# 🏗️ Project Structure

```text
Final-NST-Code/
│
├── .gitignore
│   └── Excludes virtual environments and temporary files
│
├── requirements.txt
│   └── Required Python dependencies
│
├── app.py
│   └── Flask web application
│
├── train.py
│   └── Decoder training script
│
├── aa.py
│   └── Local inference/testing script
│
├── static/
│   └── uploads/
│       └── Uploaded and generated images
│
├── templates/
│   └── index.html
│       └── Flask web interface
│
├── test_images/
│   ├── content.jpg
│   │   └── Sample content image
│   │
│   └── style.jpg
│       └── Sample style reference image
│
├── utils/
│   ├── __init__.py
│   │
│   ├── models.py
│   │   └── VGG-19 Encoder and Decoder definitions
│   │
│   └── utils.py
│       └── AdaIN and Total Variation loss utilities
│
└── weights/
    ├── vgg_normalised.pth
    │   └── Pre-trained VGG-19 weights
    │
    └── decoder_240.pth
        └── Decoder weights trained for 240 epochs
```

---

# 🛠️ Tech Stack

| Category      | Technology                             |
| ------------- | -------------------------------------- |
| Language      | **Python**                             |
| Deep Learning | **PyTorch**                            |
| Architecture  | **AdaIN + VGG-19 + Custom Decoder**    |
| Web Framework | **Flask**                              |
| Frontend      | **HTML, CSS, Bootstrap 5, JavaScript** |
| Training      | **Kaggle GPU**                         |
| GPU Support   | **CUDA / Apple MPS**                   |
| CPU Support   | **Yes**                                |

---

# 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Final-NST-Code
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Flask Web App

Start the application:

```bash
python app.py
```

Then open the local server shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

Upload:

1. 🖼️ Content image
2. 🎨 Style image
3. 🎚️ Select style strength `α`

The application will generate the stylized output.

---

# 🧪 Local Inference

For quick testing without the Flask interface:

```bash
python aa.py
```

The sample images inside:

```text
test_images/
```

can be used for testing the model.

---

# 📦 Pre-Trained Weights

The repository uses two sets of model weights:

### VGG-19 Encoder

```text
weights/vgg_normalised.pth
```

A pre-trained VGG-19 network is used as a **fixed feature extractor**.

### Custom Decoder

```text
weights/decoder_240.pth
```

This is the custom decoder trained for **240 epochs**.

---

# 🎯 Project Goal

The goal of this project is to build a complete, practical implementation of **arbitrary neural style transfer**, covering the entire pipeline:

```text
Research Paper
      ↓
AdaIN Implementation
      ↓
VGG-19 Feature Extraction
      ↓
Decoder Training
      ↓
Model Evaluation
      ↓
Inference Pipeline
      ↓
Flask Web Application
```

Rather than relying only on a pre-built style-transfer model, the project includes the **training pipeline for the custom decoder** and integrates the trained model into a usable web application.

---

# 👨‍💻 Author

**Vaibhav Mathur**

B.S.-M.S. Computer Science & Data Analytics
IIT Patna

---

⭐ If you found this project useful, consider giving the repository a star!
