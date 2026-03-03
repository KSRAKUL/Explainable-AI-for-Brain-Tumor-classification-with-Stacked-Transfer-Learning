<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![XAI](https://img.shields.io/badge/XAI-Grad--CAM-brightgreen?style=for-the-badge)
![GUI](https://img.shields.io/badge/GUI-Tkinter-blueviolet?style=for-the-badge)

<br/>

# 🧠 Explainable AI for Brain Tumor Classification with Stacked Transfer Learning

> **A deep learning framework combining Stacked Transfer Learning and Explainable AI (XAI) for accurate, interpretable, and clinically trustworthy brain tumor classification from MRI scans — with a full-featured desktop GUI.**

[![GitHub stars](https://img.shields.io/github/stars/KSRAKUL/Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning?style=social)](https://github.com/KSRAKUL/Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning)
[![GitHub forks](https://img.shields.io/github/forks/KSRAKUL/Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning?style=social)](https://github.com/KSRAKUL/Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning/forks)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-KSRAKUL-0077B5?style=social&logo=linkedin)](https://www.linkedin.com/in/ksrakul)

</div>

---


## 🔍 Overview

Brain tumors are among the most life-threatening neurological conditions, and early, accurate classification is critical to effective treatment planning. This project proposes a **Stacked Transfer Learning** approach powered by **Explainable Artificial Intelligence (XAI)** to classify brain tumors from MRI images with both high accuracy and clinical transparency.

The model combines EfficientNetB0, MobileNetV3Small, and NASNetMobile in a stacked architecture, interpreted through real-time **Grad-CAM heatmaps**. A fully functional **Brain Tumor Detection desktop application** enables clinicians to load MRI scans, receive instant AI diagnoses with confidence scores, and view spatial attention heatmaps — all in one interface.

---

## Desktop Application View

> Real application output — diagnosing a **Glioma** with **87.5% confidence** and displaying the Grad-CAM heatmap in real time.

<div align="center"> Brain Tumor Detection - AI Analysis System</div>
<img width="2880" height="1690" alt="Screenshot 2025-12-04 105509" src="https://github.com/user-attachments/assets/40eb9e5e-5dff-488e-82d1-be21cecb0877" />


### Application Panels

| Panel | Description |
|-------|-------------|
| 🎛️ **Control Panel** | Load MRI scan · Analyze Tumor · Save Results · Model status |
| 🔬 **Image Analysis** | Original Brain Scan viewer + live Grad-CAM heatmap overlay |
| 🤖 **AI Explanation** | Diagnosis · Confidence · Symptoms · Severity · Treatment info |

### Application Capabilities
- ✅ **MRI Validation** — Verifies the uploaded image is a valid brain MRI
- ✅ **Real-Time Grad-CAM** — RED (high focus) → YELLOW → BLUE (low focus)
- ✅ **Clinical Context** — Auto-generated symptoms, severity, treatment options
- ✅ **Confidence Score** — Per-prediction probability (e.g. 87.5%)
- ✅ **Save Results** — Export diagnosis report for records

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔁 **Stacked Transfer Learning** | EfficientNetB0 + MobileNetV3Small + NASNetMobile unified |
| 🧬 **Hybrid Architecture** | Three pre-trained backbones stacked into one classifier |
| 🔍 **Explainability (XAI)** | Real-time Grad-CAM heatmaps showing tumor focus regions |
| 🏷️ **Multi-Class Classification** | Glioma · Meningioma · No Tumor · Pituitary |
| 🖥️ **Desktop GUI** | Full Tkinter application for clinical & research use |
| 📊 **Comparative Evaluation** | Benchmarked against all three individual backbone models |

---

## 🏗️ Architecture & Methodology

### Stacked Transfer Learning Strategy
```
┌──────────────────────────────────────────────────────────────┐
│               STACKED TRANSFER LEARNING STRATEGY             │
│                                                              │
│  EfficientNetB0      MobileNetV3Small       NASNetMobile     │
│  (ImageNet)          (ImageNet)             (ImageNet)       │
│       │                    │                    │            │
│       ▼                    ▼                    ▼            │
│  Feature Maps         Feature Maps         Feature Maps      │
│       └──────────┬─────────┘                    │            │
│                  │                              │            │
│             Concatenate ◄────────────────────────            │
│                  │                                           │
│             Dense (512, ReLU) → Dropout (0.5)                │
│             Dense (256, ReLU) → Dropout (0.3)                │
│             Dense (4, Softmax)                               │
│                  │                                           │
│     [Glioma | Meningioma | No Tumor | Pituitary]             │
└──────────────────────────────────────────────────────────────┘
```

### Individual vs Stacked Model Comparison

| Model | Val. Accuracy | Val. Loss | Stability |
|-------|---------------|-----------|-----------|
| EfficientNetB0 | ~39% | ~1.55 | ❌ Unstable |
| MobileNetV3Small | ~28% | ~2.35 | ❌ Diverging |
| NASNetMobile | ~80% | ~0.58 | ⚠️ Oscillating |
| **Stacked Model** | **~87%** | **~0.35** | **✅ Stable** |

---

## 🔄 System Pipeline
```
┌──────────┬────────────┬──────────────────────┬──────────────┬──────────────────┐
│  INPUT   │  PREPROCESS│   MODEL              │  TRAINING    │  OUTPUT + GUI    │
│ MRI Scan │ • Resize   │ EfficientNetB0   ─┐  │ • Adam Optim │ • Diagnosis      │
│ Images   │ • Normalize│ MobileNetV3Small ─┼► │ • CE Loss    │ • Confidence %   │
│(JPEG/PNG)│ • Augment  │ NASNetMobile     ─┘  │ • LR Sched.  │ • Grad-CAM       │
│          │ 80/10/10   │  → Stacked Head      │ • Callbacks  │ • Desktop App    │
└──────────┴────────────┴──────────────────────┴──────────────┴──────────────────┘
```

---

## 📁 Project Structure
```
Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning/
│
├── 📁 MODEL/                      # Saved model weights & checkpoints
│   ├── final_model.keras           # Final stacked model (loaded by GUI)
│   ├── stacked_model.h5            # Stacked model weights
│   ├── efficientnet_stage.h5       # EfficientNetB0 stage weights
│   ├── mobilenet_stage.h5          # MobileNetV3Small stage weights
│   └── nasnet_stage.h5             # NASNetMobile stage weights
│
├── 📁 __pycache__/                # Python bytecode cache (auto-generated)
│
├── 📄 train.py                    # Training script
│                                    ├── Data loading & augmentation
│                                    ├── Stacked model construction
│                                    ├── Transfer learning pipeline
│                                    └── Training loop with callbacks
│
├── 📄 validation.py               # Evaluation, XAI & GUI script
│                                    ├── Model loading & inference
│                                    ├── Confusion matrix generation
│                                    ├── Real-time Grad-CAM heatmaps
│                                    └── Tkinter desktop application
│
└── 📄 README.md                   # Project documentation
```

---

## 📊 Dataset

| Class | Description | Label | Test Samples |
|-------|-------------|-------|-------------|
| 🔴 Glioma | Malignant brain tumor from glial cells | `0` | 300 |
| 🟡 Meningioma | Tumor from the meninges (usually benign) | `1` | 306 |
| 🟢 No Tumor | Healthy brain scan | `2` | 405 |
| 🔵 Pituitary | Tumor in the pituitary gland | `3` | 300 |

**Data Split:** 80% Training · 10% Validation · 10% Test  
**Preprocessing:** Resize 224×224 · Normalize [0,1] · Augmentation · Stratified split

> 📥 **Dataset:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## ⚙️ Installation & Setup
```bash
# 1. Clone the repository
git clone https://github.com/KSRAKUL/Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning.git
cd Explainable-AI-for-Brain-Tumor-classification-with-Stacked-Transfer-Learning

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```txt
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.23.0
opencv-python>=4.6.0
matplotlib>=3.6.0
scikit-learn>=1.1.0
shap>=0.41.0
seaborn>=0.12.0
Pillow>=9.2.0
tqdm>=4.64.0
```

> `tkinter` ships with standard Python — no separate install needed.

---

## 🚀 Usage
```bash
# Train the stacked model
python train.py

# Launch the desktop GUI application
python validation.py
```

**Inside the GUI:**
1. Click **Load Brain MRI Scan** → select a `.jpg` / `.png` MRI
2. Click **Analyze Tumor** → inference + Grad-CAM heatmap generated
3. View **Diagnosis**, **Confidence**, and **AI Explanation** in the right panel
4. Click **Save Results** → export the analysis report

---

## 📈 Model Performance

### Validation Accuracy — All Models Compared

[Model Comparison - Validation Accuracy]
<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/1810ac98-41c4-4b04-ba73-7776fdbbaf5a" />


> The **Stacked Model** reaches **~87% validation accuracy** at epoch 10 with smooth convergence. NASNetMobile peaks at ~80% but oscillates. EfficientNetB0 and MobileNetV3Small plateau below 40%.

### Validation Loss — All Models Compared

[Model Comparison - Validation Loss]
<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/dd36aba9-abec-4bfe-b465-fc7344efd180" />


> The **Stacked Model** achieves the lowest and most stable validation loss (~0.35). MobileNetV3Small diverges past 3.2 and EfficientNetB0 fluctuates — confirming stacking's superiority.

### Confusion Matrix

![Confusion Matrix]
<img width="640" height="480" alt="Figure_3" src="https://github.com/user-attachments/assets/f968da36-73cb-4a02-9fb6-f74803b6424f" />


| True Class | Correct | Misclassified | Total | Accuracy |
|------------|---------|---------------|-------|----------|
| Glioma | 276 | 24 | 300 | 92.0% |
| Meningioma | 265 | 41 | 306 | 86.6% |
| No Tumor | 393 | 12 | 405 | 97.0% |
| Pituitary | 299 | 1 | 300 | **99.7%** |

---

## 🔬 XAI Visualizations

### Real Application Output — Glioma Detected (87.5% Confidence)


### Grad-CAM Color Key
```
┌─────────────────────────────────────────────────────┐
│               GRAD-CAM HEATMAP GUIDE                │
│   🔴  RED     →  High AI Attention (tumor focus)    │
│   🟡  YELLOW  →  Medium AI Attention                │
│   🔵  BLUE    →  Low Attention (healthy tissue)     │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Results Summary

| Metric | Value |
|--------|-------|
| 🎯 **Best Validation Accuracy** | ~87% (Stacked Model) |
| 📉 **Best Validation Loss** | ~0.35 (Stacked Model) |
| 🏆 **Best Per-Class Accuracy** | Pituitary — 99.7% |
| 💡 **Live Demo Confidence** | 87.5% (Glioma detection) |
| 🔁 **Training Strategy** | Stacked Transfer Learning |
| 🏗️ **Backbone Models** | EfficientNetB0 + MobileNetV3Small + NASNetMobile |
| 🔍 **XAI Method** | Grad-CAM (real-time in GUI) |
| 🖥️ **Deployment** | Tkinter Desktop Application |
| 🧮 **Input Size** | 224 × 224 × 3 |
| 📦 **Output Classes** | 4 |
| 📊 **Total Test Samples** | 1,311 |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow, Keras |
| **Backbone Models** | EfficientNetB0, MobileNetV3Small, NASNetMobile |
| **XAI** | Grad-CAM |
| **GUI Framework** | Tkinter |
| **Data Processing** | NumPy, OpenCV, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-learn |


## 🤝 Contributing

1. **Fork** the repository
2. **Create** a branch: `git checkout -b feature/your-feature`
3. **Commit**: `git commit -m 'Add feature'`
4. **Push**: `git push origin feature/your-feature`
5. **Open** a Pull Request

---

## 👨‍💻 Author

<div align="center">

**KSRAKUL**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ksrakul)
[![GitHub](https://img.shields.io/badge/GitHub-KSRAKUL-181717?style=for-the-badge&logo=github)](https://github.com/KSRAKUL)

*If this project helped you, please ⭐ star the repository on GitHub!*

---

**Made with ❤️ for advancing AI in Medical Imaging**

`Deep Learning` · `Explainable AI` · `Transfer Learning` · `Brain MRI` · `Healthcare AI` · `Desktop Application`

</div>
