# 🧪 Model Training Guide (Multimodal Autism Detection)

---

## 📌 1. Overview

This module contains the **training pipeline** for a multimodal deep learning model designed to estimate autism risk.

The model combines **multiple data modalities** to improve prediction accuracy:

* 🖼️ Visual data (images)
* 🏃 Motion sequences
* ❤️ Physiological signals
* 🧠 Behavioral scores (ADOS)

---

## 🎯 2. Training Objective

The goal is to train a model that can:

* Classify individuals as:

  * Typical (0)
  * Autism (1)
* Learn patterns across multiple modalities
* Improve robustness compared to single-input models

---

## 📁 3. Files in This Folder

```id="trainfiles02"
training/
│
├── train.ipynb        # Main training notebook
└── README.md          # Training documentation
```

---

## 📊 4. Dataset Structure

Place dataset in project root:

```id="datasetstruct02"
dataset/
├── images/                     # Facial images
├── voice/                      # Audio recordings
├── motion/                     # Motion JSON files
├── physio/                     # Physiological CSV files
├── autism_dataset_metadata.json
└── Toddler Autism dataset July 2018.csv
```

---

## 🧾 5. Dataset Description

| Modality | Description                        |
| -------- | ---------------------------------- |
| Images   | Used for visual feature extraction |
| Audio    | Converted to MFCC features         |
| Motion   | Behavioral movement sequences      |
| Physio   | Heart rate, temperature, etc.      |
| ADOS     | Clinical behavioral assessment     |

---

## ⚠️ 6. Dataset Availability

* Dataset is **not included** due to size
* Download from:

  * Kaggle (recommended)
  * Provided dataset source

---

## ⚙️ 7. Environment Setup

### Install dependencies:

```id="traininstall02"
pip install torch torchvision numpy pandas librosa pillow matplotlib scikit-learn tqdm
```

---

## 💻 8. Hardware Recommendation

| Hardware | Recommendation             |
| -------- | -------------------------- |
| CPU      | Works but slow             |
| GPU      | Recommended (Colab / CUDA) |
| RAM      | 8GB+                       |

---

## 🧠 9. Model Architecture (Detailed)

The system uses a **multimodal neural network**:

### 🔹 Image Encoder

* ResNet18 backbone
* Extracts visual features

### 🔹 Audio Encoder

* CNN layers
* Processes MFCC features

### 🔹 Motion Encoder

* LSTM network
* Captures temporal behavior

### 🔹 Physio Encoder

* Fully connected layers
* Uses mean + standard deviation

### 🔹 ADOS Encoder

* Dense neural network
* Encodes behavioral scores

---

### 🔗 Feature Fusion

All extracted features are concatenated:

```text id="fusion01"
[Image + Audio + Motion + Physio + ADOS] → Fusion Layer → Classifier
```

---

## 🔄 10. Training Pipeline

The notebook follows this workflow:

1. Load dataset
2. Preprocess inputs
3. Split into train/test
4. Handle class imbalance (Weighted Sampling)
5. Initialize model
6. Train model (epochs)
7. Evaluate performance
8. Save trained model

---

## 📉 11. Loss Function & Optimizer

* Loss: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: 1e-4

---

## 📊 12. Evaluation Metrics

* Accuracy
* Precision
* Recall
* Confusion Matrix

---

## 🧪 13. Example Training Output

```text id="trainlog01"
Epoch 1/10 | Loss: 0.52 | Acc: 78.4%
Epoch 5/10 | Loss: 0.21 | Acc: 91.2%
Epoch 10/10 | Loss: 0.10 | Acc: 95.6%
```

---

## 💾 14. Model Saving

After training, model is saved as:

```id="modelsave02"
autism_model.pt
```

This file is used by the backend (`app.py`) for predictions.

---

## 🔁 15. How to Re-Train

1. Open notebook
2. Update dataset path if needed:

```python id="trainpath02"
BASE_PATH = "dataset"
```

3. Run all cells
4. New model will overwrite previous `.pt` file

---

## ⚠️ 16. Common Errors & Fixes

### ❌ Dataset not found

✔ Ensure:

```id="fix01"
dataset/ folder exists
```

---

### ❌ Librosa error

✔ Install:

```id="fix02"
pip install librosa
```

---

### ❌ CUDA not available

✔ Use CPU:

```python id="fix03"
DEVICE = "cpu"
```

---

## 🔮 17. Future Improvements

* Add real-time data streaming
* Improve motion feature extraction
* Use transformer-based models
* Increase dataset diversity
* Hyperparameter tuning

---

## 🧠 18. Key Insight

Multimodal learning improves performance because:

* Each modality captures different behavior
* Combined signals reduce prediction errors
* Provides more robust decision-making

---

## 👨‍💻 19. Conclusion

This training pipeline demonstrates a complete **end-to-end multimodal AI system**, from raw data to trained model, ready for real-world deployment.

---
