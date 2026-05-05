# 🧠 Autism Screening AI Dashboard

---

## 📌 1. Introduction

Autism Spectrum Disorder (ASD) is a developmental condition that affects communication, behavior, and social interaction. Early detection plays a crucial role in improving outcomes through timely intervention.

This project presents a **multimodal AI-based autism screening system** that integrates:

* Visual data (facial image)
* Behavioral assessment (ADOS questionnaire)
* Physiological signals
* Eye contact analysis (via webcam)

The system provides a **risk-level prediction (Low / Moderate / High)** along with a probability score.

---

## 🎯 2. Objectives

* Develop an AI system for early autism risk screening
* Combine multiple modalities for improved prediction
* Provide an interactive and user-friendly interface
* Demonstrate real-time AI inference using a web-based dashboard

---

## 💻 3. Software Used

| Category                | Tool                  |
| ----------------------- | --------------------- |
| Programming Language    | Python 3              |
| Backend Framework       | FastAPI               |
| Machine Learning        | PyTorch               |
| Computer Vision         | OpenCV                |
| Frontend                | HTML, CSS, JavaScript |
| Server                  | Uvicorn               |
| Development Environment | VS Code               |

---

## 📦 4. Libraries Used

```id="2bt8zj"
fastapi
uvicorn
torch
torchvision
numpy
pillow
opencv-python
python-multipart
```

---

## 🏗️ 5. System Architecture

The system follows a **multimodal architecture**:

### 🔹 Input Layer

* Image input
* Behavioral inputs (10 ADOS questions)
* Physiological inputs (heart rate, temperature)
* Optional eye tracking data

### 🔹 Feature Extraction

* CNN (ResNet18) → Image features
* LSTM → Motion features
* ANN → Physiological data
* ANN → Behavioral data

### 🔹 Fusion Layer

All extracted features are concatenated into a unified representation.

### 🔹 Classification Layer

Fully connected neural network outputs:

* Autism probability
* Binary classification

---

## ⚙️ 6. Project Structure

```id="b5rdbz"
AUTISM_SCREENING_BACKEND/
│
├── app.py                  # FastAPI backend API
├── model.py                # Full multimodal model
├── multimodel.py           # Simplified inference model
├── eye_tracking.py         # Eye detection using OpenCV
├── autism_model.pt         # Trained model weights
├── haarcascade_eye.xml     # Eye detection classifier
├── requirements.txt        # Dependencies
├── README.md
│
├── install_and_run.bat     # Windows auto setup
├── install_and_run.sh      # Mac/Linux auto setup
│
└── static/
    └── index.html          # Frontend dashboard
```

---

## ⚙️ 7. Installation Instructions

### Step 1: Open Project Folder

```id="rw6cui"
cd AUTISM_SCREENING_BACKEND
```

---

### Step 2: Create Virtual Environment

```id="gaxr1l"
python -m venv venv
```

---

### Step 3: Activate Environment

**Windows**

```id="f0i6dq"
venv\Scripts\activate
```

**Mac/Linux**

```id="g5dw7i"
source venv/bin/activate
```

---

### Step 4: Install Dependencies

```id="5s9kz9"
pip install -r requirements.txt
```

---

## 🚀 8. Running the Application

```id="a2c0jv"
uvicorn app:app --reload
```

---

## 🌐 9. Access the Application

Open in browser:

```id="u4ecqk"
http://127.0.0.1:8000
```

---

## ⚡ 10. One-Click Execution

### Windows

```id="w4d0qg"
double click install_and_run.bat
```

### Mac/Linux

```id="5i7o6c"
chmod +x install_and_run.sh
./install_and_run.sh
```

---

## 🎥 11. Eye Tracking Module

The system includes a simple eye detection mechanism using OpenCV.

### Process:

1. Webcam is activated
2. Frames are captured for 5 seconds
3. Haar Cascade detects eyes
4. Eye contact score is calculated:

```id="z9q4k7"
Eye Contact Score = Frames with Eyes Detected / Total Frames
```

---

## 📊 12. Input Parameters

| Input Type     | Description             |
| -------------- | ----------------------- |
| Image          | Facial image of child   |
| Age            | In months               |
| Gender         | Male / Female           |
| Jaundice       | Yes / No                |
| Family History | Autism history          |
| ADOS           | 10 behavioral questions |
| Heart Rate     | Physiological signal    |
| Temperature    | Physiological signal    |
| Eye Tracking   | Optional                |

---

## 📈 13. Output

* Autism Risk Level:

  * Low
  * Moderate
  * High
* Model Probability Score
* Eye Contact Score (if enabled)

---

## 🧠 14. Model Details

* Base CNN: ResNet18 (feature extractor)
* Motion modeling: LSTM
* Fusion: Feature concatenation
* Classifier: Fully connected ANN

The model is trained using multimodal data and optimized using cross-entropy loss.

---

## 📊 15. Training (Google Colab)

Training is performed using a dataset containing:

* Images
* Motion sequences
* Physiological data
* ADOS behavioral scores

Due to size limitations, dataset is not included.

## To train:

1. Open training notebook (`train.ipynb`)
2. Update dataset path
3. Run all cells
4. Save model as:

```id="y8z36y"
autism_model.pt
```

 ⚠️ 16. Limitations

* Eye tracking is basic 
* motion inputs are simplified in inference
* Model accuracy depends on dataset quality
* Not clinically validated


 🔮 17. Future Improvements

* Real eye gaze tracking
* Mobile application (Android/iOS)
* Cloud deployment (AWS / Render)
* Larger and more diverse dataset
* Improved model accuracy

---

 ⚠️ 18. Disclaimer

This system is intended for:

* Academic demonstration
* Research purposes

It is **not a medical diagnostic tool**.

---

 19. Conclusion

This project demonstrates how AI can be applied to healthcare screening by integrating multiple data sources into a single intelligent system.

It highlights the potential of multimodal deep learning for early detection and decision support.

---
## ⚙️ How to Use the Application

1. Open the web dashboard in your browser
2. Upload an image
3. Enter required details:

   * Age
   * Gender
   * Medical information
4. Answer behavioral questions (ADOS)
5. (Optional) Enable eye tracking
6. Click **Analyze**
7. View prediction result:

   * Autism risk level
   * Probability score
   * Eye contact score
