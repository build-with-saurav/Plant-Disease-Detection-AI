# 🌱 Plant Disease Detection AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-TransferLearning-green)
![CNN](https://img.shields.io/badge/CNN-Classification-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-pink)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-yellowgreen)
![PlantVillage](https://img.shields.io/badge/Dataset-PlantVillage-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-91.37%25-success)
![Classes](https://img.shields.io/badge/Classes-15-blueviolet)
![Agriculture AI](https://img.shields.io/badge/AI-Agriculture-important)
![Metrics](https://img.shields.io/badge/Metrics-ConfusionMatrix-yellow)
![Report](https://img.shields.io/badge/Report-ClassificationReport-lightgrey)

AI-powered plant disease detection system using **TensorFlow**, **MobileNetV2**, and **Streamlit** for detecting plant diseases from leaf images and providing actionable disease management recommendations.

---

## 🚀 Features

* Upload leaf images for plant disease prediction
* Real-time disease detection using live camera
* Validation dataset testing mode
* Confusion matrix visualization
* Classification report generation
* Disease description and symptom explanation
* Prevention and treatment recommendations
* Transfer learning based classification system

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* Streamlit
* OpenCV
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```text
Plant-Disease-Detection-AI/
│── app.py
│── train.py
│── requirements.txt
│── disease_knowledge_base.json
│── README.md
│── .gitignore
│
├── Models/
│   ├── plant_disease_model.weights.h5
│   └── class_indices.json
│
├── Processed/
│   └── dataset files
│
├── screenshots/
│   ├── home.png
│   ├── prediction.png
│   ├── metrics.png
│   └── classification-report.png
│
├── notebooks/
│   └── experimentation.ipynb
│
└── docs/
```

---

## 📸 Application Screenshots

### 🏠 Home Page

![Home](screenshots/home.png)

### 🔍 Prediction Output

![Prediction](screenshots/prediction.png)

### 📊 Confusion Matrix

![Metrics](screenshots/metrics.png)

### 📑 Classification Report

![Classification Report](screenshots/classification-report.png)

---

## 📈 Results

| Metric              | Value           |
| ------------------- | --------------- |
| Validation Accuracy | **91.37%**      |
| Validation Loss     | **0.2516**      |
| Training Images     | **32903**       |
| Validation Images   | **8219**        |
| Number of Classes   | **15**          |
| Base Model          | **MobileNetV2** |

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/build-with-saurav/Plant-Disease-Detection-AI.git
cd Plant-Disease-Detection-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

This project uses **MobileNetV2** as a pre-trained backbone for feature extraction.

Architecture:

* MobileNetV2 Base Model
* Global Average Pooling Layer
* Dense Layer (128 units, ReLU activation)
* Dropout Layer (0.5)
* Softmax Output Layer

This architecture helps in achieving better generalization with fewer training parameters.

---

## 🌿 Disease Knowledge Base

The system provides:

* Disease description
* Symptoms
* Prevention methods
* Treatment suggestions

This makes the model useful beyond prediction by adding practical agricultural guidance.

---

## 🔮 Future Improvements

* Grad-CAM Explainability
* Cloud Deployment
* Mobile App Version
* Multi-language Support
* Real-time Continuous Video Detection
* Treatment Recommendation Engine

---

## 👨‍💻 Author

**Saurav Singh**
Computer Science Engineer | AI/ML Enthusiast

GitHub: https://github.com/build-with-saurav
