# 🌶️ Chili Plant Disease Detection

## Overview
This project implements a **deep learning based computer vision system** for detecting diseases in chili plant leaves. The system analyzes leaf images and predicts whether the plant is healthy or affected by a disease.

The model uses **Convolutional Neural Networks (CNN)** for image classification and provides predictions through an interactive **Streamlit web interface**.

The goal of this project is to help farmers and researchers detect plant diseases early and improve crop productivity using AI.

---

## Features

- Upload chili plant leaf images
- Automatic disease detection
- CNN based image classification
- Interactive Streamlit web application
- Model evaluation and training pipeline
- Image preprocessing for improved accuracy

---

## Tech Stack

**Programming Language**
- Python

**Libraries**
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Scikit-learn

---

## Project Structure

```
chili_plant_diesase_detection
│
├── preprocess.py                # Image preprocessing
├── train_function.py            # Model training functions
├── plant_disease_detection.py   # Main prediction script
├── evaluate_model.py            # Model evaluation
├── pesticide.xlsx               # Pesticide recommendation dataset
├── uploaded_image.jpg           # Example image
├── output.log                   # Training logs
├── output1.log                  # Additional logs
└── README.md                    # Project documentation
```

---

## Workflow

### 1. Data Preprocessing
- Load plant leaf images
- Resize images
- Normalize pixel values
- Prepare training and testing datasets

### 2. Model Training
- Train CNN model using plant leaf dataset
- Learn disease patterns from images

### 3. Model Evaluation
- Evaluate model performance
- Generate prediction accuracy metrics

### 4. Prediction
- Upload a chili leaf image
- Model predicts the disease class
- Provides recommended pesticide information

---

## How to Run the Project

### 1 Clone Repository

```
git clone https://github.com/harsh01-uniyal/chili_plant_diesase_detection.git
cd chili_plant_diesase_detection
```

### 2 Install Dependencies

```
pip install -r requirements.txt
```

### 3 Train the Model

```
python train_function.py
```

### 4 Run Prediction

```
python plant_disease_detection.py
```

---

## Example Use Case

1 Upload a chili plant leaf image  
2 Model processes the image  
3 CNN predicts the disease type  
4 Recommended pesticide information is provided  

---

## Future Improvements

- Deploy model as a web service
- Add mobile application support
- Improve dataset size
- Implement real-time detection using camera
- Use advanced architectures like ResNet or EfficientNet
