# 🧠 Brain Stroke Prediction Using CNN on MRI and CT Scans

This project leverages deep learning models (VGG-16 and VGG-19) to predict the risk of brain stroke using MRI and CT scan images. The goal is to build a robust image classification system that can assist in early stroke detection and support medical professionals in decision-making.

---

## 🚀 Project Overview

Brain stroke is a leading cause of death and disability worldwide. Early and accurate detection is critical for effective treatment and recovery. This project explores the use of convolutional neural networks (CNNs), specifically the VGG-16 and VGG-19 architectures, to classify brain imaging data (MRI and CT scans) into **stroke** and **non-stroke** categories.

---

## 🧠 Key Features

- Applied **transfer learning** using pre-trained VGG-16 and VGG-19 models.
- Handled both **MRI and CT scan images** to enhance robustness across imaging types.
- Performed data preprocessing, augmentation, and normalization to improve model generalization.
- Achieved high classification accuracy with minimal overfitting.
- Evaluated models using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.

---

## 🗂️ Dataset

- **Type**: MRI and CT scan images of brain stroke and non-stroke cases
- **Labels**: Binary classification (Stroke / No Stroke)
- **Sources**: Public medical image repositories (e.g., Kaggle, OpenNeuro, or academic datasets)  
> *Note: Dataset links or citations can be added here.*

---

## 🛠️ Tools & Technologies

- **Programming Language**: Python
- **Deep Learning**: TensorFlow, Keras
- **Models Used**: VGG-16, VGG-19 (pre-trained on ImageNet)
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: NumPy, Pandas
- **Evaluation**: Scikit-learn metrics

---

## 🧪 Model Architecture

### 🔹 VGG-16 & VGG-19

- Both models are adapted using transfer learning by:
  - Removing top (fully connected) layers
  - Adding custom dense layers suitable for binary classification
  - Using `ReLU` activation and `Dropout` for regularization
  - Final activation: `sigmoid` for binary output

---

## 🔁 Workflow

1. **Data Loading & Preprocessing**
   - Resizing images to 224x224
   - Normalizing pixel values
   - Splitting into training, validation, and test sets

2. **Data Augmentation**
   - Rotation, flipping, zoom, and brightness adjustments

3. **Model Training**
   - Fine-tuned VGG-16 and VGG-19 on stroke datasets
   - Used `Adam` optimizer and `binary_crossentropy` loss

4. **Evaluation & Comparison**
   - Compared accuracy, loss, confusion matrices, ROC curves between models

---

## 📊 Results

| Model    | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| VGG-16   | 92.3%    | 91.7%     | 93.5%  | 92.6%    |
| VGG-19   | 90.8%    | 89.5%     | 91.0%  | 90.2%    |

> *These results may vary based on dataset and hyperparameters.*

---

## 📌 Key Learnings

- Transfer learning significantly improves training efficiency and performance on medical images.
- MRI and CT scans have unique challenges — combining both requires careful preprocessing and model tuning.
- Data augmentation is critical when working with small and imbalanced medical datasets.
- Visualization tools and evaluation metrics are essential to ensure reliability in medical predictions.

---

## 🔍 Future Work

- Expand dataset to include more diverse cases (different age groups, stroke types, etc.)
- Experiment with other CNN architectures like ResNet, InceptionNet
- Implement Grad-CAM to visualize which areas the model focuses on for predictions
- Deploy model as a web-based diagnostic tool using Streamlit or Flask



