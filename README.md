# 🌳 Tree Species Classification - Deep Learning Project

This is a Deep Learning web application that classifies tree species based on the uploaded leaf or tree image using a Convolutional Neural Network (CNN).

![Tree Species Classifier Banner](https://imgur.com/your-image-link.jpg) <!-- Optional: Add a project banner if you have one -->

---

## 🔍 Problem Statement

Accurate tree species classification plays a crucial role in forestry, biodiversity conservation, and agriculture. Manual identification is time-consuming and error-prone. This project aims to build an AI-based classifier to automate the species identification process using image recognition.

---

## 🎯 Objectives

- Train a CNN model on a custom tree species image dataset.
- Build an intuitive frontend using Streamlit.
- Deploy a local application that can identify species from uploaded images.

---

## 🛠️ Tools and Technologies Used

- **Python**
- **TensorFlow & Keras**
- **OpenCV**
- **Streamlit**
- **NumPy & Pandas**
- **Matplotlib**
- **Git & GitHub**
- **Git LFS** (for handling large `.h5` model file)

---

## 🧠 Model Summary

- Input Shape: `180x180x3`
- Model Type: CNN (Sequential)
- Layers: Conv2D, MaxPooling, Flatten, Dense, Dropout
- Accuracy: ~90%+ on validation data

---

## 🖼️ Output Screenshot

![App Output](https://imgur.com/your-image-link.jpg) <!-- Replace with actual screenshot -->

---

## 🚀 How to Run the App Locally

```bash
git clone https://github.com/vinayak654321/tree_species_classification-Week-3.git
cd tree_species_classification-Week-3

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
tree_species_classification/
│
├── app.py                  # Streamlit Web App
├── utils.py                # Image processing and prediction logic
├── model.h5                # Trained CNN model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── images/                 # Dataset images (if shared)
 Improvements Done
Customized frontend UI for better visuals

Added confidence score in output

Implemented error handling for image size and shape

Integrated Git LFS for large model upload

Improved model input shape compatibility (180x180)

Dataset
Custom dataset with labeled tree species images.

Image size: 180x180

Classes: Acer Palmatum, Betula Pendula, Ginkgo Biloba, etc.

Author
Vinayak Bhise

AICTE Shell–Edunet Internship 2025

GitHub: vinayak654321

