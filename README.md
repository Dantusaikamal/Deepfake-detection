# Deepfake Detection 

This project provides a lightweight deepfake detection system using a Convolutional Neural Network (CNN) trained on face images. It includes a modern Streamlit dashboard for interactive predictions.

## Features

* Binary classifier (real vs. fake face)
* Efficient image loading using `ImageDataGenerator`
* Streamlit-based user interface
* Training logs saved to CSV
* Responsive, clean UI

---

## Project Structure

```
deepfake-detector/
├── app.py                  # Streamlit dashboard
├── model_train.py          # Training script for CNN model
├── deepfake_model.h5       # Trained model (generated after training)
├── training_log.csv        # CSV log file (generated after training)
└── data/
    ├── real/               # Folder of real face images
    └── fake/               # Folder of fake (deepfake) face images
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dantusaikamal/Deepfake-detection.git
cd deepfake-detector
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On Linux/macOS
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

Or

```bash
pip install tensorflow opencv-python streamlit pillow pandas matplotlib
```

---

## Preparing the Dataset

Create a folder named `data/` in the root directory with the following structure:

```
data/
├── real/    # Add real face images here
└── fake/    # Add deepfake or synthetic face images here
```

You can use datasets such as:

* [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

Start with a small sample (e.g., 500–1000 images per class) to test.

---

## Training the Model

Run the training script:

```bash
python model_train.py
```

This will:

* Train the CNN model
* Save the best model as `deepfake_model.h5`
* Save training logs in `training_log.csv`

---

## Running the Streamlit Dashboard

After training completes:

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to access the dashboard.

---

## Notes

* Ensure the `deepfake_model.h5` file is present in the project root before running `app.py`
* The dashboard expects images with faces. Results may be unreliable for non-face inputs.
* GPU acceleration is supported via TensorFlow if available.

---
