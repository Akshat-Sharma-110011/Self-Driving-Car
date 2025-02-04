# 🚗 Advanced Self-Driving Car Model

Welcome to the **Advanced Self-Driving Car Model** repository! This project implements a deep learning-based autonomous driving model using TensorFlow and Keras, inspired by NVIDIA's End-to-End Learning for Self-Driving Cars. 🛣️

## 📌 Features
- **Multi-Output Regression Model**: Predicts steering angle, throttle, reverse, and speed.
- **Data Augmentation**: Uses `albumentations` for image transformations (flipping, brightness adjustment, noise addition, rotation, blurring, etc.).
- **Optimized NVIDIA CNN Architecture**: Implements convolutional layers with `elu` activation and batch normalization.
- **Efficient Training**: Implements `mixed_float16` precision for speedup and `ReduceLROnPlateau` scheduler.
- **Custom Data Generator**: Efficiently loads and augments images in real-time.

## 📂 Dataset
The dataset consists of images and driving logs collected from a self-driving car simulator:
- **Images**: Captured from three cameras (center, left, right)
- **CSV File**: Contains labels for `steering`, `throttle`, `reverse`, and `speed`

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-self-driving-car.git
   cd advanced-self-driving-car
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Training the Model
Run the following command to start training:
```bash
python train.py
```
The model will be saved as `advanced_self_driving_model.h5` after training.

## 📊 Model Architecture
- **Conv Layers**: Extract spatial features from images.
- **Batch Normalization & Dropout**: Improve generalization and prevent overfitting.
- **Dense Layers**: Predict the required outputs for steering, throttle, reverse, and speed.

## 🛠️ Usage
To use the trained model for inference:
```bash
python test.py --model advanced_self_driving_model.h5
```

## 📜 Acknowledgments
- NVIDIA's research on **End-to-End Learning for Self-Driving Cars**
- Udacity Self-Driving Car Nanodegree Simulator
- TensorFlow & Keras for deep learning implementation

## 🏆 Contributing
Feel free to open an issue or submit a pull request if you'd like to improve this project!

## 📜 License
This project is open-source and available under the **MIT License**.

Happy Coding! 🚀

