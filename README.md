# Gender Detection Using Deep Learning

## 📌 Project Overview
This project implements a **gender detection model** using **deep learning**. The model classifies images into **Male** or **Female** categories based on facial features.

## 🛠️ Technologies Used
- **TensorFlow & Keras** - Model building and training
- **OpenCV** - Image preprocessing
- **Matplotlib** - Visualizing predictions
- **ImageDataGenerator** - Data augmentation
- **Adam Optimizer** - Efficient training

## 📂 Dataset Structure
```
/dataset1
│── /train
│   ├── /male
│   ├── /female
│── /validation
│   ├── /male
│   ├── /female
│── /test
│   ├── /male
│   ├── /female
```

## 🚀 Model Architecture
- **Conv2D** layers with ReLU activation
- **BatchNormalization & MaxPooling2D** for feature extraction
- **Flatten & Dense layers** for classification
- **Dropout** to prevent overfitting

## 🔥 Training Strategy
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam (learning rate = 0.0001)
- **Callbacks:**
  - ModelCheckpoint (Save best model)
  - EarlyStopping (Prevent overfitting)
  - ReduceLROnPlateau (Adjust learning rate)

## 📊 Model Performance
- **Training Accuracy:** ~92%
- **Validation Accuracy:** ~94%
- **Test Accuracy:** ~94%
- **Test Loss:** 0.7182 (Can improve with more data and fine-tuning)

## 🎯 How to Use
### **1️⃣ Train the Model**
Run the notebook to train the model on the dataset.

### **2️⃣ Evaluate the Model**
Test the model using a separate test dataset.

### **3️⃣ Predict on New Images**
Use `predict_and_visualize()` to classify random test images.

### **4️⃣ Live Camera Testing**
Run live gender prediction using OpenCV.

## 🖼️ Sample Predictions
The model randomly selects test images and predicts their gender, displaying results in a grid format.

## 📌 Future Improvements
- Increase dataset diversity
- Apply advanced augmentation techniques
- Tune hyperparameters further
- Use transfer learning (e.g., VGG16, ResNet)

## 📝 Author
👨‍💻 **Dipak K. Gaikwad**

## ⭐ Star this repo if you found it useful!

