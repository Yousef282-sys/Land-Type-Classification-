# Kemet 🛰️

> Land Type & Sand Type Classification Using Sentinel-2 and EuroSAT

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Team Dune** - Automating land classification in Egypt using deep learning and satellite imagery, with a focus on unlocking the economic potential of Egypt's diverse sand resources.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Business Applications](#business-applications)
- [Team](#team)
- [Future Work](#future-work)
- [License](#license)

## 🌍 Overview

Kemet is a deep learning-based land classification system that analyzes satellite imagery from Sentinel-2 and EuroSAT to identify land types across Egypt, including agriculture, desert, water bodies, roads, urban areas, and trees. The project also explores the economic potential of classifying Egypt's diverse sand types for industrial applications.

### Why Kemet?

Egypt's geography includes vast deserts, agricultural zones, water bodies, and rapidly growing urban regions. Automating land classification is crucial for environmental monitoring, agriculture planning, water resource management, and urban expansion analysis. Additionally, Egypt contains economically valuable sand types that are currently underutilized.

## ✨ Features

- **Multi-source Dataset Integration**: Combines EuroSAT and custom Sentinel-2 Egypt datasets
- **Real-time Classification**: Deployed system for instant land type predictions
- **Robust CNN Architecture**: Custom CNN with batch normalization and dropout
- **Comprehensive Preprocessing**: Automated data cleaning, augmentation, and normalization
- **Sand Type Classification**: Extension for identifying economically valuable sand types
- **Interactive UI**: User-friendly interface for image uploads and results visualization

## 🎯 Project Objectives

1. Build a deep learning model to classify land types in Egypt
2. Combine EuroSAT and custom Sentinel-2 Egypt data
3. Create a real-time prediction system with interactive UI
4. Explore economic and industrial applications of sand type classification
5. Support national resource management and strategic planning

## 📊 Dataset

### Data Sources

**EuroSAT Dataset**: RGB and multispectral images labeled across land classes

**Custom Sentinel-2 Egypt Dataset**: Collected and cleaned specifically for Egyptian geography
- Crops
- Desert
- Urban areas
- Water bodies
- Roads
- Trees

### Data Processing Pipeline

1. **Data Collection**: QGIS-based extraction from Sentinel-2
2. **Data Cleaning**:
   - Removed corrupted and cloud-contaminated images
   - Converted non-RGB images to RGB
   - Aligned image resolution
   - Fixed spectral band ranges
   - Manual inspection and relabeling
   - Normalized pixel intensity

3. **Data Augmentation**:
   - Rotation, flipping, zoom
   - Brightness shift
   - Random transformations

4. **Data Split**: 80/20 train-test split

## 🏗️ Model Architecture

### Custom CNN

```python
Sequential([
    Rescaling(1./255),
    
    Conv2D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(256, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
```

### Training Configuration

- **Loss Function**: Categorical cross-entropy
- **Optimizer**: Adam
- **Epochs**: 20-50 (based on convergence)
- **Image Size**: 128×128 pixels
- **Early Stopping**: Implemented to prevent overfitting

### Models Tested

- Custom CNN (baseline)
- ResNet50 (transfer learning)
- VGG16 (transfer learning)
- EfficientNet

## 🚀 Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- QGIS (for data collection)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/kemet.git
cd kemet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```txt
tensorflow>=2.0
numpy
pandas
matplotlib
opencv-python
scikit-learn
pillow
fastapi
uvicorn
```

## 💻 Usage

### Training the Model

```python
from kemet import train_model

# Train with default parameters
model = train_model(
    dataset_path="data/",
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)
```

### Making Predictions

```python
from kemet import predict_land_type

# Load image and predict
result = predict_land_type("path/to/satellite_image.jpg")

print(f"Predicted Class: {result['class']}")
print(f"Confidence: {result['confidence']}%")
```

### Inference Pipeline

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('kemet_model.h5')

# Load and preprocess image
img = Image.open('image.jpg').resize((128, 128))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_idx = np.argmax(prediction)
confidence = prediction[0][class_idx] * 100

print(f"Class: {class_names[class_idx]}, Confidence: {confidence:.2f}%")
```

### Deployment

```bash
# Start the FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Access the API at http://localhost:8000
# Upload images via the interactive UI
```

## 📈 Results

### Performance Metrics

- **Overall Accuracy**: High accuracy across all land classes
- **Precision/Recall**: Balanced performance
- **Confusion Matrix**: Strong classification with minimal errors
- **Generalization**: Excellent performance on both EuroSAT and Egypt datasets

### Training Progress

- Train accuracy improved steadily across epochs
- Validation accuracy stabilized in later epochs
- Loss decreased consistently for both train and validation sets

## 💼 Business Applications

### Sand Type Classification

Egypt contains vast reserves of economically valuable sand types with diverse industrial applications:

#### White Sand (Silica Sand)
- **Purity**: Up to 99% SiO₂
- **Applications**:
  - Pharmaceuticals (drug carriers, tablets)
  - Cosmetics (exfoliants, creams)
  - Glass manufacturing
  - Solar panel production
  - Electronics and semiconductors
- **Value**: High international demand, strong export potential

#### Black Sand
- **Composition**: Rich in heavy minerals (ilmenite, zircon, magnetite)
- **Applications**:
  - Titanium extraction
  - Nuclear and military applications
  - High-strength alloys
  - Ceramics and refractories
- **Value**: Strategic mineral resource

#### Yellow/Building Sand
- **Applications**:
  - Construction and concrete
  - Road base layers
  - Bricks and building materials
- **Value**: Essential for urban expansion

#### Desert Agricultural Sand
- **Applications**:
  - Desert reclamation projects
  - Soil mixing for agriculture
- **Value**: Supports food security and agricultural expansion

### Industrial Impact

- **Glass Manufacturing**: High-purity silica for photovoltaic glass
- **Construction**: Infrastructure and smart-city projects
- **Pharmaceuticals & Cosmetics**: Regional supplier for UAE, Saudi Arabia, Europe
- **Environmental Monitoring**: Dune movement prediction, soil classification
- **National Projects**: Support for New Delta, New Administrative Capital, Sinai Development

### Market Potential

Egypt's strategic advantages include abundant desert resources, Suez Canal access, Mediterranean ports, and a growing AI ecosystem. This project transforms raw satellite data into actionable intelligence for nationwide land assessment.

## 👥 Team

**Team Dune**

- Ahmed Shaboury
- Nehal Taha
- Rahma Ahmed
- Yousef Ahmed
- Nassf Hussain

## 🔮 Future Work

### Planned Extensions

- Link web interface to targeted audiences
- Add more sand and soil categories
- Implement pixel-level segmentation models
- Analyze sand movement over time
- Extend dataset with additional Sentinel-2 bands (NIR, SWIR)
- Deploy as full-featured web app with GIS visualization
- Field verification datasets for improved accuracy
- Integration with Egyptian mega-projects

### Potential Improvements

- Enhanced spectral feature analysis for sand signatures
- Real-time monitoring dashboards
- Mobile application development
- API for third-party integration
- Multi-temporal analysis capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free t


