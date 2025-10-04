# CIFAR-10 Image Classification

A deep learning project for classifying images from the CIFAR-10 dataset using Convolutional Neural Networks (CNN) and Transfer Learning with ResNet50.

## Overview

This project implements two different approaches to classify images in the CIFAR-10 dataset:
1. A custom CNN architecture with batch normalization and dropout
2. Transfer learning using pre-trained ResNet50 model

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Prerequisites

```bash
pip install kaggle
pip install py7zr
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install Pillow
pip install opencv-python
```

## Setup

### 1. Kaggle API Configuration

Place your `kaggle.json` API token file in the project directory, then run:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Dataset

```bash
kaggle competitions download -c cifar-10
```

### 3. Extract Files

The notebook includes code to extract the downloaded zip and 7z files automatically.

## Project Structure

```
├── kaggle.json          # Kaggle API credentials
├── train/              # Training images (extracted)
├── trainLabels.csv     # Image labels
└── notebook.ipynb      # Main implementation notebook
```

## Implementation Details

### Data Preprocessing

- Images are loaded from the training directory
- Resized to 32x32 pixels
- Converted to numpy arrays
- Normalized by dividing pixel values by 255.0
- Split into 80% training and 20% testing sets

### Model 1: Custom CNN Architecture

**Architecture:**
- Conv2D (32 filters) → BatchNormalization → Conv2D (32 filters) → BatchNormalization → MaxPooling2D → Dropout(0.25)
- Conv2D (64 filters) → BatchNormalization → Conv2D (64 filters) → BatchNormalization → MaxPooling2D → Dropout(0.25)
- Flatten → Dense(256) → BatchNormalization → Dropout(0.5) → Dense(10, softmax)

**Training:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 50
- Validation Split: 10%

### Model 2: ResNet50 Transfer Learning

**Architecture:**
- UpSampling layers (3x) to match ResNet50 input requirements (256x256)
- Pre-trained ResNet50 (ImageNet weights, without top layer)
- Flatten → BatchNormalization → Dense(128) → Dropout(0.5)
- BatchNormalization → Dense(64) → Dropout(0.5)
- BatchNormalization → Dense(10, softmax)

**Training:**
- Optimizer: RMSprop (learning rate: 2e-5)
- Loss: Sparse Categorical Crossentropy
- Epochs: 10
- Validation Split: 10%

## Usage

1. Open the notebook in Google Colab or Jupyter
2. Run cells sequentially to:
   - Download and extract the dataset
   - Process images and labels
   - Train the custom CNN model
   - Train the ResNet50 transfer learning model
   - Visualize training history

## Results Visualization

The notebook includes code to plot:
- Training vs Validation Loss
- Training vs Validation Accuracy

These visualizations help monitor model performance and identify overfitting.

## Label Encoding

Classes are encoded as integers:
```python
{
    'airplane': 0, 'automobile': 1, 'bird': 2, 
    'cat': 3, 'deer': 4, 'dog': 5, 
    'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}
```

## Notes

- The ResNet50 model requires upsampling input images from 32x32 to 256x256
- Batch normalization and dropout layers help prevent overfitting
- The custom CNN is more lightweight compared to ResNet50
- ResNet50 leverages pre-trained weights for potentially better feature extraction

## Future Improvements

- Implement data augmentation for better generalization
- Add early stopping and model checkpointing
- Compare model performance metrics
- Experiment with other pre-trained architectures (VGG16, InceptionV3)
- Fine-tune hyperparameters
- Deploy the best model for inference

## License

This project uses the CIFAR-10 dataset from the Kaggle competition.

## Acknowledgments

- CIFAR-10 dataset creators
- Kaggle for hosting the competition
- TensorFlow and Keras teams
