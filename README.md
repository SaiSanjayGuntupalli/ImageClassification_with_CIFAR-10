# Image Classification with CIFAR-10

## Project Overview

This repository showcases an image classification project using the CIFAR-10 dataset, employing a Convolutional Neural Network (CNN) built with **TensorFlow** and **Keras**. The goal of this project is to accurately classify images into one of the 10 categories in the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 distinct classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

### Key Highlights:
- **Model Architecture**: Custom CNN with three convolutional layers followed by max-pooling layers and fully connected layers.
- **Evaluation**: Model performance evaluated using the CIFAR-10 test dataset.
- **Visualization**: Training/validation accuracy and loss plotted to assess model performance during training.
  
## Dataset: CIFAR-10

CIFAR-10 is a well-known dataset for image classification. It contains 60,000 32x32 RGB images across 10 classes:

![image](https://github.com/user-attachments/assets/6376a6a8-4258-4fc7-a4fa-b4d1738c816d)


### Data Split:
- **Training**: 50,000 images
- **Test**: 10,000 images

  
## Libraries & Tools Used

- **TensorFlow**: Deep learning framework used for building and training the CNN model.
- **Keras**: High-level API for building and training the model.
- **Matplotlib**: Used for visualizing training and validation accuracy and loss over epochs.
- **Python**: Programming language for model implementation.

## Getting Started

To run the project on your local machine, follow the instructions below:

### Prerequisites

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SaiSanjayGuntupalli/ImageClassification_with_CIFAR-10.git
   ```
2.  **Install dependencies**:

    It is recommended to create a virtual environment to manage dependencies.
    
    Install dependencies using the provided requirements.txt:
    
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Jupyter Notebook**:

    Open the `Image-Classification with CIFAR-10.ipynb` file in Jupyter Notebook or JupyterLab.

    Run the notebook cells sequentially to train the model and visualize the results

## Model Architecture

The model consists of the following layers:

1. Conv2D Layer: 32 filters, kernel size of (3,3), ReLU activation.
2. MaxPooling2D Layer: Pooling size of (2,2).
3. Conv2D Layer: 64 filters, kernel size of (3,3), ReLU activation.
4. MaxPooling2D Layer: Pooling size of (2,2).
5. Conv2D Layer: 64 filters, kernel size of (3,3), ReLU activation.
6. Flatten Layer: Converts the 3D feature maps to 1D feature vectors.
7. Dense Layer: 64 units, ReLU activation.
8. Dense Layer: 10 units (one per class), used for classification output.

## Training

- **Optimizer**: `Adam` optimizer, a popular choice for training deep learning models due to its adaptive learning rate.
- **Loss Function**: `SparseCategoricalCrossentropy`, which is appropriate for multi-class classification tasks.
- **Metrics**: `Accuracy`, to track the percentage of correctly classified images during training and validation.

The model is trained for 10 epochs with training and validation accuracy tracked throughout.

## Results
After training, the model is evaluated on the test data to measure its performance, with metrics like test accuracy displayed.

## Training & Validation Accuracy Graphs
During training, the model’s accuracy and loss for both training and validation sets are plotted to assess overfitting or underfitting.

## Future Scope

The current model can be improved in several ways. Here are a few suggestions:

- **Improving Model Performance**: Experiment with advanced techniques like **data augmentation**, **dropout**, **batch normalization**, or try more complex architectures like **ResNet** or **VGG**.
- **Transfer Learning**: Leverage pre-trained models like **VGG16**, **ResNet50**, or **InceptionV3** and fine-tune them for better performance on the CIFAR-10 dataset.
- **Hyperparameter Tuning**: Optimize model performance by experimenting with different hyperparameters like learning rate, batch size, and the number of epochs.
- **Ensemble Methods**: Combine multiple models or use ensemble techniques to improve accuracy and robustness.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

