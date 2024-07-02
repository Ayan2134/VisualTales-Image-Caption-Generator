## Task 1: CNN Model for Pizza vs. Non-Pizza Classification

### Objective
Build a Convolutional Neural Network (CNN) model to classify images as either pizza or non-pizza using a dataset from Kaggle.

### Requirements
1. **Data Loading**: Download a pizza vs. non-pizza dataset from Kaggle and load it using data loading utilities.
2. **Model Definition**: Create a CNN model with the following layers:
   - Convolutional layers
   - Pooling layers
   - Fully connected layers
3. **Sepration**: Split the dataset into 2 parts in 80:20 ratio for training and testing respectively.
4. **Training**: Train your model on the training set. Use an appropriate loss function and optimizer.
5. **Testing**: Test your trained model on a test set that was not used during training or validation.
6. **Visualization**: Visualize a few input images along with their corresponding predicted and actual labels.

### Model Architecture
**Sample Architecture**
- **Input Layer**: Input image of shape (height, width, channels)
- **Convolutional Layer 1**: Number of filters, filter size, activation function
- **Pooling Layer 1**: Pool size, stride
- **Convolutional Layer 2**: Number of filters, filter size, activation function
- **Pooling Layer 2**: Pool size, stride
- **Flatten Layer**
- **Fully Connected Layer 1**: Number of units, activation function
- **Output Layer**: Number of units (2 for binary classification), activation function (softmax for classification)

### Kaggle Dataset
- Dataset: [Pizza vs. Non-Pizza](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)
- Instructions: Download the dataset from Kaggle and prepare it for training.

## Task 2: Transfer Learning Model with Moderate-Level Dataset

### Objective
Build a transfer learning model using a pre-trained network to classify images from a moderate-level dataset. Describe the problem statement and create a model with a few additional layers.

### Requirements
1. **Problem Statement**: You have to add layers to a pre-trained model to work on new dataset.
2. **Data Loading**: Load the dataset using data loading utilities.
3. **Model Definition**: Use a pre-trained network (e.g., VGG16, ResNet50) and add custom layers:
   - A few convolutional layers
   - A few fully connected layers
4. **Sepration**: Split the dataset into 2 parts in 80:20 ratio for training and testing respectively.
5. **Training**: Train your model on the training set. Use an appropriate loss function and optimizer.
6. **Testing**: Test your trained model on a test set that was not used during training or validation.
7. **Visualization**: Visualize a few input images along with their corresponding predicted and actual labels.

### Dataset
- Dataset: [CIFAR 10](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python)
- Instructions: Download the dataset from Kaggle and prepare it for training.

## Notes
Feel free to explore additional features or techniques to enhance your models. Good luck!
