# AI GENERATED IMAGE DETECTION USING ML

Abstract
This repository contains code for training a Convolutional Neural Network (CNN) model using TensorFlow and Keras to detect AI-generated images. With the increasing use of generative models like GANs and deepfakes. It involves loading training and testing data from CSV files, preprocessing the data, creating a CNN model with specialized architecture, and training the model.
## Prerequisites

Before running the code, ensure that you have the necessary libraries installed:

- pandas: Library for data manipulation and analysis.
- numpy: Library for numerical computations.
- scikit-learn: Library for machine learning algorithms.
- TensorFlow: Open-source deep learning framework.

You can install the required libraries using pip:

```shell
pip install pandas numpy scikit-learn tensorflow
```
## Dataset
The project assumes that you have a CSV file named train.csv and test.csv that contains your training and testing data respectively. The file should have the following structure:

### Train.csv
- labels represents the target variable indicating whether the image is real (0) or fake (1).
- feature_1 to feature_n represent the input features of the images.
Ensure that you update the file path in the code where the dataset is loaded:

  `train_data = pd.read_csv('/path/to/train.csv')`


## Description

This code performs the following steps:

1. Load the training data from a CSV file (`train.csv`): The CSV file contains the image data and corresponding labels for training the CNN model.(Change the directory in code to the directory of train.csv file in your pc)

2. Separate the features (neural layers) and the target variable: The features are extracted from the CSV file, while the target variable represents the labels associated with each image.

3. Reshape the features to an image-like format: The features are reshaped to match the dimensions of an image to be compatible with the CNN model.

4. Convert the target variable to categorical format: The target variable (labels) is converted to categorical format using one-hot encoding to represent each class as a binary vector.

5. Split the data into training and validation sets: The data is divided into training and validation sets to evaluate the performance of the model.

6. Create a CNN model using Keras: The CNN model is built using the Sequential API from Keras, consisting of multiple convolutional and fully connected layers.

7. Compile the model with the Adam optimizer and a lower learning rate: The model is compiled with the Adam optimizer, which adapts the learning rate during training to improve performance.

8. Train the model for a specified number of epochs: The model is trained using the training data and evaluated using the validation data for a specific number of epochs, updating the model's weights to minimize the loss function.

9. Load the testing data from a CSV file (`test.csv`): The CSV file contains the image data for which the model needs to make predictions.(Change the directory in code to the directory of test.csv file in your pc)

10. Separate the features from the testing data and reshape them: The features from the testing data are extracted and reshaped to match the input shape of the trained model.

11. Make predictions on the testing data using the trained model: The trained model predicts the labels for the testing data, providing a probability distribution for each class.

12. Save the predicted labels to a CSV file (`solution_format.csv`): The predicted labels are stored in a DataFrame along with their corresponding IDs and saved to a CSV file(Specify the directory where you want to store the solution file).
    `result_df.to_csv('path/solution_format.csv', index=False)`

13. Print the validation accuracy of the model: The accuracy of the model on the validation set is calculated and printed to evaluate the performance of the trained model.

## Accuracy
83.14%

## Author
NIBHRANT VAISHNAV
