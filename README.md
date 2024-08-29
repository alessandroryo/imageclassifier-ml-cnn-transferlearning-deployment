# ImageClassifier - ML, CNN, Transfer Learning, and Deployment

**Author:** Alessandro Javva Ananda Satriyo

This project focuses on building and deploying an image classification model using machine learning techniques, specifically Convolutional Neural Networks (CNN) and transfer learning. The primary goal is to learn how to deploy models using TensorFlow across various platforms like TensorFlow Lite (TFLite), TensorFlow SavedModel, and TensorFlow.js (TFJS).

## Project Overview

The notebook (`image-classification.ipynb`) contains several key sections:

1. **Data Preparation**:
   - The dataset is sourced from [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data).
   - This section covers data loading, augmentation, and splitting into training, validation, and test sets.
2. **Model Building and Training**:
   - **Convolutional Neural Networks (CNN)**: Implements a CNN model to classify images from the dataset.
   - **Transfer Learning**: Utilizes a pre-trained model to improve accuracy and reduce training time.
3. **Model Evaluation**:
   - The models are evaluated using accuracy metrics on both the training and validation datasets.
4. **Model Deployment**:
   - **TensorFlow Lite (TFLite)**: Converts the trained model into a format suitable for mobile and embedded devices.
   - **TensorFlow SavedModel**: Exports the model for production deployment.
   - **TensorFlow.js (TFJS)**: Converts the model for use in web applications.

## Dataset

The dataset used in this project is sourced from the Intel Image Classification dataset on Kaggle, which contains images categorized into various scenes such as buildings, forests, glaciers, mountains, seas, and streets. The dataset can be accessed [here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data).

## Project Prerequisites

### Required Libraries

The necessary libraries for running this project are listed in the notebook. To install them, run the following command:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Clone the repository to your local machine.
2. Ensure you have all the prerequisites installed.
3. Download the dataset from Kaggle and place it in the appropriate directory as specified in the notebook.
4. Open the notebook using Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook image-classification.ipynb
   ```

5. Execute the cells in order to train the model and observe the results.

## Results

The project concludes with an evaluation of the CNN and transfer learning models, focusing on accuracy metrics for both training and validation datasets. The notebook also demonstrates the deployment of the trained models in various formats (TFLite, SavedModel, and TFJS).

## Research Purpose

The primary goal of this project is to learn and demonstrate the process of deploying TensorFlow models across different platforms. The project serves as an educational tool and is not intended for commercial use.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
