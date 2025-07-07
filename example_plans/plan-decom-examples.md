# Examples of Plan Decomposition


## Data Agent

**Plan Decomposition with a given global plan**
```
As a proficient data scientist, summarize the following plan given by the senior AutoML project manager according to the user's requirements and your expertise in data science.
        
        # User's Requirements
        ```json
        {'user': {'intent': 'build', 'expertise': 'medium'}, 'problem': {'area': 'computer vision', 'downstream_task': 'image classification', 'application_domain': 'fashion', 'description': 'Please provide a classification model that categorizes images into one of four clothing categories. The image path, along with its label information, can be found in the files train_labels.csv and test_labels.csv. The model should achieve at least 85% accuracy on the test set and be implemented using PyTorch. Additionally, please include data augmentation techniques and a confusion matrix in the evaluation.', 'performance_metrics': [{'name': 'accuracy', 'value': 0.85}], 'complexity_metrics': []}, 'dataset': [{'name': '', 'modality': ['image', 'text'], 'target_variables': [], 'specification': None, 'description': 'The dataset includes images of clothing items along with their label information in the files train_labels.csv and test_labels.csv.', 'preprocessing': [], 'augmentation': ['data augmentation techniques'], 'visualization': ['confusion matrix'], 'source': 'user-upload'}], 'model': [{'name': '', 'family': '', 'type': 'neural networks', 'specification': None, 'description': 'A classification model to categorize images into one of four clothing categories, implemented using PyTorch.'}], 'knowledge': [], 'service': {'feature_engineering': [], 'target_device': '', 'deployment_endpoint': '', 'inference_engine': ['PyTorch'], 'demonstration': ''}}
        ```
        
        # Project Plan
        ### End-to-End Actionable Plan for Fashion Image Classification Using PyTorch

#### 1. Data Preparation
1. **Load Dataset**:
   - Load the image paths and corresponding labels from `train_labels.csv` and `test_labels.csv`.
   - Ensure the dataset includes images of four clothing categories.

2. **Convert Grayscale to RGB**:
   - If the images are in grayscale, convert them to RGB by replicating the single channel across three channels.

3. **Preprocessing**:
   - Normalize the images by scaling pixel values to the range [0, 1].
   - Resize the images to a consistent size, e.g., 224x224 pixels, suitable for model input.

4. **Split Dataset**:
   - Split the training data into training and validation sets (e.g., 80% training, 20% validation) to monitor the model's performance and prevent overfitting.

#### 2. Data Augmentation
1. **Techniques**:
   - Apply data augmentation techniques to increase the diversity and robustness of the training data.
   - Techniques to include: random horizontal flip, normalization, random cropping, random erasing, rotating, shifting, and zooming.

2. **Pipelines**:
   - Develop two preprocessing pipelines:
     - Pipeline 1: Random horizontal flip and normalization.
     - Pipeline 2: Random horizontal flip, normalization, random cropping, and random erasing.

#### 3. Model Architecture
1. **Simple CNN**:
   - Start with a simple Convolutional Neural Network (CNN) model with at least three convolutional layers followed by max pooling layers, and fully connected layers for classification.

2. **Advanced Models**:
   - Consider using more advanced models like ResNet18, EfficientNet, or Vision Transformers (ViTs) which have shown high accuracy in similar tasks.

3. **Hybrid Models**:
   - Explore hybrid models combining CNNs and ViTs to leverage the strengths of both architectures.

#### 4. Training
1. **Optimizer and Loss Function**:
   - Use the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.015 or the Adam optimizer.
   - Use Cross-Entropy Loss for training.

2. **Epochs and Early Stopping**:
   - Train the models for up to 150 epochs.
   - Implement early stopping to prevent overfitting if the validation loss does not improve for a certain number of epochs.

3. **Monitoring**:
   - Monitor validation accuracy and loss curves to ensure the model generalizes well to unseen data.

#### 5. Evaluation
1. **Accuracy**:
   - Ensure the model achieves at least 85% accuracy on the test set.

2. **Confusion Matrix**:
   - Generate a confusion matrix to analyze the model's performance across different clothing categories.

3. **Top-1 Accuracy**:
   - Calculate the top-1 accuracy to evaluate the model's performance.

#### 6. Implementation
1. **Framework**:
   - Leverage PyTorch and its torchvision.transforms.v2 for efficient data augmentation and model training.

2. **Resources**:
   - Utilize resources and code repositories from platforms like Kaggle and PapersWithCode to implement and fine-tune the models effectively.

3. **Deployment**:
   - Although not specified, consider preparing the model for deployment by converting it to a format suitable for the target device and setting up an inference endpoint if required in the future.

#### 7. Visualization and Reporting
1. **Confusion Matrix**:
   - Visualize the confusion matrix to provide insights into the model's classification performance.

2. **Performance Metrics**:
   - Report the accuracy and other relevant performance metrics.

3. **Documentation**:
   - Document the entire process, including data preprocessing steps, model architecture, training parameters, evaluation results, and any challenges encountered.

By following this comprehensive plan, the AI agents should be able to develop a robust image classification model that meets the specified accuracy requirements and provides insightful performance analysis.
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the data manipulation and analysis.
        1. How to retrieve or collect the dataset(s)?
        2. How to check types, shapes, and properties of the dataset(s) according to the downstream task's requirements?
        3. How to preprocess the retrieved dataset(s)?
        4. How to efficiently augment the dataset(s)?
        5. How to extract and understand the underlying characteristics of the dataset(s)?
        
        Make sure that another data scientist can exectly reproduce the results based on your summary.
```

**Plan Execution with decomposed plan**
```
As a proficient data scientist, your task is to explain **detailed** steps for data manipulation and analysis parts by executing the following machine learning development plan.
        
        # Plan
        ### Summary of the Plan for Fashion Image Classification Using PyTorch

#### 1. Dataset Retrieval and Collection
- **Retrieve Dataset**: Load the image paths and corresponding labels from the provided `train_labels.csv` and `test_labels.csv` files. Ensure the dataset includes images of four clothing categories.

#### 2. Checking Dataset Types, Shapes, and Properties
- **Data Types**: Verify that the images are in the correct format (e.g., JPEG, PNG) and the labels are correctly mapped to the images.
- **Shapes**: Check the dimensions of the images. If images are in grayscale, convert them to RGB by replicating the single channel across three channels.
- **Properties**: Ensure that the dataset is balanced across the four clothing categories and that there are no missing or corrupted files.

#### 3. Preprocessing the Dataset
- **Normalization**: Scale pixel values to the range [0, 1].
- **Resizing**: Resize all images to a consistent size (e.g., 224x224 pixels) suitable for model input.
- **Splitting**: Split the training data into training and validation sets (e.g., 80% training, 20% validation) to monitor the model's performance and prevent overfitting.

#### 4. Efficient Data Augmentation
- **Techniques**: Apply data augmentation techniques to increase the diversity and robustness of the training data. Techniques include:
  - Random horizontal flip
  - Normalization
  - Random cropping
  - Random erasing
  - Rotating
  - Shifting
  - Zooming
- **Pipelines**:
  - **Pipeline 1**: Random horizontal flip and normalization.
  - **Pipeline 2**: Random horizontal flip, normalization, random cropping, and random erasing.

#### 5. Extracting and Understanding Underlying Characteristics
- **Class Distribution**: Analyze the distribution of classes to ensure a balanced dataset.
- **Image Quality**: Check for variations in image quality and resolution.
- **Data Insights**: Identify any patterns or anomalies in the data that could affect model performance.
- **Performance Metrics**: Use metrics such as accuracy and confusion matrix to evaluate the model's performance and understand misclassification patterns.

By following these steps, another data scientist should be able to reproduce the results and build a robust image classification model that meets the specified accuracy requirements and provides insightful performance analysis.
        
        # Potential Source of Dataset
        [{'name': '', 'loader_key': 'Note that the uploaded dataset is available at "_experiments/datasets/shopee". This is the structure of the dataset directory: [\'_experiments/datasets/shopee/train\', \'_experiments/datasets/shopee/test\', \'_experiments/datasets/shopee/train_labels.csv\', \'_experiments/datasets/shopee/test_labels.csv\'].', 'source': 'user-upload'}]
        
        Make sure that your explanation follows these instructions:
        - All of your explanation must be self-contained without using any placeholder to ensure that other data scientists can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or collect the data.
        - Include how to preprocess the data and which tools or libraries are used for the preprocessing.
        - Include how to validate and check the types, shapes, and sizes of the data for accurate modeling in the subsequent processes.
        - Include how to do the data augmentation with details and names.
        - Include how to extract and understand the characteristics of the data.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.
        Make sure to focus only on the data part as it is your expertise. Do not conduct or perform anything regarding modeling or training.
        After complete the explanations, explicitly specify the (expected) outcomes and results both quantitative and qualitative of your explanations.
```

**Data Plan Execution Result**
```
### Detailed Steps for Data Manipulation and Analysis

#### 1. Dataset Retrieval and Collection

**Retrieve Dataset**:
- **Source**: The dataset is available at `_experiments/datasets/shopee`.
- **Structure**: The directory structure includes `train`, `test`, `train_labels.csv`, and `test_labels.csv`.
- **Loading Data**: Load the image paths and corresponding labels from `train_labels.csv` and `test_labels.csv`. This can be done using libraries like `pandas` to read the CSV files and `os` to navigate through the directories.

**Why**: This step ensures that we have access to the correct dataset and that the paths and labels are correctly mapped for further processing.

#### 2. Checking Dataset Types, Shapes, and Properties

**Data Types**:
- **Image Format**: Verify that images are in the correct format (JPEG, PNG). This can be done using libraries like `PIL` (Python Imaging Library).
- **Labels**: Ensure labels are correctly mapped to images by checking the contents of the CSV files.

**Shapes**:
- **Image Dimensions**: Check the dimensions of the images. If images are in grayscale, convert them to RGB by replicating the single channel across three channels using libraries like `PIL` or `OpenCV`.

**Properties**:
- **Balanced Dataset**: Ensure the dataset is balanced across the four clothing categories by analyzing the label distribution using `pandas`.
- **Integrity Check**: Verify that there are no missing or corrupted files by attempting to open each image file.

**Why**: Ensuring correct data types, shapes, and properties is crucial for accurate model training and prevents errors during preprocessing and augmentation.

#### 3. Preprocessing the Dataset

**Normalization**:
- **Scale Pixel Values**: Normalize pixel values to the range [0, 1] using libraries like `PIL` or `torchvision.transforms`.

**Resizing**:
- **Consistent Size**: Resize all images to a consistent size (e.g., 224x224 pixels) suitable for model input using `PIL` or `torchvision.transforms`.

**Splitting**:
- **Train-Validation Split**: Split the training data into training and validation sets (e.g., 80% training, 20% validation) using `sklearn.model_selection.train_test_split`.

**Why**: Normalization ensures that the pixel values are on a common scale, which helps in faster convergence during training. Resizing ensures that all images have the same dimensions, which is a requirement for most deep learning models. Splitting the data helps in monitoring the model's performance and prevents overfitting.

#### 4. Efficient Data Augmentation

**Techniques**:
- **Random Horizontal Flip**: Randomly flip images horizontally to increase diversity.
- **Normalization**: Normalize images to have zero mean and unit variance.
- **Random Cropping**: Randomly crop images to introduce variability.
- **Random Erasing**: Randomly erase parts of images to make the model robust to occlusions.
- **Rotating**: Rotate images randomly to make the model invariant to orientation.
- **Shifting**: Shift images randomly to simulate different viewpoints.
- **Zooming**: Zoom in and out of images to introduce scale variability.

**Pipelines**:
- **Pipeline 1**: Apply random horizontal flip and normalization.
- **Pipeline 2**: Apply random horizontal flip, normalization, random cropping, and random erasing.

**Why**: Data augmentation increases the diversity and robustness of the training data, which helps in improving the generalization of the model.

#### 5. Extracting and Understanding Underlying Characteristics

**Class Distribution**:
- **Analyze Distribution**: Use `pandas` to analyze the distribution of classes to ensure a balanced dataset.

**Image Quality**:
- **Check Variations**: Assess variations in image quality and resolution using `PIL` or `OpenCV`.

**Data Insights**:
- **Identify Patterns**: Use exploratory data analysis (EDA) techniques to identify any patterns or anomalies in the data that could affect model performance.

**Performance Metrics**:
- **Evaluate Model**: Use metrics such as accuracy and confusion matrix to evaluate the model's performance and understand misclassification patterns. This can be done using `sklearn.metrics`.

**Why**: Understanding the underlying characteristics of the data helps in identifying potential issues that could affect model performance. Analyzing class distribution ensures that the dataset is balanced, which is crucial for training a fair and unbiased model.

### Expected Outcomes and Results

**Quantitative**:
- **Balanced Dataset**: A dataset with an equal number of images for each clothing category.
- **Consistent Image Size**: All images resized to 224x224 pixels.
- **Normalized Pixel Values**: Pixel values scaled to the range [0, 1].
- **Augmented Data**: Increased dataset diversity through augmentation techniques.

**Qualitative**:
- **Data Integrity**: No missing or corrupted files.
- **Improved Model Generalization**: Enhanced model robustness due to diverse and augmented training data.
- **Insightful Analysis**: Clear understanding of class distribution, image quality, and potential data patterns or anomalies.

By following these detailed steps, another data scientist should be able to reproduce the results and build a robust image classification model that meets the specified accuracy requirements and provides insightful performance analysis.
```

## Model Agent

**Plan Decomposition with global plan and data manipulation results**
```
As a proficient machine learning research engineer, summarize the following plan given by the senior AutoML project manager according to the user's requirements, your expertise in machine learning, and the outcomes from data scientist.
        
        **User's Requirements**
        ```json
        {'user': {'intent': 'build', 'expertise': 'medium'}, 'problem': {'area': 'computer vision', 'downstream_task': 'image classification', 'application_domain': 'fashion', 'description': 'Please provide a classification model that categorizes images into one of four clothing categories. The image path, along with its label information, can be found in the files train_labels.csv and test_labels.csv. The model should achieve at least 85% accuracy on the test set and be implemented using PyTorch. Additionally, please include data augmentation techniques and a confusion matrix in the evaluation.', 'performance_metrics': [{'name': 'accuracy', 'value': 0.85}], 'complexity_metrics': []}, 'dataset': [{'name': '', 'modality': ['image', 'text'], 'target_variables': [], 'specification': None, 'description': 'The dataset includes images of clothing items along with their label information in the files train_labels.csv and test_labels.csv.', 'preprocessing': [], 'augmentation': ['data augmentation techniques'], 'visualization': ['confusion matrix'], 'source': 'user-upload', 'task': 'image classification'}], 'model': [{'name': '', 'family': '', 'type': 'neural networks', 'specification': None, 'description': 'A classification model to categorize images into one of four clothing categories, implemented using PyTorch.'}], 'knowledge': [], 'service': {'feature_engineering': [], 'target_device': '', 'deployment_endpoint': '', 'inference_engine': ['PyTorch'], 'demonstration': ''}}
        ```
        
        **Project Plan**
        ### End-to-End Actionable Plan for Fashion Image Classification Using PyTorch

#### 1. Data Preparation
1. **Load Dataset**:
   - Load the image paths and corresponding labels from `train_labels.csv` and `test_labels.csv`.
   - Ensure the dataset includes images of four clothing categories.

2. **Convert Grayscale to RGB**:
   - If the images are in grayscale, convert them to RGB by replicating the single channel across three channels.

3. **Preprocessing**:
   - Normalize the images by scaling pixel values to the range [0, 1].
   - Resize the images to a consistent size, e.g., 224x224 pixels, suitable for model input.

4. **Split Dataset**:
   - Split the training data into training and validation sets (e.g., 80% training, 20% validation) to monitor the model's performance and prevent overfitting.

#### 2. Data Augmentation
1. **Techniques**:
   - Apply data augmentation techniques to increase the diversity and robustness of the training data.
   - Techniques to include: random horizontal flip, normalization, random cropping, random erasing, rotating, shifting, and zooming.

2. **Pipelines**:
   - Develop two preprocessing pipelines:
     - Pipeline 1: Random horizontal flip and normalization.
     - Pipeline 2: Random horizontal flip, normalization, random cropping, and random erasing.

#### 3. Model Architecture
1. **Simple CNN**:
   - Start with a simple Convolutional Neural Network (CNN) model with at least three convolutional layers followed by max pooling layers, and fully connected layers for classification.

2. **Advanced Models**:
   - Consider using more advanced models like ResNet18, EfficientNet, or Vision Transformers (ViTs) which have shown high accuracy in similar tasks.

3. **Hybrid Models**:
   - Explore hybrid models combining CNNs and ViTs to leverage the strengths of both architectures.

#### 4. Training
1. **Optimizer and Loss Function**:
   - Use the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.015 or the Adam optimizer.
   - Use Cross-Entropy Loss for training.

2. **Epochs and Early Stopping**:
   - Train the models for up to 150 epochs.
   - Implement early stopping to prevent overfitting if the validation loss does not improve for a certain number of epochs.

3. **Monitoring**:
   - Monitor validation accuracy and loss curves to ensure the model generalizes well to unseen data.

#### 5. Evaluation
1. **Accuracy**:
   - Ensure the model achieves at least 85% accuracy on the test set.

2. **Confusion Matrix**:
   - Generate a confusion matrix to analyze the model's performance across different clothing categories.

3. **Top-1 Accuracy**:
   - Calculate the top-1 accuracy to evaluate the model's performance.

#### 6. Implementation
1. **Framework**:
   - Leverage PyTorch and its torchvision.transforms.v2 for efficient data augmentation and model training.

2. **Resources**:
   - Utilize resources and code repositories from platforms like Kaggle and PapersWithCode to implement and fine-tune the models effectively.

3. **Deployment**:
   - Although not specified, consider preparing the model for deployment by converting it to a format suitable for the target device and setting up an inference endpoint if required in the future.

#### 7. Visualization and Reporting
1. **Confusion Matrix**:
   - Visualize the confusion matrix to provide insights into the model's classification performance.

2. **Performance Metrics**:
   - Report the accuracy and other relevant performance metrics.

3. **Documentation**:
   - Document the entire process, including data preprocessing steps, model architecture, training parameters, evaluation results, and any challenges encountered.

By following this comprehensive plan, the AI agents should be able to develop a robust image classification model that meets the specified accuracy requirements and provides insightful performance analysis.
        
        **Explanations and Results from the Data Scientist**
        ### Detailed Steps for Data Manipulation and Analysis

#### 1. Dataset Retrieval and Collection

**Retrieve Dataset**:
- **Source**: The dataset is available at `_experiments/datasets/shopee`.
- **Structure**: The directory structure includes `train`, `test`, `train_labels.csv`, and `test_labels.csv`.
- **Loading Data**: Load the image paths and corresponding labels from `train_labels.csv` and `test_labels.csv`. This can be done using libraries like `pandas` to read the CSV files and `os` to navigate through the directories.

**Why**: This step ensures that we have access to the correct dataset and that the paths and labels are correctly mapped for further processing.

#### 2. Checking Dataset Types, Shapes, and Properties

**Data Types**:
- **Image Format**: Verify that images are in the correct format (JPEG, PNG). This can be done using libraries like `PIL` (Python Imaging Library).
- **Labels**: Ensure labels are correctly mapped to images by checking the contents of the CSV files.

**Shapes**:
- **Image Dimensions**: Check the dimensions of the images. If images are in grayscale, convert them to RGB by replicating the single channel across three channels using libraries like `PIL` or `OpenCV`.

**Properties**:
- **Balanced Dataset**: Ensure the dataset is balanced across the four clothing categories by analyzing the label distribution using `pandas`.
- **Integrity Check**: Verify that there are no missing or corrupted files by attempting to open each image file.

**Why**: Ensuring correct data types, shapes, and properties is crucial for accurate model training and prevents errors during preprocessing and augmentation.

#### 3. Preprocessing the Dataset

**Normalization**:
- **Scale Pixel Values**: Normalize pixel values to the range [0, 1] using libraries like `PIL` or `torchvision.transforms`.

**Resizing**:
- **Consistent Size**: Resize all images to a consistent size (e.g., 224x224 pixels) suitable for model input using `PIL` or `torchvision.transforms`.

**Splitting**:
- **Train-Validation Split**: Split the training data into training and validation sets (e.g., 80% training, 20% validation) using `sklearn.model_selection.train_test_split`.

**Why**: Normalization ensures that the pixel values are on a common scale, which helps in faster convergence during training. Resizing ensures that all images have the same dimensions, which is a requirement for most deep learning models. Splitting the data helps in monitoring the model's performance and prevents overfitting.

#### 4. Efficient Data Augmentation

**Techniques**:
- **Random Horizontal Flip**: Randomly flip images horizontally to increase diversity.
- **Normalization**: Normalize images to have zero mean and unit variance.
- **Random Cropping**: Randomly crop images to introduce variability.
- **Random Erasing**: Randomly erase parts of images to make the model robust to occlusions.
- **Rotating**: Rotate images randomly to make the model invariant to orientation.
- **Shifting**: Shift images randomly to simulate different viewpoints.
- **Zooming**: Zoom in and out of images to introduce scale variability.

**Pipelines**:
- **Pipeline 1**: Apply random horizontal flip and normalization.
- **Pipeline 2**: Apply random horizontal flip, normalization, random cropping, and random erasing.

**Why**: Data augmentation increases the diversity and robustness of the training data, which helps in improving the generalization of the model.

#### 5. Extracting and Understanding Underlying Characteristics

**Class Distribution**:
- **Analyze Distribution**: Use `pandas` to analyze the distribution of classes to ensure a balanced dataset.

**Image Quality**:
- **Check Variations**: Assess variations in image quality and resolution using `PIL` or `OpenCV`.

**Data Insights**:
- **Identify Patterns**: Use exploratory data analysis (EDA) techniques to identify any patterns or anomalies in the data that could affect model performance.

**Performance Metrics**:
- **Evaluate Model**: Use metrics such as accuracy and confusion matrix to evaluate the model's performance and understand misclassification patterns. This can be done using `sklearn.metrics`.

**Why**: Understanding the underlying characteristics of the data helps in identifying potential issues that could affect model performance. Analyzing class distribution ensures that the dataset is balanced, which is crucial for training a fair and unbiased model.

### Expected Outcomes and Results

**Quantitative**:
- **Balanced Dataset**: A dataset with an equal number of images for each clothing category.
- **Consistent Image Size**: All images resized to 224x224 pixels.
- **Normalized Pixel Values**: Pixel values scaled to the range [0, 1].
- **Augmented Data**: Increased dataset diversity through augmentation techniques.

**Qualitative**:
- **Data Integrity**: No missing or corrupted files.
- **Improved Model Generalization**: Enhanced model robustness due to diverse and augmented training data.
- **Insightful Analysis**: Clear understanding of class distribution, image quality, and potential data patterns or anomalies.

By following these detailed steps, another data scientist should be able to reproduce the results and build a robust image classification model that meets the specified accuracy requirements and provides insightful performance analysis.
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the modeling and optimization tasks.
        1. How to retrieve or find the high-performance model(s)?
        2. How to optimize the hyperparamters of the retrieved models?
        3. How to extract and understand the underlying characteristics of the dataset(s)?
        4. How to select the top-k models or algorithms based on the given plans?
```

**Model Plan Execution for HPO an model search**
```
As a proficient machine learning research engineer, your task is to explain **detailed** steps for modeling and optimization parts by executing the following machine learning development plan with the goal of finding top-3 candidate models/algorithms.
        
        # Suggested Plan
        ### Summary of the Plan for Fashion Image Classification Using PyTorch

#### User's Requirements
The user needs a classification model that categorizes images into one of four clothing categories. The model should achieve at least 85% accuracy on the test set and be implemented using PyTorch. Data augmentation techniques and a confusion matrix for evaluation are also required.

### Project Plan Overview

#### 1. Data Preparation
- **Load Dataset**: Load image paths and labels from `train_labels.csv` and `test_labels.csv`.
- **Convert Grayscale to RGB**: Convert grayscale images to RGB.
- **Preprocessing**: Normalize pixel values to [0, 1] and resize images to 224x224 pixels.
- **Split Dataset**: Split training data into training (80%) and validation (20%) sets.

#### 2. Data Augmentation
- **Techniques**: Apply random horizontal flip, normalization, random cropping, random erasing, rotating, shifting, and zooming.
- **Pipelines**: Develop two preprocessing pipelines with different augmentation techniques.

#### 3. Model Architecture
- **Simple CNN**: Start with a simple CNN model.
- **Advanced Models**: Consider ResNet18, EfficientNet, or Vision Transformers (ViTs).
- **Hybrid Models**: Explore hybrid models combining CNNs and ViTs.

#### 4. Training
- **Optimizer and Loss Function**: Use SGD or Adam optimizer and Cross-Entropy Loss.
- **Epochs and Early Stopping**: Train for up to 150 epochs with early stopping.
- **Monitoring**: Monitor validation accuracy and loss curves.

#### 5. Evaluation
- **Accuracy**: Ensure at least 85% accuracy on the test set.
- **Confusion Matrix**: Generate a confusion matrix.
- **Top-1 Accuracy**: Calculate top-1 accuracy.

#### 6. Implementation
- **Framework**: Use PyTorch and torchvision.transforms.v2.
- **Resources**: Utilize resources from Kaggle and PapersWithCode.
- **Deployment**: Prepare the model for deployment if needed.

#### 7. Visualization and Reporting
- **Confusion Matrix**: Visualize the confusion matrix.
- **Performance Metrics**: Report accuracy and other metrics.
- **Documentation**: Document the entire process.

### Detailed Steps for Data Manipulation and Analysis

#### 1. Dataset Retrieval and Collection
- **Retrieve Dataset**: Load image paths and labels from `train_labels.csv` and `test_labels.csv`.

#### 2. Checking Dataset Types, Shapes, and Properties
- **Data Types**: Verify image formats and label mappings.
- **Shapes**: Check and convert image dimensions to RGB if needed.
- **Properties**: Ensure balanced dataset and data integrity.

#### 3. Preprocessing the Dataset
- **Normalization**: Scale pixel values to [0, 1].
- **Resizing**: Resize images to 224x224 pixels.
- **Splitting**: Split data into training and validation sets.

#### 4. Efficient Data Augmentation
- **Techniques**: Apply various augmentation techniques.
- **Pipelines**: Create two augmentation pipelines.

#### 5. Extracting and Understanding Underlying Characteristics
- **Class Distribution**: Analyze class distribution.
- **Image Quality**: Assess variations in image quality.
- **Data Insights**: Identify patterns or anomalies.
- **Performance Metrics**: Evaluate model performance using accuracy and confusion matrix.

### Expected Outcomes and Results
- **Quantitative**: Balanced dataset, consistent image size, normalized pixel values, augmented data.
- **Qualitative**: Data integrity, improved model generalization, insightful analysis.

### Responsibilities Fulfillment

#### 1. How to Retrieve or Find High-Performance Model(s)?
- **Initial Models**: Start with simple CNN architectures.
- **Advanced Models**: Experiment with ResNet18, EfficientNet, and Vision Transformers.
- **Hybrid Models**: Explore combinations of CNNs and ViTs.
- **Resources**: Utilize code repositories from Kaggle and PapersWithCode for implementation and fine-tuning.

#### 2. How to Optimize the Hyperparameters of the Retrieved Models?
- **Optimizers**: Use SGD with a learning rate of 0.015 or Adam optimizer.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and augmentation techniques.
- **Early Stopping**: Implement early stopping based on validation loss to prevent overfitting.
- **Monitoring**: Continuously monitor validation accuracy and loss curves.

#### 3. How to Extract and Understand the Underlying Characteristics of the Dataset(s)?
- **Class Distribution**: Use `pandas` to analyze the distribution of classes.
- **Image Quality**: Assess image quality and resolution using `PIL` or `OpenCV`.
- **Data Insights**: Perform exploratory data analysis (EDA) to identify patterns or anomalies.
- **Performance Metrics**: Use `sklearn.metrics` to evaluate model performance and generate confusion matrices.

#### 4. How to Select the Top-k Models or Algorithms Based on the Given Plans?
- **Evaluation Metrics**: Use accuracy and confusion matrix to evaluate models.
- **Top-k Selection**: Rank models based on their performance metrics and select the top-k models that meet or exceed the 85% accuracy requirement.
- **Documentation**: Document the performance of each model, including accuracy, confusion matrix, and any other relevant metrics.

By following this comprehensive plan, you will be able to retrieve high-performance models, optimize their hyperparameters, understand the dataset's characteristics, and select the top-k models based on their performance.
        
        # Available Model Source
        There is no available source for model loading. Please directly build the model by strictly following the user's requirements.
                
        Make sure that your explanation for finding the top-3 high-performance models or algorithms follows these instructions:
        - All of your explanations must be self-contained without using any placeholder to ensure that other machine learning research engineers can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or find the top-3 well-performing models/algorithms.
        - Include how to optimize the hyperparamters of the candidate models or algorithms by clearly specifying which hyperparamters are optimized in detail.
        - Corresponding to each hyperparamter, explicitly include the actual numerical value that you think it is the optimal value for the given dataset and machine learning task.
        - Include how to extract and understand the characteristics of the candidate models or algorithms, such as their computation complexity, memory usage, and inference latency. This part is not related to visualization and interpretability.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.
        Make sure to focus only on the modeling part as it is your expertise. Do not conduct or perform anything regarding data manipulation or analysis.
        After complete the explanations, explicitly specify the names and (expected) quantitative performance using relevant numerical performance and complexity metrics (e.g., number of parameters, FLOPs, model size, training time, inference speed, and so on) of the three candidate models/algorithms potentially to be the optimal model below.
        Do not use any placeholder for the quantitative performance. If you do not know the exact values, please use the knowledge and expertise you have to estimate those performance and complexity values.
```

**Plan Execution Result with candidate ranking**
```
### Detailed Steps for Modeling and Optimization

#### 1. Retrieving High-Performance Models

**Initial Models:**
- **Simple CNN**: Start with a basic Convolutional Neural Network (CNN) architecture. This model will serve as a baseline to understand the dataset and the task.

**Advanced Models:**
- **ResNet18**: A well-known deep learning model that has shown excellent performance in image classification tasks. It uses residual connections to ease the training of deeper networks.
- **EfficientNet**: Known for its efficiency in terms of parameter count and FLOPs while maintaining high accuracy. EfficientNet-B0 can be a good starting point.
- **Vision Transformers (ViTs)**: These models have gained popularity for their performance in image classification tasks. A small variant like ViT-Small can be considered.

**Hybrid Models:**
- **CNN + ViT**: Combining the feature extraction capabilities of CNNs with the global context understanding of ViTs. This hybrid approach can potentially leverage the strengths of both architectures.

#### 2. Optimizing Hyperparameters

**Optimizers:**
- **SGD**: Stochastic Gradient Descent with a learning rate of 0.015.
- **Adam**: Adaptive Moment Estimation with a learning rate of 0.001.

**Hyperparameter Tuning:**
- **Learning Rate**: Experiment with learning rates (0.001, 0.005, 0.01, 0.015) to find the optimal rate for convergence.
- **Batch Size**: Test batch sizes (16, 32, 64) to balance between memory usage and training speed.
- **Data Augmentation Techniques**: Apply different augmentation techniques and observe their impact on validation accuracy.
- **Number of Epochs**: Train for up to 150 epochs with early stopping based on validation loss to prevent overfitting.
- **Weight Decay**: Regularization parameter for SGD, set to 0.0005.

**Optimal Values:**
- **Learning Rate**: 0.001 for Adam, 0.015 for SGD.
- **Batch Size**: 32.
- **Weight Decay**: 0.0005.
- **Number of Epochs**: Up to 150 with early stopping.

#### 3. Extracting and Understanding Characteristics

**Computation Complexity:**
- **Number of Parameters**: Calculate the total number of trainable parameters in each model.
- **FLOPs (Floating Point Operations per Second)**: Measure the computational complexity of each model.

**Memory Usage:**
- **Model Size**: Determine the size of the model in memory (MB).
- **Training Memory Usage**: Monitor GPU memory usage during training.

**Inference Latency:**
- **Inference Speed**: Measure the time taken for a single forward pass (in milliseconds).

**Why Each Step is Essential:**
- **Model Selection**: Choosing a variety of models ensures that we explore different architectures and their strengths.
- **Hyperparameter Tuning**: Fine-tuning hyperparameters is crucial for achieving optimal performance and ensuring the model generalizes well.
- **Characteristic Analysis**: Understanding the computational and memory requirements helps in selecting models that are not only accurate but also efficient and deployable.

### Candidate Models and Expected Performance

**1. ResNet18**
- **Accuracy**: Expected to achieve around 88% on the test set.
- **Number of Parameters**: Approximately 11.7 million.
- **FLOPs**: Around 1.8 billion.
- **Model Size**: Approximately 44 MB.
- **Training Time**: Approximately 2 hours on a single GPU.
- **Inference Speed**: Around 10 milliseconds per image.

**2. EfficientNet-B0**
- **Accuracy**: Expected to achieve around 87% on the test set.
- **Number of Parameters**: Approximately 5.3 million.
- **FLOPs**: Around 390 million.
- **Model Size**: Approximately 20 MB.
- **Training Time**: Approximately 3 hours on a single GPU.
- **Inference Speed**: Around 8 milliseconds per image.

**3. Vision Transformer (ViT-Small)**
- **Accuracy**: Expected to achieve around 86% on the test set.
- **Number of Parameters**: Approximately 22 million.
- **FLOPs**: Around 4.6 billion.
- **Model Size**: Approximately 85 MB.
- **Training Time**: Approximately 4 hours on a single GPU.
- **Inference Speed**: Around 15 milliseconds per image.

By following these detailed steps and focusing on the modeling and optimization parts, you will be able to identify and select the top-3 high-performance models for the given image classification task.
```
