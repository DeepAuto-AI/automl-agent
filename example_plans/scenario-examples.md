# Examples of Multi-Scenario Planning

### Example 1: Image Classification (Constraint-Free)
> I need a very accurate model to classify images in the Butterfly Image Classification dataset into their respective categories. The dataset has been uploaded with its label information in the labels.csv file.

**Scenario 1**
```
### End-to-End Actionable Plan for Butterfly Image Classification

**Objective**: Develop a highly accurate neural network model to classify images in the Butterfly Image Classification dataset into their respective categories.

#### 1. Data Collection and Preprocessing
1. **Dataset Loading**:
   - Load the Butterfly Image Classification dataset and the associated labels from `labels.csv`.
   - Verify the integrity of the dataset, ensuring all images have corresponding labels.

2. **Data Preprocessing**:
   - **Image Resizing**: Resize all images to a uniform size (e.g., 224x224 pixels) to ensure consistency.
   - **Normalization**: Normalize pixel values to the range [0, 1] or standardize to have zero mean and unit variance.

3. **Data Augmentation**:
   - Apply data augmentation techniques such as cropping, horizontal and vertical flips, rescaling, shear, zoom, and rotation to increase dataset variability.
   - Utilize libraries like TensorFlow's `ImageDataGenerator` or PyTorch's `transforms` for implementation.

#### 2. Model Selection and Architecture
1. **Transfer Learning**:
   - Leverage pre-trained convolutional neural networks (CNNs) such as VGG16, VGG19, MobileNet, Xception, ResNet50, and InceptionV3.
   - Fine-tune these pre-trained models on the Butterfly Image Classification dataset to utilize their learned features.

2. **Model Architecture**:
   - Start with InceptionV3, known for high accuracy in similar tasks.
   - Explore other models like ResNet, DenseNet, EfficientNet, and Vision Transformers.
   - Consider combining CNNs with Transformer-based architectures to capture both local and global features.

3. **Ensemble Methods**:
   - Implement ensemble methods by combining predictions from multiple models to improve overall classification performance.
   - Use techniques like bagging or stacking to enhance model robustness.

#### 3. Model Training
1. **Dataset Splitting**:
   - Split the dataset into training, validation, and testing sets (e.g., 80% training and 20% validation).

2. **Training Configuration**:
   - Use categorical cross-entropy as the loss function.
   - Select optimizers like SGD or RMSprop.
   - Set up callbacks for early stopping and learning rate reduction on plateau to prevent overfitting.

3. **Hyperparameter Tuning**:
   - Perform hyperparameter tuning using libraries like PyCaret or Optuna to find the optimal configuration.
   - Utilize techniques like grid search or random search for hyperparameter optimization.

#### 4. Model Evaluation and Validation
1. **Performance Metrics**:
   - Regularly monitor accuracy as the primary performance metric.
   - Additionally, track precision, recall, and F1-score to ensure balanced performance across all classes.

2. **Cross-Validation**:
   - Employ k-fold cross-validation to assess the model's robustness and generalization ability.

3. **Handling Class Imbalance**:
   - Address class imbalance through weighted loss functions or oversampling techniques to ensure fair performance across all categories.

#### 5. Interpretability and Debugging
1. **Model Interpretability**:
   - Use interpretability methods like Grad-CAM or LIME to visualize which parts of the images influence the model's predictions.
   - Ensure the model focuses on relevant biological features of butterflies.

#### 6. Reproducibility and Documentation
1. **Reproducibility**:
   - Set random seeds for all operations to ensure reproducibility of results.
   - Document the entire process, including data splits, model configurations, and training parameters.

2. **Model Deployment**:
   - Prepare the model for deployment by exporting it in a suitable format (e.g., TensorFlow SavedModel, ONNX).
   - Set up an inference engine and deployment endpoint to make predictions on new data.

3. **Demonstration**:
   - Provide a demonstration of the model's capabilities by creating a user-friendly interface or visualization to showcase its performance.

By following this comprehensive plan, AI agents can develop a robust and accurate neural network model tailored for classifying butterfly images in the given dataset.
```

**Scenario 2**
```
## End-to-End Actionable Plan for Butterfly Image Classification

### 1. Data Collection
- **Dataset Name**: Butterfly Image Classification
- **Source**: User-uploaded
- **Description**: A dataset for classifying images of butterflies into their respective categories.
- **Label Information**: Available in `labels.csv` file.

### 2. Data Preprocessing
- **Image Preprocessing**:
  - **Resize Images**: Resize all images to a consistent size (e.g., 224x224 pixels) to match the input size required by pre-trained models.
  - **Normalization**: Normalize pixel values to the range [0, 1] or use mean subtraction based on the pre-trained model's requirements.
  - **Handling High-Resolution and Illumination Variations**: Apply techniques to standardize illumination and handle high-resolution images.

### 3. Data Augmentation
- **Techniques**:
  - **Cropping**: Randomly crop sections of the images.
  - **Flipping**: Apply horizontal and vertical flips.
  - **Rescaling**: Rescale images by a factor.
  - **Shearing**: Apply shear transformations.
  - **Zooming**: Apply random zoom.
  - **Rotation**: Rotate images by random angles.
- **Libraries**: Use TensorFlow's `ImageDataGenerator` or PyTorch's `transforms`.

### 4. Dataset Splitting
- **Training Set**: 80% of the dataset
- **Validation Set**: 20% of the dataset

### 5. Model Selection and Architecture
- **Transfer Learning**:
  - **Pre-trained Models**: Utilize models such as VGG16, VGG19, MobileNet, Xception, ResNet50, and InceptionV3.
  - **Fine-tuning**: Fine-tune these models on the Butterfly Image Classification dataset.
- **Model Architecture**:
  - **Primary Model**: Start with InceptionV3 due to its high performance in similar tasks.
  - **Alternative Models**: Evaluate ResNet, DenseNet, EfficientNet, and Vision Transformers.
  - **Combined Architectures**: Explore combining CNNs with Transformer-based architectures to capture both local and global features.

### 6. Handling Class Imbalance
- **Techniques**:
  - **Weighted Loss Functions**: Assign higher weights to underrepresented classes.
  - **Oversampling**: Increase the number of samples for underrepresented classes.

### 7. Model Training
- **Loss Function**: Use categorical cross-entropy.
- **Optimizers**: Start with SGD or RMSprop.
- **Hyperparameter Tuning**:
  - **Libraries**: Use PyCaret or Optuna for hyperparameter optimization.
  - **Parameters**: Tune learning rate, batch size, number of epochs, etc.
- **Cross-Validation**: Use k-fold cross-validation to ensure model robustness and generalization.

### 8. Performance Evaluation
- **Metrics**:
  - **Accuracy**: Primary metric for evaluation.
  - **Precision, Recall, F1-Score**: Monitor these metrics to ensure balanced performance across classes.
- **Evaluation Techniques**:
  - **Confusion Matrix**: Visualize to understand misclassifications.
  - **ROC Curve**: Evaluate model performance across different thresholds.

### 9. Ensemble Methods and Knowledge Distillation
- **Ensemble Methods**: Combine predictions from multiple models (e.g., patch-wise and image-wise classifiers).
- **Knowledge Distillation**: Transfer knowledge from larger teacher models to smaller student models to create compact yet powerful models.

### 10. Interpretability and Debugging
- **Techniques**:
  - **Grad-CAM**: Visualize which parts of the images influence the model's predictions.
  - **LIME**: Provide explanations for individual predictions.
- **Purpose**: Ensure the model focuses on relevant biological features and understand model behavior.

### 11. Reproducibility
- **Random Seeds**: Set random seeds for all processes to ensure reproducibility.
- **Documentation**: Document data splits, model configurations, and training parameters.

### 12. Deployment
- **Target Device**: Specify the target device for deployment (e.g., cloud, edge device).
- **Deployment Endpoint**: Define the endpoint for model deployment.
- **Inference Engine**: Choose an appropriate inference engine (e.g., TensorFlow Serving, TorchServe).
- **Demonstration**: Prepare a demonstration of the model's capabilities and performance.

By following this comprehensive plan, AI agents can effectively build, train, and deploy a highly accurate neural network model tailored for classifying butterfly images in the specified dataset.
```

**Scenario 3**
```
### End-to-End Actionable Plan for Butterfly Image Classification using Neural Networks

#### 1. **Project Initialization**

**Objective**: Build a highly accurate image classification model for the Butterfly Image Classification dataset in the biology domain.

**User Expertise**: Medium

---

#### 2. **Data Collection and Preprocessing**

**Dataset Source**: User-uploaded Butterfly Image Classification dataset with label information in `labels.csv`.

**Steps**:

1. **Load Dataset**:
   - Load images and labels from the specified dataset and `labels.csv`.

2. **Data Cleaning**:
   - Ensure all images are correctly labeled.
   - Remove any corrupted or unusable images.

3. **Data Preprocessing**:
   - Resize images to a consistent size (e.g., 224x224 pixels).
   - Normalize pixel values to the range [0, 1].

4. **Data Augmentation**:
   - Apply augmentation techniques such as cropping, horizontal and vertical flips, rescaling, shear, zoom, and rotation to increase dataset size and variability.
   - Use libraries like TensorFlow's `ImageDataGenerator` or PyTorch's `transforms`.

---

#### 3. **Model Development**

**Model Type**: Neural Networks

**Steps**:

1. **Leverage Transfer Learning**:
   - Use pre-trained CNNs such as VGG16, VGG19, MobileNet, Xception, ResNet50, and InceptionV3.
   - Fine-tune these models on the Butterfly Image Classification dataset.

2. **Model Selection and Architecture**:
   - Start with InceptionV3 due to its high accuracy in similar tasks.
   - Experiment with other models like ResNet, DenseNet, EfficientNet, and Vision Transformers.
   - Consider combining CNNs with Transformer-based architectures for improved performance.

3. **Ensemble Methods and Knowledge Distillation**:
   - Implement ensemble methods by combining patch-wise and image-wise classifiers.
   - Use knowledge distillation to transfer knowledge from larger models to smaller, more efficient models.

---

#### 4. **Training and Evaluation**

**Steps**:

1. **Dataset Splitting**:
   - Split the dataset into training (80%), validation (10%), and testing (10%) sets.

2. **Model Training**:
   - Use categorical cross-entropy as the loss function.
   - Utilize optimizers such as SGD or RMSprop.
   - Implement early stopping and learning rate reduction on plateau to prevent overfitting.

3. **Performance Monitoring**:
   - Regularly monitor metrics such as accuracy, precision, recall, and F1-score.
   - Use validation data to fine-tune hyperparameters.

4. **Handling Class Imbalance**:
   - Address class imbalance using weighted loss functions or oversampling techniques.

5. **Hyperparameter Tuning and Cross-Validation**:
   - Perform hyperparameter tuning with libraries like PyCaret or Optuna.
   - Use cross-validation to ensure model robustness and generalization.

---

#### 5. **Model Interpretability and Debugging**

**Steps**:

1. **Interpretability**:
   - Use Grad-CAM or LIME to visualize important image regions influencing model predictions.
   - Ensure the model focuses on relevant biological features.

2. **Debugging**:
   - Analyze misclassified images to understand model limitations.
   - Iterate on model improvements based on these insights.

---

#### 6. **Reproducibility and Documentation**

**Steps**:

1. **Reproducibility**:
   - Set random seeds for all processes to ensure reproducibility.
   - Document all steps, including data splits, model configurations, and training parameters.

2. **Version Control**:
   - Use version control systems (e.g., Git) to track changes and collaborate efficiently.

---

#### 7. **Deployment and Inference**

**Steps**:

1. **Model Export**:
   - Export the trained model in a suitable format (e.g., TensorFlow SavedModel, ONNX).

2. **Deployment**:
   - Choose a deployment platform (e.g., AWS SageMaker, Google AI Platform).
   - Set up the deployment endpoint for real-time inference.

3. **Inference Engine**:
   - Optimize the model for the target device (e.g., mobile, edge devices) if necessary.
   - Implement efficient inference pipelines to handle incoming image data.

---

#### 8. **Demonstration and User Feedback**

**Steps**:

1. **Demonstration**:
   - Create a demo application to showcase the model's capabilities.
   - Provide a user-friendly interface for uploading images and viewing classification results.

2. **User Feedback**:
   - Collect feedback from users to identify areas for improvement.
   - Implement iterative updates based on user insights.

---

By following this detailed plan, AI agents can build a robust and accurate neural network model for classifying butterfly images in the biology domain, ensuring high performance and user satisfaction.
```



### Example 2: Image Classification (Constraint-Aware)
> I need a highly accurate machine learning model developed to classify images within the Butterfly Image Classification dataset into their correct species categories. The dataset has been uploaded with its label information in the labels.csv file. Please use a convolutional neural network (CNN) architecture for this task, leveraging transfer learning from a pre-trained ResNet-50 model to improve accuracy. Optimize the model using cross-validation on the training split to fine-tune hyperparameters, and aim for an accuracy of at least 0.95 on the test split. Provide the final trained model, a detailed report of the training process, hyperparameter settings, accuracy metrics, and a confusion matrix to evaluate performance across different categories.

**Scenario 1**
```
### End-to-End Actionable Plan for Butterfly Species Classification Using CNN with ResNet-50 Transfer Learning

#### 1. **Project Initialization**
- **Objective**: Develop a highly accurate machine learning model for classifying butterfly species using the Butterfly Image Classification dataset.
- **Performance Metric**: Achieve at least 0.95 accuracy on the test split.
- **Model Architecture**: Convolutional Neural Network (CNN) leveraging transfer learning from a pre-trained ResNet-50 model.

#### 2. **Data Preparation**
- **Dataset**: Butterfly Image Classification dataset with label information in labels.csv.
- **Steps**:
  1. **Data Splitting**:
     - Split the dataset into training (70%), validation (15%), and test (15%) sets.
     - Ensure that the validation and test sets are representative of the entire dataset.
  2. **Data Augmentation**:
     - Apply techniques such as rotations, horizontal and vertical flips, zooms, and random crops to increase the diversity of the training data.
     - Implement augmentation in real-time during the training process to save storage space.

#### 3. **Model Architecture**
- **Base Model**: Use ResNet-50 pre-trained on the ImageNet dataset.
- **Modifications**:
  1. **Replace Final Layer**:
     - Replace the final fully connected layer of ResNet-50 with a new layer that matches the number of butterfly species.
     - Initialize this new layer randomly while keeping the pre-trained layers frozen initially.
  2. **Attention Mechanisms** (Optional):
     - Integrate squeeze-and-excitation (SE) and spatial attention (SA) modules to enhance feature extraction.

#### 4. **Training Process**
- **Steps**:
  1. **Initial Training**:
     - Train the modified ResNet-50 model with the new classification layer.
     - Use a lower learning rate for fine-tuning the pre-trained layers after initial training.
  2. **Cross-Validation**:
     - Implement cross-validation on the training split to fine-tune hyperparameters such as learning rate, batch size, and number of epochs.
  3. **Early Stopping**:
     - Monitor the validation loss to implement early stopping when performance stops improving to avoid overfitting.

#### 5. **Optimization Techniques**
- **Hyperparameter Tuning**:
  - Use techniques like grid search or Bayesian optimization to find the best hyperparameters.
- **Advanced Techniques** (Optional):
  - Consider knowledge distillation (KD) to transfer knowledge from a larger model to a smaller one.
  - Explore ensemble methods combining CNN with transformer architectures for improved accuracy.
  - Use contrastive learning (e.g., SimCLR) and semi-supervised learning (e.g., FixMatch) to enhance model performance.

#### 6. **Evaluation Metrics**
- **Metrics**:
  1. **Accuracy**: Ensure the model achieves at least 0.95 accuracy on the test set.
  2. **Precision, Recall, and F1-Score**: Evaluate these metrics to get a comprehensive understanding of model performance.
  3. **Confusion Matrix**: Generate a confusion matrix to visualize performance across different butterfly species.
  4. **Visual Explanations**: Use Grad-CAM to provide visual explanations of model predictions.

#### 7. **Documentation and Reporting**
- **Documentation**:
  1. **Data Preparation**: Document the steps taken for data splitting and augmentation.
  2. **Model Architecture**: Describe the modifications made to the ResNet-50 model.
  3. **Training Process**: Detail the training procedure, including hyperparameter settings and early stopping criteria.
  4. **Evaluation**: Report the evaluation metrics and provide a comprehensive analysis of the results.
- **Report**:
  - Include the final trained model, accuracy metrics, confusion matrix, and visual explanations in the report.
  - Ensure the report is detailed enough for stakeholders to understand the model's performance and decision-making process.

#### 8. **Deployment and Demonstration**
- **Deployment**:
  - Package the trained model and necessary preprocessing steps for deployment.
  - Provide instructions for deploying the model on the target device or cloud platform.
- **Demonstration**:
  - Create a demonstration script to showcase the model's capabilities and performance on sample images.

By following this detailed plan, the AI agents can successfully develop, train, and evaluate a highly accurate butterfly species classification model using state-of-the-art techniques and methodologies.
```

**Scenario 2**
```
### End-to-End Actionable Plan for Butterfly Species Classification

#### 1. **Project Initialization**

- **Objective**: Develop a highly accurate machine learning model for classifying images within the Butterfly Image Classification dataset into their correct species categories.
- **Target Accuracy**: At least 0.95 on the test split.
- **Model Architecture**: Convolutional Neural Network (CNN) leveraging transfer learning from a pre-trained ResNet-50 model.

#### 2. **Data Preparation**

1. **Dataset Loading**:
   - Load the Butterfly Image Classification dataset and the accompanying `labels.csv` file.

2. **Data Splitting**:
   - Split the dataset into training (70%), validation (15%), and test (15%) sets. Ensure that the splits are representative of the entire dataset.

3. **Data Augmentation**:
   - Apply data augmentation techniques such as rotations, flips, zooms, and random crops to increase the diversity of the training data and improve model robustness.

4. **Data Normalization**:
   - Normalize the images to have pixel values in the range of [0, 1] or standardized based on the ImageNet dataset statistics (mean and standard deviation).

#### 3. **Model Architecture**

1. **Base Model**:
   - Load the pre-trained ResNet-50 model without the top classification layer.

2. **Custom Classification Layer**:
   - Add a new fully connected layer that matches the number of butterfly species in the dataset. This layer will be trained from scratch.

3. **Attention Mechanisms (Optional)**:
   - Integrate advanced attention mechanisms such as squeeze-and-excitation (SE) and spatial attention (SA) modules to enhance feature extraction and improve classification accuracy.

#### 4. **Training Process**

1. **Hyperparameter Tuning**:
   - Use cross-validation to fine-tune hyperparameters such as learning rate, batch size, and number of epochs.

2. **Training Loop**:
   - Train the model using the training set while monitoring performance on the validation set.
   - Implement early stopping to halt training when the validation performance stops improving.

3. **Optimization Techniques**:
   - Consider techniques such as knowledge distillation (KD) and ensemble models combining CNN and transformer architectures to further boost accuracy and robustness.
   - Explore contrastive learning (e.g., SimCLR) and semi-supervised learning (e.g., FixMatch) to enhance model accuracy and generalization.

#### 5. **Model Evaluation**

1. **Performance Metrics**:
   - Evaluate the model on the test set to ensure it meets the desired accuracy threshold of at least 0.95.
   - Use metrics such as accuracy, precision, recall, and F1-score to assess performance comprehensively.

2. **Confusion Matrix**:
   - Generate a confusion matrix to visualize the model's performance across different butterfly species and identify any specific categories where the model may be underperforming.

3. **Visual Explanations**:
   - Utilize techniques like Grad-CAM to provide visual explanations of model predictions. This helps highlight areas of images that contributed most to the classification decision and can be useful for model interpretability.

#### 6. **Model Deployment**

1. **Model Saving**:
   - Save the final trained model in a format suitable for deployment (e.g., TensorFlow SavedModel, PyTorch model file).

2. **Documentation and Reporting**:
   - Document the entire process, including data preparation, model architecture, training process, hyperparameter settings, and evaluation metrics.
   - Provide a detailed report that includes the trained model, accuracy metrics, and a confusion matrix to evaluate performance across different butterfly species categories.

#### 7. **Final Deliverables**

1. **Trained Model**:
   - Provide the final trained model ready for deployment.

2. **Detailed Report**:
   - Include a comprehensive report detailing:
     - Data preparation steps
     - Model architecture and modifications
     - Training process and hyperparameter settings
     - Evaluation metrics and performance analysis
     - Visual explanations of model predictions (Grad-CAM)
     - Confusion matrix

By following this plan, AI agents can develop a robust and accurate butterfly species classification model that leverages the strengths of transfer learning with ResNet-50 and is fine-tuned to achieve high accuracy on the specific dataset.
```

**Scenario 3**
```
### End-to-End Actionable Plan

#### 1. **Project Initialization**

**Objective**: Develop a highly accurate machine learning model for classifying butterfly species using the Butterfly Image Classification dataset. The target accuracy is at least 0.95 on the test split.

**Model**: Convolutional Neural Network (CNN) leveraging transfer learning from a pre-trained ResNet-50 model.

**Performance Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

---

#### 2. **Data Preparation**

**a. Data Splitting**:
- Split the dataset into training (70%), validation (15%), and test (15%) sets.
- Ensure that the validation and test sets are representative of the entire dataset to avoid overfitting.

**b. Data Augmentation**:
- Apply data augmentation techniques to increase the diversity of the training data:
  - Rotations
  - Flips (horizontal and vertical)
  - Zooms
  - Random crops
  - Color jittering

**c. Data Preprocessing**:
- Normalize the images to have pixel values between 0 and 1.
- Resize the images to 224x224 pixels to match the input size of the ResNet-50 model.

---

#### 3. **Model Architecture**

**a. Base Model**:
- Load the pre-trained ResNet-50 model, excluding the top classification layer.

**b. Custom Layers**:
- Add a Global Average Pooling (GAP) layer.
- Add a Dense layer with units equal to the number of butterfly species (output classes) and a softmax activation function.

**c. Attention Mechanisms**:
- Integrate Squeeze-and-Excitation (SE) blocks to enhance feature extraction.
- Optionally, add Spatial Attention (SA) modules to further improve classification accuracy.

---

#### 4. **Training Process**

**a. Hyperparameters**:
- Use cross-validation to fine-tune hyperparameters such as learning rate, batch size, and number of epochs.
- Suggested ranges:
  - Learning rate: [1e-4, 1e-3]
  - Batch size: [16, 32, 64]
  - Epochs: [20, 50, 100]

**b. Training Strategy**:
- Train the model using the training set while monitoring performance on the validation set.
- Implement early stopping to halt training when the validation performance stops improving for a predefined number of epochs (e.g., 5 epochs).

**c. Optimization Techniques**:
- Utilize Adam optimizer with an initial learning rate of 1e-4.
- Implement learning rate decay to reduce the learning rate by a factor of 0.1 if the validation loss plateaus.

---

#### 5. **Advanced Techniques**

**a. Knowledge Distillation**:
- Train a smaller student model using the predictions of the larger ResNet-50 model to improve inference speed while maintaining accuracy.

**b. Ensemble Models**:
- Combine predictions from multiple models (e.g., CNN and transformer-based models) to enhance robustness and accuracy.

**c. Semi-Supervised Learning**:
- Apply techniques like FixMatch to leverage both labeled and unlabeled data for improved generalization.

---

#### 6. **Evaluation Metrics**

**a. Accuracy**:
- Calculate the overall accuracy on the test set.

**b. Precision, Recall, F1-score**:
- Compute these metrics for each class to assess performance comprehensively.

**c. Confusion Matrix**:
- Generate a confusion matrix to visualize the model's performance across different butterfly species.

**d. Visual Explanations**:
- Use Grad-CAM to provide visual explanations of model predictions, highlighting areas of images that contributed most to the classification decision.

---

#### 7. **Documentation and Reporting**

**a. Process Documentation**:
- Document the entire process including data preparation, model architecture, training process, and hyperparameter settings.

**b. Final Report**:
- Provide a detailed report that includes:
  - Description of the dataset and preprocessing steps.
  - Model architecture and training process.
  - Hyperparameter settings and cross-validation results.
  - Accuracy metrics, precision, recall, F1-score, and confusion matrix.
  - Visual explanations using Grad-CAM.
  - The final trained model.

---

By following this comprehensive plan, AI agents can develop a robust and accurate butterfly species classification model leveraging the strengths of transfer learning with ResNet-50 and fine-tune it to achieve high accuracy on the specific dataset.
```

### Example 3: Tabular Regression (Constraint-Free)
> The dataset presented here (the Crab Age Dataset) comprises a lot of both categorical and numerical features. Pleae split the dataset into three parts of train, valid and test. Your task is to predict the age item. The evaluation metric is the RMSLE (root mean squared log error).
Now train a regression model to get a good performance on this task.


**Scenario 1**
```
### End-to-End Actionable Plan for AI Agents

#### 1. **Data Collection and Understanding**
- **Dataset Source**: Directly retrieve the Crab Age Dataset from the specified source.
- **Dataset Description**: The dataset contains both categorical and numerical features, with the target variable being 'age'.

#### 2. **Data Preprocessing**
- **Standardization/Normalization**: 
  - Standardize or normalize numerical features to improve model accuracy.
- **Encoding Categorical Features**: 
  - Apply one-hot encoding for categorical variables to make them suitable for machine learning algorithms.
- **Handling Missing Values**: 
  - Identify and handle any missing values in the dataset appropriately (e.g., imputation with mean/median for numerical features, mode for categorical features).

#### 3. **Feature Engineering**
- **Aggregation**:
  - Aggregate features where necessary to capture underlying trends.
- **Smoothing**:
  - Apply smoothing techniques to reduce noise in the data.
- **Polynomial Features**:
  - Create polynomial features and interaction terms to capture complex relationships in the data.

#### 4. **Model Selection**
- **Ensemble Methods**:
  - Implement ensemble methods such as Voting Regressor and Bagging to balance out individual model weaknesses and improve stability and accuracy.
- **Gradient Boosting**:
  - Utilize gradient boosting techniques like XGBoost, LightGBM, and CatBoost, which are highly effective for tabular data and can handle both categorical and numerical features efficiently.

#### 5. **Model Training**
- **Cross-Validation**:
  - Employ k-fold cross-validation to evaluate model performance. This technique helps assess the model's ability to generalize to an independent dataset and reduces the risk of overfitting.
- **Hyperparameter Tuning**:
  - Use hyperparameter tuning techniques such as grid search or random search to optimize model performance. This step is crucial for finding the best parameters that enhance the model's predictive accuracy.
- **Regularization**:
  - Apply regularization techniques like L1 (lasso) and L2 (ridge) to prevent overfitting and handle multicollinearity.

#### 6. **Model Evaluation**
- **Evaluation Metric**:
  - Focus on optimizing the Root Mean Squared Log Error (RMSLE) as the primary evaluation metric.
- **Performance Metrics**:
  - Calculate and report RMSLE on the validation set to evaluate model performance.

#### 7. **Model Interpretation**
- **Feature Importance**:
  - Utilize tools like LIME (Local Interpretable Model-agnostic Explanations) to interpret the predictions of complex models. Understanding model predictions can provide insights into feature importance and model behavior.

#### 8. **Model Improvement**
- **Pseudo Labeling**:
  - Consider pseudo labeling to enhance the training dataset. This involves adding confident predictions from the test set to the training set, retraining the model, and improving its performance.

#### 9. **Deployment Preparation**
- **Target Device**:
  - Prepare the model for deployment on the specified target device.
- **Deployment Endpoint**:
  - Set up the deployment endpoint for model inference.
- **Inference Engine**:
  - Ensure the inference engine is ready to handle model predictions in real-time or batch processing.

#### 10. **Demonstration**
- **Demonstration Setup**:
  - Prepare a demonstration to showcase the model's performance and capabilities in predicting the age of crabs using the Crab Age Dataset.

### Summary
This plan provides a comprehensive, step-by-step guide for AI agents to build a high-performance regression model tailored for predicting the age of crabs using the Crab Age Dataset. The plan covers data preprocessing, feature engineering, model selection, training, evaluation, interpretation, and deployment preparation, ensuring an end-to-end machine learning pipeline.
```

**Scenario 2**
```
### End-to-End Actionable Plan for Building a Tabular Regression Model to Predict Crab Age

#### 1. Data Collection and Preparation
- **Data Source**: Retrieve the "Crab Age Dataset" from the specified source. Ensure the dataset includes both categorical and numerical features.
- **Data Loading**: Load the dataset into a suitable data structure (e.g., pandas DataFrame) for processing.

#### 2. Data Preprocessing
- **Handling Missing Values**: Identify and handle any missing values in the dataset. Use imputation techniques such as mean/mode/median imputation or more advanced methods like KNN imputation.
- **Standardization/Normalization**: Standardize or normalize numerical features to ensure they have a mean of 0 and a standard deviation of 1, or scale them to a range [0, 1].
- **Encoding Categorical Features**: Apply one-hot encoding or target encoding to transform categorical variables into a numerical format suitable for machine learning algorithms.
- **Feature Scaling**: Ensure all features are on a similar scale to improve model performance.

#### 3. Feature Engineering
- **Aggregation**: Create aggregated features that summarize information from existing features.
- **Smoothing**: Apply smoothing techniques to reduce noise and improve the signal in the data.
- **Polynomial Features**: Generate polynomial features and interaction terms to capture non-linear relationships.
- **Feature Selection**: Use techniques like mutual information, correlation analysis, or feature importance from a preliminary model to select relevant features.

#### 4. Model Selection and Training
- **Model Choice**:
  - **Ensemble Methods**: Implement ensemble methods such as Voting Regressor and Bagging.
  - **Gradient Boosting**: Use gradient boosting models like XGBoost, LightGBM, and CatBoost.
- **Cross-Validation**: Employ k-fold cross-validation (e.g., k=5 or k=10) to assess model performance and ensure robustness.
- **Hyperparameter Tuning**: Use grid search or random search to find the optimal hyperparameters for each model.
- **Regularization**: Implement L1 (lasso) and L2 (ridge) regularization to control model complexity and prevent overfitting.

#### 5. Model Evaluation
- **Evaluation Metric**: Focus on optimizing the Root Mean Squared Log Error (RMSLE) as it is the primary evaluation metric.
- **Performance Metrics**: Track RMSLE during cross-validation and on a hold-out test set to ensure the model generalizes well.

#### 6. Model Interpretation
- **Interpretability Tools**: Use tools like LIME (Local Interpretable Model-agnostic Explanations) to interpret the model's predictions and understand feature importance.

#### 7. Pseudo Labeling (Optional)
- **Enhancing Training Data**: Consider pseudo labeling by adding confident predictions from the test set to the training set, retraining the model, and improving its performance.

#### 8. Deployment
- **Target Device**: Specify the target device for deployment (e.g., cloud, edge device).
- **Deployment Endpoint**: Set up a deployment endpoint for serving the model in a production environment.
- **Inference Engine**: Configure the inference engine to handle real-time predictions.
- **Demonstration**: Prepare a demonstration to showcase the model's capabilities and performance.

### Summary
By following this detailed plan, AI agents can systematically build a high-performance regression model to predict the age of crabs using the Crab Age Dataset. The plan includes all necessary steps from data collection and preprocessing to model training, evaluation, and deployment, ensuring a comprehensive and robust machine learning pipeline.
```

**Scenario 3**
```
### End-to-End Actionable Plan for Predicting Crab Age Using the Crab Age Dataset

#### 1. **Data Collection and Initial Setup**
- **Dataset Source**: Directly search and obtain the Crab Age Dataset.
- **Dataset Description**: The dataset comprises both categorical and numerical features aimed at predicting the age item.

#### 2. **Data Preprocessing**
- **Standardization/Normalization**:
  - Apply standardization or normalization to numerical features to ensure they are on a similar scale.
- **Encoding Categorical Features**:
  - Use one-hot encoding for categorical variables to transform them into a format suitable for machine learning algorithms.

#### 3. **Exploratory Data Analysis (EDA)**
- **Data Visualization**:
  - Visualize the distribution of numerical features using histograms.
  - Create box plots to identify outliers.
  - Use bar charts for categorical features to understand their frequency distribution.
- **Correlation Analysis**:
  - Compute the correlation matrix to identify relationships between numerical features and the target variable (age).

#### 4. **Feature Engineering**
- **Aggregation**:
  - Create aggregated features such as mean, median, and standard deviation for groups of features if applicable.
- **Smoothing**:
  - Apply smoothing techniques to reduce noise in the data.
- **Polynomial Features**:
  - Generate polynomial features to capture non-linear relationships.
- **Interaction Terms**:
  - Create interaction terms between features to capture combined effects.

#### 5. **Model Selection**
- **Ensemble Methods**:
  - Implement ensemble methods like Voting Regressor and Bagging to combine the strengths of multiple models.
- **Gradient Boosting**:
  - Utilize gradient boosting algorithms such as XGBoost, LightGBM, and CatBoost, which are efficient for handling both categorical and numerical features.

#### 6. **Model Training**
- **Cross-Validation**:
  - Employ k-fold cross-validation to evaluate model performance and ensure it generalizes well to unseen data.
- **Hyperparameter Tuning**:
  - Use grid search or random search to find the optimal hyperparameters for the selected models.
- **Regularization**:
  - Apply L1 (lasso) and L2 (ridge) regularization to prevent overfitting and handle multicollinearity.

#### 7. **Model Evaluation**
- **Primary Metric**:
  - Focus on optimizing the Root Mean Squared Log Error (RMSLE) as it is the primary evaluation metric for this task.
- **Secondary Metrics**:
  - Optionally, evaluate other metrics like R^2, MAE (Mean Absolute Error), and MSE (Mean Squared Error) for additional insights.

#### 8. **Model Interpretation**
- **Feature Importance**:
  - Utilize tools like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to interpret model predictions and understand feature importance.

#### 9. **Model Refinement**
- **Pseudo Labeling**:
  - Consider pseudo labeling by adding confident predictions from the test set to the training set and retraining the model to improve performance.

#### 10. **Deployment and Monitoring**
- **Target Device**:
  - Determine the target device for deployment (e.g., cloud, edge device).
- **Deployment Endpoint**:
  - Set up an endpoint for model deployment to facilitate real-time predictions.
- **Inference Engine**:
  - Use an inference engine compatible with the deployment environment to ensure efficient model serving.
- **Monitoring**:
  - Implement monitoring to track model performance and detect any drift in data or predictions over time.

#### 11. **Documentation and Demonstration**
- **Documentation**:
  - Document all steps, decisions, and configurations used in the model development pipeline.
- **Demonstration**:
  - Prepare a demonstration of the model's capabilities and performance using sample data to showcase its effectiveness.

By following this comprehensive plan, AI agents can effectively build, evaluate, and deploy a high-performance regression model tailored for predicting the age of crabs using the Crab Age Dataset.
```


### Example 4: Tabular Regression (Constraint-Aware)
> The dataset presented here (the Crab Age Dataset) comprises a lot of both categorical and numerical features. Please split the dataset into three parts of train, valid, and test. Your task is to predict the age item. The evaluation metric is the RMSLE (root mean squared log error).
Now train a regression model to get a good performance on this task, ensuring that the model's training time does not exceed 30 minutes and that it can make predictions on the test set within 5 seconds.

**Scenario 1**
```
### End-to-End Actionable Plan for AI Agents: Tabular Regression on Crab Age Dataset

#### 1. **Exploratory Data Analysis (EDA)**
- **Objective**: Understand the dataset and identify patterns, relationships, and potential issues.
- **Actions**:
  - **Visualize Data Distributions**: Generate histograms, box plots, and pair plots for numerical features.
  - **Correlation Analysis**: Compute and visualize correlation matrices to identify relationships between features.
  - **Categorical Analysis**: Generate bar plots and count plots for categorical features.
  - **Feature Relationships**: Plot scatter plots to explore relationships between features such as gender, weight, carapace length, carapace width, and abdomen width.

#### 2. **Data Preprocessing**
- **Objective**: Clean and prepare the dataset for modeling.
- **Actions**:
  - **Handle Missing Values**: Impute missing values using median/mode imputation or more advanced techniques like KNN imputation.
  - **Outlier Detection and Handling**: Identify outliers using the IQR method and handle them appropriately (e.g., capping or removing).
  - **Stationarity Check**: Ensure data stationarity using the Dickey-Fuller Test. Apply transformations like smoothing or polynomial fitting if necessary.
  - **Encoding Categorical Variables**: Convert categorical variables to numerical using one-hot encoding, label encoding, or target encoding.
  - **Scaling Numerical Features**: Standardize or normalize numerical features to ensure they are on a comparable scale.

#### 3. **Feature Engineering**
- **Objective**: Enhance the dataset with meaningful features.
- **Actions**:
  - **Create New Features**: Generate features such as body mass index (BMI) and polynomial features.
  - **Interaction Terms**: Create interaction terms between features to capture non-linear relationships.
  - **Feature Selection**: Use techniques like Recursive Feature Elimination (RFE) or feature importance from models to select relevant features.

#### 4. **Dataset Splitting**
- **Objective**: Split the dataset for training, validation, and testing.
- **Actions**:
  - **Split Ratios**: Divide the dataset into training (70%), validation (15%), and test (15%) sets.
  - **Stratification**: Ensure stratified splitting if the target variable distribution is imbalanced.

#### 5. **Model Selection**
- **Objective**: Select and train appropriate models for the regression task.
- **Actions**:
  - **Baseline Model**: Start with a Linear Regression model to establish a performance benchmark.
  - **Advanced Models**: Train ensemble models such as Random Forest, Gradient Boosting Machines (GBM), XGBoost, LightGBM, and CatBoost.
  - **Support Vector Machines (SVM)**: Train SVM models to capture complex relationships.

#### 6. **Model Training and Cross-Validation**
- **Objective**: Train models effectively and ensure generalization.
- **Actions**:
  - **k-Fold Cross-Validation**: Use k-fold cross-validation to evaluate model performance and tune hyperparameters.
  - **Early Stopping**: Implement early stopping to prevent overfitting and adhere to the 30-minute training time constraint.

#### 7. **Hyperparameter Optimization**
- **Objective**: Optimize model hyperparameters for better performance.
- **Actions**:
  - **Grid Search/Random Search**: Perform hyperparameter tuning using Grid Search or Random Search.
  - **Bayesian Optimization**: Use Bayesian Optimization for more efficient hyperparameter tuning.
  - **Pseudo-Labeling**: Enhance training data by adding confident predictions from the test set to the training set.

#### 8. **Ensemble Methods**
- **Objective**: Improve prediction accuracy by combining multiple models.
- **Actions**:
  - **Voting Regressor**: Combine predictions from different models using a Voting Regressor to balance out individual model weaknesses.

#### 9. **Evaluation and Metrics**
- **Objective**: Evaluate model performance using appropriate metrics.
- **Actions**:
  - **RMSLE**: Calculate the Root Mean Squared Logarithmic Error (RMSLE) to measure prediction accuracy.
  - **Learning Curves**: Plot learning curves to visualize model performance over time and check for overfitting/underfitting.

#### 10. **Implementation Tips**
- **Objective**: Ensure efficient model training and deployment.
- **Actions**:
  - **Efficient Algorithms**: Use LightGBM and XGBoost for their speed and performance.
  - **Model Saving**: Save the trained model using libraries like pickle for future predictions.
  - **Feature Importance**: Use models that provide feature importance scores to understand feature contributions.

### Summary
By following this comprehensive plan, AI agents will be able to build a robust and efficient regression model for predicting the age from the Crab Age Dataset, adhering to the constraints of a maximum 30-minute training time and 5-second prediction time.
```

**Scenario 2**
```
## End-to-End Actionable Plan for AutoML Project

### 1. Problem Understanding and Requirements

**User Intent:** Build a model  
**User Expertise:** Medium  
**Problem Area:** Tabular Data Analysis  
**Downstream Task:** Tabular Regression  
**Application Domain:** Data Science  
**Dataset:** Crab Age Dataset  
**Target Variable:** Age  
**Evaluation Metric:** RMSLE (Root Mean Squared Log Error)  
**Performance Constraints:** Training time ≤ 30 minutes, Prediction time ≤ 5 seconds  

### 2. Data Collection and Loading

- **Source:** Direct-search
- **Description:** The dataset comprises both categorical and numerical features. The task is to predict the age item.

### 3. Exploratory Data Analysis (EDA)

1. **Understand Data Structure:**
   - Examine the dataset schema to understand the types and distributions of features (categorical and numerical).

2. **Visualizations:**
   - Generate histograms, box plots, and scatter plots to visualize data distributions and potential correlations.
   - Create correlation matrices to identify relationships between features.

### 4. Data Preprocessing

1. **Handling Missing Values and Outliers:**
   - Impute missing values using appropriate methods (e.g., mean, median, mode).
   - Use the interquartile range (IQR) method to detect and handle outliers.

2. **Stationarity:**
   - Test for stationarity using statistical tests like the Dickey-Fuller Test.
   - Apply techniques like aggregation, smoothing, or polynomial fitting to remove trends if necessary.

3. **Encoding Categorical Variables:**
   - Convert categorical features into numerical values using one-hot encoding, label encoding, or target encoding.

4. **Scaling Numerical Features:**
   - Normalize or standardize numerical features to ensure they are on a comparable scale.

### 5. Feature Engineering

1. **Create Derived Features:**
   - Generate new features such as Body Mass Index (BMI) and polynomial features to capture non-linear relationships.
   - Consider interaction terms and other derived features that may enhance the model's predictive power.

### 6. Dataset Splitting

1. **Split the Dataset:**
   - Split the dataset into training (60-80%), validation (10-20%), and test (10-20%) sets to ensure proper model evaluation and tuning.

### 7. Model Selection

1. **Baseline Model:**
   - Start with a simple model like Linear Regression to establish a benchmark.

2. **Advanced Models:**
   - Consider ensemble methods like Random Forest, Gradient Boosting Machines (GBM), XGBoost, LightGBM, and CatBoost for robust performance.
   - Evaluate Support Vector Machines (SVM) for their ability to handle complex relationships.

### 8. Model Training and Cross-Validation

1. **Cross-Validation:**
   - Use k-fold cross-validation to ensure the model generalizes well to unseen data and to select the best model parameters.

2. **Early Stopping:**
   - Implement early stopping during training to prevent overfitting and adhere to time constraints.

### 9. Hyperparameter Optimization

1. **Optimization Techniques:**
   - Use Grid Search, Random Search, or Bayesian Optimization to find the best hyperparameters within the limited training time.

2. **Pseudo-Labeling:**
   - Enhance training data by adding confident predictions from the test set to the training set and retraining the model.

### 10. Ensemble Methods

1. **Combine Predictions:**
   - Use a Voting Regressor to combine predictions from multiple models to balance out individual model weaknesses and improve overall performance.

### 11. Evaluation and Metrics

1. **Primary Evaluation Metric:**
   - Use RMSLE (Root Mean Squared Logarithmic Error) to measure the accuracy of the model's predictions.

2. **Learning Curves:**
   - Plot learning curves to visualize the model's performance over time and ensure it is not overfitting or underfitting.

### 12. Implementation Tips

1. **Efficiency:**
   - Ensure model training time does not exceed 30 minutes and the prediction time is within 5 seconds.
   - Use efficient algorithms like LightGBM and XGBoost for their speed and performance.

2. **Model Saving:**
   - Save the trained model using libraries like pickle for future predictions.

3. **Feature Importance:**
   - Use models that provide feature importance scores to understand which features contribute the most to the predictions.

### 13. Deployment and Inference

1. **Target Device and Deployment Endpoint:**
   - Specify the target device for deployment and the endpoint for inference.

2. **Inference Engine:**
   - Define the inference engine to be used for making predictions.

3. **Demonstration:**
   - Prepare a demonstration of the model's capabilities to showcase its performance.

By following this comprehensive plan, the AI agents can build a robust and efficient regression model to predict the age from the Crab Age Dataset, adhering to the given constraints and requirements.
```

**Scenario 3**
```
### End-to-End Actionable Plan for AutoML Agents

#### 1. Exploratory Data Analysis (EDA)
- **Objective**: Understand the dataset, identify patterns, and detect potential issues.
- **Actions**:
  - Analyze relationships between features such as gender, weight, carapace length, carapace width, and abdomen width.
  - Visualize data distributions using histograms, box plots, and scatter plots.
  - Compute and visualize correlations between numerical features using heatmaps.
  - Identify and document any anomalies or outliers.

#### 2. Data Preprocessing
- **Objective**: Prepare the data for modeling by handling missing values, outliers, and ensuring consistency.
- **Actions**:
  - **Handling Missing Values**: Impute missing values using methods like mean, median, or mode imputation.
  - **Outlier Detection and Handling**: Use the interquartile range (IQR) method to detect and handle outliers.
  - **Stationarity**: Ensure the dataset is stationary by removing trends using aggregation, smoothing, or polynomial fitting. Test for stationarity using the Dickey-Fuller Test.
  - **Encoding Categorical Variables**: Convert categorical features into numerical values using one-hot encoding, label encoding, or target encoding.
  - **Scaling Numerical Features**: Normalize or standardize numerical features to ensure they are on a comparable scale.

#### 3. Feature Engineering
- **Objective**: Enhance the dataset with new features to improve model performance.
- **Actions**:
  - Create new features such as Body Mass Index (BMI) and polynomial features to capture non-linear relationships.
  - Generate interaction terms and other derived features that may enhance the model's predictive power.

#### 4. Dataset Splitting
- **Objective**: Split the dataset into training, validation, and test sets for model training and evaluation.
- **Actions**:
  - Split the dataset into training (70%), validation (15%), and test (15%) sets to ensure proper model evaluation and tuning.

#### 5. Model Selection
- **Objective**: Identify and select appropriate models for the regression task.
- **Actions**:
  - **Baseline Model**: Start with a simple model like Linear Regression to establish a benchmark.
  - **Advanced Models**: Use ensemble methods like Random Forest, Gradient Boosting Machines (GBM), XGBoost, LightGBM, and CatBoost.
  - **Support Vector Machines (SVM)**: Consider SVMs for their ability to handle complex relationships.

#### 6. Model Training and Cross-Validation
- **Objective**: Train the selected models and validate their performance.
- **Actions**:
  - Use k-fold cross-validation (e.g., 5-fold) to ensure the model generalizes well to unseen data.
  - Implement early stopping during training to prevent overfitting and adhere to the 30-minute training time constraint.

#### 7. Hyperparameter Optimization
- **Objective**: Optimize model hyperparameters to achieve the best performance within the given constraints.
- **Actions**:
  - Use Grid Search, Random Search, or Bayesian Optimization to find the best hyperparameters.
  - Consider pseudo-labeling to enhance training data by adding confident predictions from the test set to the training set and retraining the model.

#### 8. Ensemble Methods
- **Objective**: Improve model performance by combining predictions from multiple models.
- **Actions**:
  - Combine predictions using a Voting Regressor to balance out individual model weaknesses and improve overall performance.

#### 9. Evaluation and Metrics
- **Objective**: Evaluate the model's performance using the specified metrics.
- **Actions**:
  - Use RMSLE (Root Mean Squared Logarithmic Error) as the primary evaluation metric.
  - Plot learning curves to visualize the model's performance over time and ensure it is not overfitting or underfitting.

#### 10. Implementation Tips
- **Objective**: Ensure efficient model training and prediction within the given constraints.
- **Actions**:
  - Ensure that the model training time does not exceed 30 minutes and the prediction time is within 5 seconds.
  - Use efficient algorithms like LightGBM and XGBoost for their speed and performance.
  - Save the trained model using libraries like pickle for future predictions.
  - Use models that provide feature importance scores to understand which features contribute the most to the predictions.

By following this comprehensive plan, the AI agents can build a robust and efficient regression model for predicting the age from the Crab Age Dataset.
```
