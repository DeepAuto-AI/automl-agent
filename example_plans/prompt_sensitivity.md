# Sensitivity Analysis of Different System Prompts

## Experimental Results
| Dataset                | Generic | Generic + Instructions | Entry Role | Experienced Role | World's Best Role | AutoML-Agent<br>(Mixed Level) |
| ---------------------- | ------- | ---------------------- | ---------- | ---------------- | ----------------- | ----------------------------- |
| smoker-status          | 0.753   | 0.788                  | 0.830      | 0.769            | 0.912             | 0.762                         |
| click-prediction-small | 0.110   | 0.293                  | 0.345      | 0.132            | 0.344             | 0.352                         |
| mfeat-factors          | 0.950   | 0.870                  | 0.918      | 0.913            | 0.950             | 0.940                         |
| wine-quality-white     | 0.326   | 0.629                  | 0.621      | 0.607            | 0.571             | 0.652                         |
| colleges               | 0.879   | 0.876                  | 0.876      | 0.985            | 0.931             | 0.878                         |
| house-prices           | 0.087   | 0.089                  | 0.087      | 0.100            | 0.090             | 0.090                         |
| **_Average_**                | 0.518   | 0.591                  | 0.613      | 0.584            | 0.633             | 0.612                         |

## Examples of Intermediate Generated Outcomes
### Generic Prompt
> You are a helpful assistant.

#### Tabular Classification (`mfeat-factors`)
- Global Plan
```
Here's a comprehensive end-to-end plan designed for AI agents to build a solution for tabular classification using the mfeat-factors dataset, ensuring alignment with the user's requirements and leveraging the state-of-the-art methodologies:

### Step 1: Data Retrieval and Initial Exploration

1. **Data Loading:**
   - Retrieve the `mfeat-factors` dataset from the specified source.
   - Verify data integrity by checking for completeness and correctness of the dataset.

2. **Initial Data Exploration:**
   - Perform exploratory data analysis (EDA) to understand the dataset's structure and distribution.
   - Summarize statistics for numeric and symbolic features.
   - Visualize class distribution to assess potential class imbalances.

### Step 2: Data Preprocessing

1. **Normalization/Standardization:**
   - Normalize or standardize numeric features to ensure consistent scaling. Use methods such as Min-Max scaling or Z-score normalization.

2. **Data Reshaping:**
   - Assess if reshaping of data is required, especially if considering neural network architectures.

3. **Handle Missing Values:**
   - Confirm the absence of missing values as indicated in the dataset specification.

### Step 3: Feature Engineering

1. **Dimensionality Reduction:**
   - Apply Principal Component Analysis (PCA) to reduce dimensionality while retaining significant variance.

2. **Feature Selection:**
   - Use techniques like Recursive Feature Elimination (RFE) or mutual information to select the most informative features.

3. **Interaction and Polynomial Features:**
   - Generate interaction terms or polynomial features to capture complex relationships.

### Step 4: Data Augmentation

1. **Tabular Data Augmentation:**
   - Implement augmentation techniques such as SMOTE for class imbalance or feature noise addition to enhance model robustness.

### Step 5: Model Selection and Development

1. **Model Architectures:**
   - For neural networks, design a Multi-Layer Perceptron (MLP) tailored for the dataset.
   - Explore ensemble methods like Random Forests, Gradient Boosting, XGBoost, LightGBM, and CatBoost due to their effectiveness in handling large feature spaces.

2. **Feature Extraction:**
   - Consider CNN-inspired feature extraction techniques, leveraging convolutional layers to extract complex patterns, if applicable.

### Step 6: Model Training

1. **Training Strategy:**
   - Implement stratified cross-validation to ensure robust evaluation and avoid overfitting.
   - Use the weighted F1 score as the primary evaluation metric.

2. **Hyperparameter Tuning:**
   - Conduct hyperparameter optimization using grid search or random search to fine-tune model parameters.

### Step 7: Model Evaluation and Reporting

1. **Evaluation:**
   - Evaluate model performance on validation data using the weighted F1 score.
   - Analyze confusion matrix to understand classification errors.

2. **Result Reporting:**
   - Document the model's performance, highlighting the weighted F1 score achieved on evaluation data.

### Step 8: Implementation and Deployment

1. **Model Implementation:**
   - Utilize libraries such as TensorFlow/Keras for neural networks and Scikit-learn/XGBoost/LightGBM/CatBoost for ensemble methods.

2. **Deployment Considerations:**
   - Prepare the model for deployment, ensuring it meets performance requirements for the target environment.
   - Document the model inference process and prepare an API or endpoint for deployment.

By following this structured plan, AI agents can efficiently build, evaluate, and deploy a robust solution for the classification of handwritten digits using the mfeat-factors dataset, ensuring high performance and generalization.
```
- Plan Execution: **Data Agent**
```
To effectively manipulate and analyze the `mfeat-factors` dataset, follow this detailed plan, which encompasses data retrieval, preprocessing, augmentation, and feature engineering. Each step is crucial for ensuring the dataset is ready for subsequent modeling and evaluation tasks.

### 1. Data Retrieval

- **Dataset Collection:**
  - Retrieve the dataset from the specified user-uploaded source. The dataset files are located at:
    - `agent_workspace/datasets/mfeat-factors/split_dev.csv`
    - `agent_workspace/datasets/mfeat-factors/split_dev_target.csv`
    - `agent_workspace/datasets/mfeat-factors/split_dev_wo_target.csv`
    - `agent_workspace/datasets/mfeat-factors/split_test.csv`
    - `agent_workspace/datasets/mfeat-factors/split_test_target.csv`
    - `agent_workspace/datasets/mfeat-factors/split_test_wo_target.csv`
    - `agent_workspace/datasets/mfeat-factors/split_train.csv`
  - Ensure the dataset is complete and correct by verifying the integrity of each file. Check for consistent formatting and confirm there are no missing values as per the dataset specification.

### 2. Data Preprocessing

- **Normalization/Standardization:**
  - Use Min-Max scaling or Z-score normalization to standardize numeric features. This ensures that all features contribute equally to the analysis and prevents features with larger ranges from dominating the model. Libraries such as `scikit-learn` can be used for this purpose.
  - Min-Max scaling transforms features to a specific range, typically [0, 1], while Z-score normalization scales features to have a mean of 0 and a standard deviation of 1.

- **Data Reshaping:**
  - Evaluate the need for reshaping the data, especially if neural network architectures are considered. This may involve reshaping the data into a format compatible with the input layer of a neural network, such as converting a 2D array into a 3D array if required.

- **Missing Values:**
  - Confirm the absence of missing values by conducting a thorough check using tools like `pandas` to ensure data integrity. This step is crucial as missing values can lead to biased results and affect model performance.

### 3. Data Augmentation

- **Tabular Data Augmentation:**
  - Address potential class imbalances using SMOTE (Synthetic Minority Over-sampling Technique). SMOTE generates synthetic samples for the minority class, helping to balance the class distribution and improve model performance.
  - Introduce feature noise addition by slightly perturbing feature values. This can enhance model robustness and generalization capabilities by simulating real-world data variability.

### 4. Feature Engineering and Understanding

- **Dimensionality Reduction:**
  - Apply Principal Component Analysis (PCA) to reduce the dimensionality of the dataset while retaining significant variance. PCA simplifies the feature space, reduces computational cost, and can improve model performance by eliminating noise.

- **Feature Selection:**
  - Implement techniques such as Recursive Feature Elimination (RFE) or mutual information to identify and select the most informative features. This step helps in focusing on features that contribute most to the target variable, potentially improving model accuracy and interpretability.

- **Interaction and Polynomial Features:**
  - Generate interaction terms or polynomial features to capture complex relationships within the data. This can be done using libraries like `scikit-learn` to create new features that represent interactions between existing features, potentially enhancing model performance.

### Expected Outcomes and Results

- **Quantitative Outcomes:**
  - A balanced dataset with normalized features, ready for modeling.
  - Reduced dimensionality with PCA, retaining significant variance.
  - A set of selected features that are most informative for the target variable.

- **Qualitative Outcomes:**
  - Improved data integrity and consistency across features.
  - Enhanced model robustness and generalization due to data augmentation.
  - Simplified feature space, making the data more manageable and interpretable.

By following these steps, the `mfeat-factors` dataset will be effectively prepared for downstream tasks, ensuring reproducibility and alignment with the project's objectives.
```
- Plan Execution: **Model Agent**
```
To find the top-3 high-performance models for a tabular classification task using the mfeat-factors dataset, we will follow a structured approach focusing on model architecture selection, hyperparameter optimization, and evaluation. Below are the detailed steps:

### 1. Retrieve or Find High-Performance Model(s)

#### Model Architectures

1. **Multi-Layer Perceptron (MLP):**
   - Design a neural network with an input layer matching the number of features in the dataset.
   - Use two hidden layers with 64 and 32 neurons respectively, employing ReLU activation functions.
   - The output layer should have neurons equal to the number of classes, using a softmax activation function for classification.

2. **Gradient Boosting Machines (GBM):**
   - Consider using XGBoost, LightGBM, and CatBoost, which are highly effective for tabular data.
   - These models are robust to feature scaling and can handle missing values natively.

3. **Random Forest:**
   - Use an ensemble of decision trees to capture complex interactions between features.
   - Random Forest is known for its robustness and ability to handle overfitting through averaging.

Each of these architectures is chosen for their ability to handle tabular data, with ensemble methods providing robustness and neural networks offering flexibility in capturing non-linear patterns.

### 2. Optimize the Hyperparameters of the Retrieved Models

#### Hyperparameter Tuning

1. **MLP:**
   - Learning Rate: Optimize between 0.001 and 0.01. Optimal value: 0.005.
   - Batch Size: Test values 32, 64, 128. Optimal value: 64.
   - Number of Epochs: Start with 100 epochs, using early stopping based on validation loss.

2. **XGBoost:**
   - Learning Rate (eta): Explore 0.01 to 0.3. Optimal value: 0.1.
   - Max Depth: Test values 3 to 10. Optimal value: 6.
   - Number of Estimators: Between 100 and 500. Optimal value: 300.

3. **Random Forest:**
   - Number of Trees: Between 100 and 500. Optimal value: 200.
   - Max Features: Test 'sqrt' and 'log2'. Optimal value: 'sqrt'.
   - Max Depth: Test values None, 10, 20. Optimal value: None (fully grown trees).

Hyperparameter tuning is crucial to optimize the performance of each model. Using techniques like grid search or random search ensures a comprehensive exploration of the hyperparameter space.

### 3. Extract and Understand the Underlying Characteristics

#### Model Characteristics

1. **MLP:**
   - Computational Complexity: Moderate, due to matrix operations in dense layers.
   - Memory Usage: Depends on the number of neurons and layers. Estimated model size: ~1MB.
   - Inference Latency: Fast, typically less than 10ms per sample.

2. **XGBoost:**
   - Computational Complexity: High during training due to boosting iterations.
   - Memory Usage: Moderate, as it stores multiple trees. Estimated model size: ~5MB.
   - Inference Latency: Moderate, typically 20-50ms per sample.

3. **Random Forest:**
   - Computational Complexity: Moderate, as trees are trained independently.
   - Memory Usage: High, due to storing multiple full trees. Estimated model size: ~10MB.
   - Inference Latency: Moderate, typically 30-60ms per sample.

Understanding these characteristics helps in selecting models that not only perform well but also meet operational constraints like memory and latency.

### 4. Select the Top-k Models or Algorithms

#### Model Evaluation and Selection

- Evaluate each model using the weighted F1 score on a validation set.
- Analyze confusion matrices to understand classification errors.
- Select the top-3 models based on their performance metrics and robustness across cross-validation folds.

### Expected Quantitative Performance

1. **MLP:**
   - Weighted F1 Score: ~0.85
   - Model Size: ~1MB
   - Inference Speed: <10ms per sample

2. **XGBoost:**
   - Weighted F1 Score: ~0.88
   - Model Size: ~5MB
   - Inference Speed: 20-50ms per sample

3. **Random Forest:**
   - Weighted F1 Score: ~0.86
   - Model Size: ~10MB
   - Inference Speed: 30-60ms per sample

These models are selected based on their ability to generalize well to unseen data, as evidenced by their cross-validation performance and operational characteristics.
```


#### Tabular Regression (`colleges`)
- Global Plan
```
Here's a detailed, end-to-end actionable plan for the AI agents to predict the `percent_pell_grant` using the colleges dataset, tailored to the user's medium expertise level in tabular regression within the education domain. This plan will cover the entire machine learning pipeline from data preprocessing to model evaluation and interpretation.

### Step 1: Data Retrieval and Understanding
- **Objective**: Load and understand the dataset to prepare for preprocessing and feature engineering.
- **Actions**:
  1. Load the dataset into a suitable environment (e.g., Pandas DataFrame).
  2. Perform an initial exploration to understand the data types, distribution, and presence of missing values.
  3. Summarize the dataset characteristics, including the number of instances, features, and target distribution.

### Step 2: Data Preprocessing
- **Objective**: Clean and prepare the data for analysis and modeling.
- **Actions**:
  1. **Handle Missing Values**:
     - Identify columns with missing values and the proportion of missing data.
     - Apply imputation techniques: Use mean for numeric features and the most frequent category for categorical features. Consider k-nearest neighbors or iterative imputation for more advanced handling.
  2. **Normalize Numeric Features**:
     - Apply Min-Max scaling or Standard Scaler to ensure consistent feature scaling.
  3. **Encode Categorical Features**:
     - Convert categorical features to numerical using one-hot encoding or ordinal encoding.
  
### Step 3: Feature Engineering
- **Objective**: Enhance the dataset with meaningful features that improve model performance.
- **Actions**:
  1. **Feature Selection**:
     - Utilize Lasso regression to identify and select impactful features.
  2. **Create Interaction Terms**:
     - Generate interaction terms or polynomial features to capture non-linear relationships.
  3. **Transformations**:
     - Apply log or square root transformations to stabilize variance or normalize skewed distributions.

### Step 4: Model Selection and Development
- **Objective**: Develop models capable of capturing relationships in the data.
- **Actions**:
  1. **Model Selection**:
     - Start with ensemble methods like Random Forests and Gradient Boosting Machines.
     - Experiment with CatBoost and LightGBM for handling categorical features and missing values.
     - Consider advanced models like TabNet for its attention mechanisms.
  2. **Neural Networks**:
     - Evaluate architectures from the fast.ai library designed for tabular data.
  3. **Hyperparameter Tuning**:
     - Use grid search or random search to optimize hyperparameters for selected models.

### Step 5: Model Evaluation and Validation
- **Objective**: Assess model performance and ensure generalizability.
- **Actions**:
  1. **Cross-validation**:
     - Implement k-fold cross-validation to estimate model performance.
  2. **Performance Metric**:
     - Focus on minimizing RMSE as the primary metric for evaluation.
  3. **Model Interpretability**:
     - Use SHAP (SHapley Additive exPlanations) to understand feature contributions.
     - Employ LIME for local model interpretability.

### Step 6: Model Deployment and Reporting
- **Objective**: Prepare the model for deployment and report findings.
- **Actions**:
  1. **Deployment**:
     - Ensure the model is ready for deployment on the specified target device or endpoint.
  2. **Reporting**:
     - Document the entire process, including preprocessing steps, feature engineering, model selection, tuning, and evaluation results.
     - Highlight key insights and feature importances derived from interpretability tools.

By following this comprehensive plan, the AI agents will be able to effectively develop and evaluate a predictive model for the `percent_pell_grant` with a focus on accuracy, interpretability, and robust handling of mixed data types and missing values.
```
- Plan Execution: **Data Agent**
```
To effectively execute the machine learning development plan for predicting `percent_pell_grant` in the education domain, we will follow a structured approach focusing on data manipulation and analysis. Below are the detailed steps:

### 1. Retrieve or Collect the Dataset(s)

**Objective**: Load the dataset into a suitable environment for analysis.

- **Data Source**: The dataset is provided by the user and is available in multiple CSV files located at specified paths. These include subsets for development, testing, and training, with and without target variables.

- **Loading the Dataset**: Use the Pandas library to read these CSV files into DataFrames. This is done using the `read_csv()` function, which allows for easy manipulation and analysis of tabular data. Ensure that each file is loaded correctly by checking the first few rows and data types.

### 2. Preprocess the Retrieved Dataset(s)

**Objective**: Clean and prepare the data for analysis and modeling.

- **Handle Missing Values**:
  - **Identification**: Use Pandas functions to identify columns with missing values and calculate the proportion of missing data in each column. This helps in understanding the extent of missingness and deciding on appropriate imputation strategies.
  - **Imputation**:
    - For numeric features, replace missing values with the mean of the column. This is a simple yet effective method to handle missing data without introducing bias.
    - For categorical features, replace missing values with the most frequent category. This helps maintain the distribution of categories.
    - Consider advanced imputation techniques like k-nearest neighbors or iterative imputation for more accurate handling, especially if missing data is substantial.

- **Normalize Numeric Features**:
  - Apply Min-Max scaling to rescale features to a range between 0 and 1, or use Standard Scaler to center the data around the mean with unit variance. Normalization ensures that all numeric features contribute equally to the analysis and prevents features with larger ranges from dominating.

- **Encode Categorical Features**:
  - Convert categorical variables into numerical format using one-hot encoding, which creates binary columns for each category, or ordinal encoding if the categories have a meaningful order. This step is crucial for machine learning algorithms that require numerical input.

### 3. Efficiently Augment the Dataset(s)

**Objective**: Enhance the dataset by creating new features and selecting the most relevant ones.

- **Feature Engineering**:
  - **Feature Selection**: Use Lasso regression to identify and select impactful features. Lasso adds a penalty to the regression that can shrink some coefficients to zero, effectively selecting a subset of features.
  - **Create Interaction Terms**: Generate interaction terms or polynomial features to capture non-linear relationships between features. This can improve model performance by allowing it to learn more complex patterns.
  - **Transformations**: Apply transformations such as log or square root to stabilize variance or normalize skewed distributions, which can improve the robustness and performance of the model.

### 4. Extract and Understand the Underlying Characteristics of the Dataset(s)

**Objective**: Gain insights into the dataset's structure and content.

- **Initial Exploration**:
  - Use Pandas functions like `info()` to examine data types and non-null counts, `describe()` to get summary statistics, and `isnull().sum()` to check for missing values. This provides a comprehensive overview of the dataset.
  - Summarize key characteristics such as the number of instances, number of features, and distribution of the target variable. Understanding these aspects is essential for designing effective preprocessing and modeling strategies.

### Expected Outcomes and Results

**Quantitative Outcomes**:
- A clean dataset with no missing values, ready for modeling.
- Numeric features normalized to a consistent scale.
- Categorical features encoded into a numerical format.
- A reduced set of impactful features identified through feature selection.
- Enhanced dataset with interaction terms and transformed features for improved model performance.

**Qualitative Outcomes**:
- A deeper understanding of the dataset's structure and characteristics.
- Insights into potential relationships and patterns within the data.
- A well-prepared dataset that is suitable for the downstream task of regression modeling to predict `percent_pell_grant`.

By following these steps, we ensure that the dataset is meticulously prepared, enhancing the likelihood of successful model development and accurate predictions.
```
- Plan Execution: **Model Agent**
```
To identify and optimize the top-3 high-performance models for predicting `percent_pell_grant`, we will follow a structured approach focusing on model selection, hyperparameter optimization, and understanding model characteristics. This process will ensure that we select models that are not only accurate but also efficient and interpretable.

### 1. Model Selection

**Ensemble Methods**:
- **Random Forests**: Start with Random Forests due to their robustness and ability to handle high-dimensional data. They are less prone to overfitting compared to single decision trees.
- **Gradient Boosting Machines (GBM)**: Use GBM for its ability to improve model accuracy through iterative boosting. It is effective for capturing complex patterns in the data.
- **CatBoost and LightGBM**: These are particularly effective for datasets with categorical features and missing values. CatBoost handles categorical variables natively, while LightGBM is known for its speed and efficiency.
- **TabNet**: Consider TabNet for its attention mechanisms, which can capture complex relationships and interactions in tabular data.
- **Fast.ai Neural Networks**: Explore neural network architectures from the fast.ai library, which are designed to leverage deep learning capabilities for tabular data.

### 2. Hyperparameter Optimization

**Random Forests**:
- **Number of Trees**: Optimize between 100 and 1000 trees. A good starting point might be 500 trees.
- **Max Depth**: Optimize between 10 and 50. A depth of 20 often balances complexity and performance.
- **Min Samples Split**: Optimize between 2 and 10. A value of 5 is often effective.

**Gradient Boosting Machines**:
- **Learning Rate**: Optimize between 0.01 and 0.1. A learning rate of 0.05 is a good balance.
- **Number of Estimators**: Optimize between 100 and 1000. Start with 300 estimators.
- **Max Depth**: Optimize between 3 and 10. A depth of 5 is commonly used.

**CatBoost**:
- **Learning Rate**: Optimize between 0.01 and 0.1. A learning rate of 0.03 is effective.
- **Depth**: Optimize between 6 and 10. A depth of 8 is recommended.
- **Iterations**: Optimize between 500 and 2000. Start with 1000 iterations.

**LightGBM**:
- **Learning Rate**: Optimize between 0.01 and 0.1. A learning rate of 0.05 is suitable.
- **Number of Leaves**: Optimize between 20 and 100. A value of 31 is often used.
- **Max Depth**: Optimize between -1 (no limit) and 50. A depth of 15 is a good compromise.

**TabNet**:
- **Learning Rate**: Optimize between 0.01 and 0.1. A learning rate of 0.02 is effective.
- **Number of Steps**: Optimize between 3 and 10. A value of 5 is recommended.
- **Batch Size**: Optimize between 256 and 1024. A batch size of 512 is common.

**Fast.ai Neural Networks**:
- **Learning Rate**: Optimize using a learning rate finder, typically between 1e-5 and 1e-1. A value of 1e-3 is often effective.
- **Number of Layers**: Optimize between 2 and 5. A 3-layer network is a good starting point.
- **Batch Size**: Optimize between 64 and 256. A batch size of 128 is often used.

### 3. Understanding Model Characteristics

**Random Forests**:
- **Computation Complexity**: Moderate, due to the ensemble of trees.
- **Memory Usage**: High, as it stores multiple trees.
- **Inference Latency**: Moderate, as predictions require aggregating results from all trees.

**Gradient Boosting Machines**:
- **Computation Complexity**: High, due to iterative boosting.
- **Memory Usage**: Moderate, as it stores fewer trees than Random Forests.
- **Inference Latency**: Moderate, similar to Random Forests.

**CatBoost**:
- **Computation Complexity**: Moderate, due to efficient handling of categorical features.
- **Memory Usage**: Moderate, optimized for categorical data.
- **Inference Latency**: Fast, due to efficient implementation.

**LightGBM**:
- **Computation Complexity**: Low, due to histogram-based learning.
- **Memory Usage**: Low, as it uses a leaf-wise growth strategy.
- **Inference Latency**: Fast, due to efficient tree traversal.

**TabNet**:
- **Computation Complexity**: High, due to attention mechanisms.
- **Memory Usage**: Moderate, as it requires storing attention weights.
- **Inference Latency**: Moderate, due to sequential attention steps.

**Fast.ai Neural Networks**:
- **Computation Complexity**: High, due to deep learning architecture.
- **Memory Usage**: High, as it stores weights for multiple layers.
- **Inference Latency**: Fast, due to GPU acceleration.

### 4. Model Evaluation and Selection

- Implement k-fold cross-validation to ensure reliable performance estimates.
- Focus on minimizing RMSE as the primary metric for model accuracy.
- Use model interpretability tools like SHAP and LIME to ensure decisions are explainable.
- Rank models based on cross-validated RMSE scores and interpretability insights.

### Top-3 Candidate Models

1. **LightGBM**
   - **Expected RMSE**: 0.05
   - **Number of Parameters**: Approximately 100,000
   - **FLOPs**: 1 million
   - **Model Size**: 10 MB
   - **Training Time**: 2 hours
   - **Inference Speed**: 10 ms per sample

2. **CatBoost**
   - **Expected RMSE**: 0.06
   - **Number of Parameters**: Approximately 120,000
   - **FLOPs**: 1.2 million
   - **Model Size**: 12 MB
   - **Training Time**: 3 hours
   - **Inference Speed**: 15 ms per sample

3. **TabNet**
   - **Expected RMSE**: 0.07
   - **Number of Parameters**: Approximately 150,000
   - **FLOPs**: 1.5 million
   - **Model Size**: 15 MB
   - **Training Time**: 4 hours
   - **Inference Speed**: 20 ms per sample

These models are selected based on their balance of performance, efficiency, and interpretability, making them suitable candidates for the task at hand.
```

---

### Generic Prompt with Instruction Steps
> You are a helpful assistant. You have main responsibilities as follows.
> _...agent-specific steps..._

#### Tabular Classification (`mfeat-factors`)
- Global Plan
```
## Actionable Plan for Building a Tabular Classification Model for Handwritten Digit Recognition

### 1. Data Understanding and Initial Preparation

#### 1.1 Data Retrieval
- Retrieve the `mfeat-factors` dataset, which consists of 2,000 instances and 217 features, from the user-uploaded source.

#### 1.2 Exploratory Data Analysis (EDA)
- Conduct a comprehensive exploratory data analysis to understand the dataset's feature distributions and relationships.
  - Calculate summary statistics (mean, median, mode) for numerical features.
  - Plot histograms and boxplots for feature distribution visualization.
  - Use pair plots to identify potential correlations between features.

### 2. Data Preprocessing

#### 2.1 Data Cleaning
- Confirm that there are no missing values in the dataset.

#### 2.2 Feature Scaling
- Standardize or normalize the numerical features to ensure equal contribution to the model's performance.

#### 2.3 Data Splitting
- Split the dataset into training (80%) and validation (20%) sets, ensuring a balanced representation of all classes.

### 3. Feature Engineering

#### 3.1 Dimensionality Reduction
- Apply Principal Component Analysis (PCA) to reduce dimensionality while retaining 95% variance in the data.

#### 3.2 Feature Selection
- Use Recursive Feature Elimination (RFE) with a base model (e.g., Logistic Regression) to identify and retain the most significant features.

### 4. Model Development

#### 4.1 Model Selection
- Implement and evaluate several classical machine learning models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting Machines (GBM)
  - Support Vector Machines (SVM)
  - Decision Trees
  - XGBoost

#### 4.2 Model Training
- Train each model using the training dataset. Ensure proper handling of class imbalance if any is detected.

### 5. Model Evaluation

#### 5.1 Performance Metric
- Evaluate models using the F1 weighted score to account for class imbalance and ensure balanced precision and recall across classes.

#### 5.2 Cross-Validation
- Perform k-fold cross-validation (e.g., k=5) to assess model stability and robustness.

### 6. Model Optimization

#### 6.1 Hyperparameter Tuning
- Use GridSearchCV or Random Search to explore and optimize hyperparameters of each model, targeting an increase in the F1 weighted score.

#### 6.2 Classification Threshold Adjustment
- Analyze ROC and precision-recall curves to determine the optimal threshold that maximizes the F1 score.

### 7. Performance Reporting

#### 7.1 Reporting
- Compile a detailed report of model performances, highlighting the F1 weighted scores.
- Document the model with the highest score, including its tuned hyperparameters and important features.

### 8. Knowledge Sharing and Future Steps

#### 8.1 Documentation
- Ensure all steps, from data preprocessing to model evaluation, are documented with clear instructions and insights gathered during the process.

#### 8.2 Future Work
- Suggest potential areas for future exploration, such as exploring deep learning approaches or additional feature engineering techniques.

By following this structured plan, AI agents can effectively build, evaluate, and optimize a tabular classification model for handwritten digit recognition using the mfeat-factors dataset.
```
- Plan Execution: **Data Agent**
```
### Detailed Steps for Data Manipulation and Analysis of the mfeat-factors Dataset

#### 1. Dataset Retrieval
- **Source Identification**: The `mfeat-factors` dataset is user-uploaded and available in several CSV files as specified in the potential source list. These files are located in the 'agent_workspace/datasets/mfeat-factors/' directory.
- **Data Collection**: Retrieve the dataset by accessing the specified CSV files. The main files of interest are:
  - `split_train.csv`: Contains the training data.
  - `split_dev.csv`: Contains the development/validation data.
  - `split_test.csv`: Contains the test data.
  - `split_dev_target.csv`, `split_test_target.csv`, `split_train.csv`: Contain the target labels for the respective datasets.
  - `split_dev_wo_target.csv`, `split_test_wo_target.csv`: Contain the features without target labels for the respective datasets.

#### 2. Data Preprocessing
- **Data Cleaning**: 
  - **Verification of Missing Values**: Since the dataset specification indicates no missing values, use a data analysis library like Pandas to confirm this by checking for null values in each column of the dataset.
  - **Importance**: Ensures data integrity and that no information is lost due to missing values.

- **Feature Scaling**:
  - **Standardization or Normalization**: Use a library like Scikit-learn to apply either StandardScaler or MinMaxScaler to the 216 numerical features. Standardization (z-score normalization) centers the data around zero with a standard deviation of one, while normalization scales the data to a range between 0 and 1.
  - **Importance**: Ensures that all features contribute equally to the model's performance, preventing features with larger scales from dominating the learning process.

- **Data Splitting**:
  - **Proportional Splitting**: Use Scikit-learn's `train_test_split` function with the `stratify` parameter to split the dataset into training (80%) and validation (20%) sets, ensuring that the distribution of classes is maintained in both sets.
  - **Importance**: Maintains the representativeness of the dataset and prevents bias in model evaluation.

#### 3. Data Augmentation
- **Class Imbalance Check**: Perform an exploratory data analysis (EDA) to check for class imbalance by examining the distribution of target labels.
- **Synthetic Data Generation**: If class imbalance is detected, consider using SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of the minority class.
- **Importance**: Addresses class imbalance, which can lead to biased models that perform poorly on underrepresented classes.

#### 4. Feature Engineering and Understanding
- **Dimensionality Reduction**:
  - **Principal Component Analysis (PCA)**: Use Scikit-learn's PCA to reduce the dataset's dimensionality while retaining 95% of the variance. This involves calculating the principal components and transforming the data accordingly.
  - **Importance**: Simplifies the model, reduces computation time, and can improve model performance by eliminating noise and redundant features.

- **Feature Selection**:
  - **Recursive Feature Elimination (RFE)**: Use RFE with a base model like Logistic Regression to iteratively remove less important features and select the most significant ones.
  - **Importance**: Identifies the most relevant features, which can improve model interpretability and performance.

- **Exploratory Data Analysis (EDA)**:
  - **Summary Statistics**: Calculate mean, median, and mode for numerical features to understand central tendencies.
  - **Feature Distributions**: Use statistical methods to assess the distribution of features, such as skewness and kurtosis.
  - **Correlation Analysis**: Calculate pairwise correlation metrics (e.g., Pearson correlation) to identify relationships between features.
  - **Importance**: Provides insights into the data's structure, informs feature engineering decisions, and helps identify potential multicollinearity issues.

### Expected Outcomes and Results
- **Quantitative Outcomes**:
  - A clean dataset with standardized or normalized features.
  - A balanced training and validation set with proportional class representation.
  - A reduced feature set that retains 95% of the variance through PCA.
  - A subset of significant features identified through RFE.

- **Qualitative Outcomes**:
  - Enhanced understanding of the dataset's characteristics, such as feature distributions and correlations.
  - Improved model interpretability and performance through effective feature engineering.
  - Confidence in the data's quality and representativeness, leading to more reliable model evaluation and development.

By following these detailed steps, data scientists can effectively prepare the mfeat-factors dataset for building a robust handwritten digit recognition model, ensuring data quality and leveraging the most informative features.
```
- Plan Execution: **Model Agent**
```
To effectively find the top-3 high-performance models for the handwritten digit recognition task using the mfeat-factors dataset, follow these detailed steps:

### 1. Retrieving High-Performance Models

**Model Selection**: 
- Begin by selecting a diverse set of classical machine learning models suitable for tabular data classification. The models to consider include Logistic Regression, Random Forest, Gradient Boosting Machines (GBM), Support Vector Machines (SVM), Decision Trees, and XGBoost. These models are chosen for their proven effectiveness in handling classification tasks and their ability to capture complex patterns in the data.

**Initial Evaluation**:
- Implement each model using the training dataset. Evaluate their baseline performance using the F1 weighted score, which is crucial for handling class imbalance in multi-class classification tasks. This step helps in understanding the initial capability of each model without any tuning.

### 2. Hyperparameter Optimization

**Hyperparameter Tuning**:
- Use GridSearchCV or Random Search to explore the hyperparameter space for each model. The goal is to maximize the F1 weighted score. Here are the specific hyperparameters to optimize for each model:

  - **Random Forest**:
    - Number of trees (`n_estimators`): Optimal value around 100.
    - Maximum depth (`max_depth`): Optimal value around 15.
    - Minimum samples split (`min_samples_split`): Optimal value around 2.

  - **Gradient Boosting Machines (GBM) and XGBoost**:
    - Learning rate (`learning_rate`): Optimal value around 0.1.
    - Number of boosting rounds (`n_estimators`): Optimal value around 100.
    - Maximum depth (`max_depth`): Optimal value around 6.

  - **Support Vector Machines (SVM)**:
    - Kernel type (`kernel`): Optimal choice is 'rbf'.
    - Regularization parameter (`C`): Optimal value around 1.0.
    - Gamma (`gamma`): Optimal value around 'scale'.

  - **Logistic Regression**:
    - Regularization strength (`C`): Optimal value around 1.0.
    - Solver type (`solver`): Optimal choice is 'lbfgs'.

**Threshold Adjustment**:
- Analyze ROC and precision-recall curves to fine-tune the classification threshold, further optimizing the F1 score. This step is essential to ensure that the model's predictions are well-calibrated for the specific class distribution.

### 3. Extracting and Understanding Model Characteristics

**Computation Complexity**:
- Evaluate the computational complexity of each model by considering the number of operations required during training and inference. For example, Random Forest and Decision Trees have complexity related to the number of trees and depth, while SVM's complexity is influenced by the number of support vectors.

**Memory Usage**:
- Assess the memory usage of each model, which depends on the number of parameters and the model's architecture. For instance, Random Forest and XGBoost tend to use more memory due to multiple trees, while Logistic Regression is more memory-efficient.

**Inference Latency**:
- Measure the inference latency, which is the time taken by the model to make predictions. Models like Logistic Regression and SVM with linear kernels typically have lower latency compared to ensemble methods like Random Forest and XGBoost.

### 4. Selecting the Top-k Models

**Model Evaluation**:
- After hyperparameter tuning, evaluate each model using k-fold cross-validation (e.g., k=5) to ensure stability and robustness. Calculate the F1 weighted score for each fold and average the results. This step is crucial for understanding the model's performance across different subsets of the data.

**Performance Reporting**:
- Compile a detailed report of model performances, highlighting the F1 weighted scores. Document the model with the highest score, including its tuned hyperparameters and important features. This report provides a comprehensive overview of each model's strengths and weaknesses.

**Top-k Selection**:
- Rank the models based on their average F1 weighted scores from cross-validation. Select the top-3 models that demonstrate the best balance of precision and recall across classes. This ensures that the selected models are not only accurate but also generalize well to unseen data.

### Expected Performance and Complexity Metrics for Top-3 Models

1. **XGBoost**:
   - F1 Weighted Score: ~0.95
   - Number of Parameters: ~100,000
   - FLOPs: ~10^6
   - Model Size: ~50 MB
   - Training Time: ~30 minutes
   - Inference Speed: ~10 ms per sample

2. **Random Forest**:
   - F1 Weighted Score: ~0.93
   - Number of Parameters: ~50,000
   - FLOPs: ~10^5
   - Model Size: ~100 MB
   - Training Time: ~20 minutes
   - Inference Speed: ~15 ms per sample

3. **SVM (RBF Kernel)**:
   - F1 Weighted Score: ~0.92
   - Number of Parameters: ~10,000
   - FLOPs: ~10^5
   - Model Size: ~10 MB
   - Training Time: ~40 minutes
   - Inference Speed: ~5 ms per sample

These steps and metrics provide a comprehensive guide to selecting and optimizing machine learning models for the task at hand, ensuring that the chosen models are both effective and efficient.
```

#### Tabular Regression (`colleges`)
- Global Plan
```
Here is a comprehensive end-to-end actionable plan for building a tabular regression model to predict `percent_pell_grant` using the colleges dataset. This plan is designed for AI agents capable of conducting machine learning and artificial intelligence research, and it adheres to current state-of-the-art practices.

### 1. Data Retrieval and Initial Inspection
- **Objective**: Obtain and inspect the dataset to understand its structure and content.
- **Instructions**:
  - Load the dataset from the provided source.
  - Analyze the dataset to confirm the number of instances, features, and target variable (`percent_pell_grant`).
  - Identify and quantify missing values and feature types (numeric vs. categorical).

### 2. Data Preprocessing
- **Objective**: Clean and prepare the data for analysis and modeling.
- **Instructions**:
  - **Missing Value Imputation**:
    - Use imputation techniques such as mean, median, KNN, or iterative imputation to handle missing values. Given the high percentage of missing data, evaluate the effectiveness of each method.
  - **Descriptive Analysis**:
    - Conduct a thorough analysis to identify data distribution, patterns, and anomalies.
- **Tools**:
  - Consider using Python libraries such as Pandas and Scikit-learn for preprocessing tasks.

### 3. Feature Engineering
- **Objective**: Enhance the dataset by creating informative features.
- **Instructions**:
  - Create new features from existing ones, especially focusing on geographical and demographic data.
  - Generate interaction terms between geographical information and college statistics.
  - Apply transformations to skewed features to improve model performance.
  - Encode categorical variables using one-hot encoding or entity embedding.
  - Scale numerical features to standardize them.
  
### 4. Feature Selection
- **Objective**: Identify the most relevant features for the model.
- **Instructions**:
  - Use correlation analysis to detect and remove highly correlated features.
  - Employ feature importance techniques from tree-based models like Random Forests to rank features based on their relevance.

### 5. Modeling
- **Objective**: Build and train a regression model.
- **Instructions**:
  - Select classical machine learning algorithms such as Linear Regression, Decision Trees, Random Forests, and Gradient Boosting.
  - Train each model using the processed dataset.
- **Tools**:
  - Utilize Scikit-learn for implementing and training these models.

### 6. Model Evaluation
- **Objective**: Assess model performance and ensure robustness.
- **Instructions**:
  - Evaluate models using RMSE (Root Mean Squared Error) as the primary metric.
  - Implement cross-validation to assess model robustness and generalization to unseen data.

### 7. Hyperparameter Tuning
- **Objective**: Optimize model performance.
- **Instructions**:
  - Conduct hyperparameter optimization using techniques such as grid search or random search to identify optimal settings for each model.

### 8. Model Interpretability
- **Objective**: Provide insights into the model's decision-making process.
- **Instructions**:
  - Analyze feature importance using methods such as SHAP values to understand the impact of each feature on predictions.
  - Document findings to provide insights for stakeholders in the education sector.

### 9. Reporting and Visualization
- **Objective**: Present analysis and model results effectively.
- **Instructions**:
  - Use visualization libraries like Matplotlib or Seaborn to create plots that summarize data distribution, feature importance, and model performance.
  - Compile a comprehensive report detailing the methodology, model performance, and key insights.

### 10. Continuous Improvement
- **Objective**: Enhance model performance over time.
- **Instructions**:
  - Monitor model performance on new data and iteratively refine the model based on feedback and additional data.
  - Stay updated with the latest research in tabular regression and integrate new techniques as appropriate.

By following this detailed plan, AI agents are equipped to build a robust and interpretable regression model tailored to the educational context, delivering accurate predictions and valuable insights.
```
- Plan Execution: **Data Agent**
```
To effectively build a tabular regression model for predicting `percent_pell_grant` using the colleges dataset, follow this detailed plan focusing on data manipulation and analysis:

### 1. Dataset Retrieval

**Objective**: Obtain and inspect the dataset.

**Steps**:
- **Data Source**: The dataset is provided by the user and can be accessed from the specified paths: 
  - `agent_workspace/datasets/colleges/split_dev.csv`
  - `agent_workspace/datasets/colleges/split_dev_target.csv`
  - `agent_workspace/datasets/colleges/split_dev_wo_target.csv`
  - `agent_workspace/datasets/colleges/split_test.csv`
  - `agent_workspace/datasets/colleges/split_test_target.csv`
  - `agent_workspace/datasets/colleges/split_test_wo_target.csv`
  - `agent_workspace/datasets/colleges/split_train.csv`
- **Loading the Data**: Use Python's Pandas library to load the dataset into a DataFrame. This allows for easy manipulation and inspection.
- **Inspecting the Dataset**: Verify that the dataset contains approximately 7,063 instances and 45 features. Check for the presence of the target variable `percent_pell_grant`.
- **Identify Missing Values**: Use Pandas functions to identify missing values in the dataset. This step is crucial as missing data can significantly impact model performance.
- **Categorize Features**: Distinguish between numeric and categorical features. This classification is essential for choosing appropriate preprocessing techniques.

### 2. Data Preprocessing

**Objective**: Clean and prepare the data for analysis and modeling.

**Steps**:
- **Missing Value Imputation**: 
  - Evaluate different imputation techniques such as mean, median, KNN, or iterative imputation using Scikit-learn's `SimpleImputer` or `KNNImputer`. Given the high percentage of missing data (32.8%), it is vital to assess which method best preserves the dataset's integrity and predictive power.
- **Descriptive Analysis**:
  - Conduct a thorough analysis using Pandas to understand data distribution, identify patterns, and detect anomalies. This step helps in understanding the dataset's characteristics and informs subsequent preprocessing decisions.

### 3. Data Augmentation

**Objective**: Enhance the dataset with additional features.

**Steps**:
- **Feature Creation**:
  - Derive new features from existing ones, particularly focusing on geographical and demographic data. For instance, create features that capture regional trends or demographic proportions.
- **Interaction Terms**:
  - Generate interaction terms between geographical information and college statistics. This can uncover hidden relationships that might be predictive of the target variable.
- **Transformations**:
  - Apply transformations to skewed features to normalize their distribution, which can improve model performance. Common transformations include logarithmic or Box-Cox transformations.
- **Encoding Categorical Variables**:
  - Use one-hot encoding for categorical variables with a small number of categories. For high cardinality features, consider entity embedding techniques.
- **Scaling Numerical Features**:
  - Standardize numerical features using techniques such as Min-Max scaling or Z-score normalization to ensure that all features contribute equally to the model.

### 4. Feature Extraction and Understanding

**Objective**: Extract and understand the underlying characteristics of the dataset.

**Steps**:
- **Correlation Analysis**:
  - Perform correlation analysis using Pandas to detect and remove highly correlated features. This reduces multicollinearity, which can adversely affect model performance.
- **Feature Importance**:
  - Use tree-based models like Random Forests to rank features based on their relevance. Scikit-learn's `RandomForestRegressor` can be used to extract feature importances.
- **SHAP Values**:
  - Analyze feature importance using SHAP (SHapley Additive exPlanations) values to understand the impact of each feature on predictions. This provides insights into which features are most influential and why.

### Expected Outcomes and Results

**Quantitative Outcomes**:
- A cleaned and preprocessed dataset ready for modeling, with missing values appropriately imputed and features standardized.
- A set of augmented features that potentially improve model performance by capturing additional information.

**Qualitative Outcomes**:
- A deeper understanding of the dataset's structure, distribution, and underlying patterns.
- Identification of key features that drive the prediction of `percent_pell_grant`, providing insights into the factors influencing this educational metric.

By following these steps, you ensure a robust and interpretable regression model tailored to the educational context, with a focus on data integrity and feature relevance.
```
- Plan Execution: **Model Agent**
```
To effectively find the top-3 high-performance models for a tabular regression task, we will follow a structured approach focusing on candidate model selection, hyperparameter optimization, and understanding model characteristics. Here is a detailed explanation of each step:

### 1. Candidate Model Selection

**Selection of Models:**
- **Linear Regression**: A simple yet effective model for linear relationships. It serves as a baseline model.
- **Decision Trees**: Useful for capturing non-linear relationships and interactions between features.
- **Random Forests**: An ensemble of decision trees that improves robustness and accuracy by averaging predictions.
- **Gradient Boosting**: Another ensemble method that builds trees sequentially, focusing on correcting errors made by previous trees.

**Initial Model Evaluation:**
- Train each model using the provided dataset.
- Perform k-fold cross-validation (e.g., k=5) to ensure that the evaluation is robust and not biased by a particular train-test split.
- Use Root Mean Square Error (RMSE) as the evaluation metric to measure the model's prediction accuracy.

### 2. Hyperparameter Optimization

**Hyperparameter Tuning:**
- **Linear Regression**: Typically does not require hyperparameter tuning, but regularization techniques like Lasso or Ridge can be considered. For Ridge, optimize the regularization parameter `alpha`. Optimal value: `alpha = 1.0`.
- **Decision Trees**: Optimize `max_depth`, `min_samples_split`, and `min_samples_leaf`. Optimal values: `max_depth = 10`, `min_samples_split = 2`, `min_samples_leaf = 1`.
- **Random Forests**: Optimize `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. Optimal values: `n_estimators = 100`, `max_depth = 20`, `min_samples_split = 2`, `min_samples_leaf = 1`.
- **Gradient Boosting**: Optimize `n_estimators`, `learning_rate`, `max_depth`, and `subsample`. Optimal values: `n_estimators = 100`, `learning_rate = 0.1`, `max_depth = 3`, `subsample = 0.8`.

**Optimization Techniques:**
- Use `GridSearchCV` or `RandomizedSearchCV` to systematically explore the hyperparameter space.
- For more complex models like Gradient Boosting, consider Bayesian Optimization to efficiently navigate the hyperparameter space.

### 3. Extracting and Understanding Model Characteristics

**Computation Complexity and Memory Usage:**
- **Linear Regression**: Low computational complexity and memory usage. Suitable for large datasets.
- **Decision Trees**: Complexity grows with the depth of the tree. Memory usage depends on the number of nodes.
- **Random Forests**: Higher complexity due to multiple trees. Memory usage increases with the number of trees.
- **Gradient Boosting**: Similar complexity to Random Forests but more sensitive to hyperparameters like learning rate.

**Inference Latency:**
- **Linear Regression**: Fast inference due to simple linear calculations.
- **Decision Trees**: Moderate inference speed, depending on tree depth.
- **Random Forests**: Slower inference due to averaging over many trees.
- **Gradient Boosting**: Typically slower than Random Forests due to sequential tree building.

### 4. Selection of Top-k Models

**Model Evaluation and Ranking:**
- After hyperparameter tuning, evaluate each model using RMSE on a validation set.
- Rank models based on RMSE, selecting the top-3 models with the lowest RMSE values.

**Consideration of Model Interpretability:**
- Consider the trade-off between model performance and interpretability, especially in educational contexts where understanding predictions is crucial.

**Final Selection:**
- Choose the top-3 models that balance performance and interpretability, offering insights into the factors affecting the target variable.

### Expected Performance and Complexity Metrics

1. **Random Forests**
   - RMSE: Estimated around 0.5 (depends on dataset specifics)
   - Number of Parameters: Varies with `n_estimators` and `max_depth`
   - FLOPs: High due to multiple trees
   - Model Size: Moderate to large
   - Training Time: Moderate
   - Inference Speed: Moderate

2. **Gradient Boosting**
   - RMSE: Estimated around 0.45 (depends on dataset specifics)
   - Number of Parameters: Varies with `n_estimators` and `max_depth`
   - FLOPs: High due to sequential tree building
   - Model Size: Moderate
   - Training Time: Longer than Random Forests
   - Inference Speed: Slower

3. **Linear Regression (with Ridge)**
   - RMSE: Estimated around 0.6 (depends on dataset specifics)
   - Number of Parameters: Low
   - FLOPs: Low
   - Model Size: Small
   - Training Time: Fast
   - Inference Speed: Fast

By following these steps, you can effectively identify and optimize the top-3 models suitable for the given regression task.
```

---

### Entry Role with Instruction Steps
> You are a {`job position`} of a automated machine learning project (AutoML). You have main responsibilities as follows.
> _...agent-specific steps..._

#### Tabular Classification (`mfeat-factors`)
- Global Plan
```
### End-to-End Actionable Plan for AI Agents

#### Objective
Develop a robust solution for tabular classification in handwritten digit recognition using the mfeat-factors dataset. The primary goal is to predict the target column 'class' with a focus on maximizing the weighted F1 score.

#### 1. Data Retrieval and Initial Analysis
- **Data Source**: Retrieve the mfeat-factors dataset from the user-uploaded source.
- **Dataset Overview**: Confirm dataset specifications:
  - Instances: 2,000
  - Features: 217
  - Classes: 10 (digits 0-9)
  - No missing values
- **Initial Analysis**:
  - Conduct exploratory data analysis (EDA) to understand feature distributions and potential correlations.
  - Generate summary statistics and visualizations (e.g., histograms, pair plots) to identify patterns or anomalies.

#### 2. Data Preprocessing
- **Normalization/Standardization**:
  - Apply Min-Max scaling or Z-score normalization to ensure all features have consistent input scales.
- **Outlier Detection**:
  - Use methods like Z-score analysis or IQR to detect and handle outliers.
- **Feature Encoding**:
  - Ensure any categorical features (if applicable) are properly encoded.

#### 3. Feature Engineering
- **Dimensionality Reduction**:
  - Implement Principal Component Analysis (PCA) to reduce dimensionality while retaining the most informative features.
  - Determine the optimal number of components that explain a significant variance (e.g., 95%).
- **Feature Selection**:
  - Utilize methods like Recursive Feature Elimination (RFE) to select the most relevant features for the classification task.

#### 4. Model Selection and Training
- **Model Candidates**:
  - Consider classical machine learning models: k-Nearest Neighbors (kNN), Support Vector Machines (SVM), Random Forests, and Gradient Boosting Machines (GBM).
- **Ensemble Methods**:
  - Explore ensemble techniques (e.g., Random Forests, GBM) for improved robustness and handling of high-dimensional data.
- **Automated Model Selection**:
  - Use AutoML frameworks to automate the model selection process and identify the best-performing model based on the weighted F1 score.

#### 5. Hyperparameter Tuning
- **Optimization Strategy**:
  - Implement cross-validation techniques (e.g., Grid Search, Random Search) to fine-tune hyperparameters.
  - Focus on optimizing parameters that significantly impact model performance, such as the number of trees in Random Forests or the learning rate in GBMs.

#### 6. Model Evaluation
- **Evaluation Metric**:
  - Calculate the weighted F1 score to ensure a balanced measure of precision and recall across all classes.
- **Cross-Validation**:
  - Conduct k-fold cross-validation to assess model performance and ensure generalizability.
- **Performance Analysis**:
  - Analyze confusion matrices and classification reports to identify areas for improvement.

#### 7. Implementation and Deployment
- **Model Implementation**:
  - Utilize Scikit-learn for efficient model implementation and ensure code is optimized for performance.
- **Documentation**:
  - Thoroughly document each step of the process, including data preprocessing, feature engineering, model training, and evaluation results.
- **Deployment Considerations**:
  - If deployment is required, prepare the model for deployment by considering potential endpoints and inference engines suitable for the user's target device.

By adhering to this comprehensive plan, AI agents can effectively develop and deploy a high-performing tabular classification model for handwritten digit recognition using the mfeat-factors dataset.
```
- Plan Execution: **Data Agent**
```
### Explanation of the Machine Learning Development Plan

#### 1. How to Retrieve or Collect the Dataset(s)?

**Data Source and Retrieval**:  
The dataset named 'mfeat-factors' is provided by the user through an upload. It is crucial to access this user-uploaded source to retrieve the dataset. The dataset is split into multiple CSV files located in the following paths:
- `agent_workspace/datasets/mfeat-factors/split_dev.csv`
- `agent_workspace/datasets/mfeat-factors/split_dev_target.csv`
- `agent_workspace/datasets/mfeat-factors/split_dev_wo_target.csv`
- `agent_workspace/datasets/mfeat-factors/split_test.csv`
- `agent_workspace/datasets/mfeat-factors/split_test_target.csv`
- `agent_workspace/datasets/mfeat-factors/split_test_wo_target.csv`
- `agent_workspace/datasets/mfeat-factors/split_train.csv`

Ensure to load these files using a data manipulation library such as Pandas in Python to handle CSV files effectively. Verify that the dataset contains 2,000 instances, 217 features, and 10 classes (digits 0-9). Additionally, confirm there are no missing values to ensure data integrity.

**Importance**: Proper retrieval ensures that the dataset is accurately loaded for subsequent preprocessing and analysis steps, maintaining data integrity and consistency.

#### 2. How to Preprocess the Retrieved Dataset(s)?

**Normalization/Standardization**:  
Apply Min-Max scaling or Z-score normalization to all features. Min-Max scaling transforms the features to a range between 0 and 1, while Z-score normalization standardizes the features to have a mean of 0 and a standard deviation of 1. Use libraries like Scikit-learn in Python for these transformations.

**Outlier Detection**:  
Identify and handle outliers using statistical methods such as Z-score analysis or the Interquartile Range (IQR). Z-score analysis flags data points that are several standard deviations away from the mean, while IQR identifies outliers as those outside 1.5 times the IQR above the third quartile or below the first quartile.

**Feature Encoding**:  
Check for any categorical features and encode them using techniques such as one-hot encoding if necessary. This step is crucial for algorithms that require numerical input.

**Importance**: Preprocessing ensures that the data is in a suitable format for analysis, improving model performance and accuracy by providing consistent input scales and handling outliers that could skew results.

#### 3. How to Efficiently Augment the Dataset(s)?

**Data Augmentation**:  
The plan does not specify augmentation techniques, indicating that augmentation may not be necessary for this tabular dataset. Focus on preprocessing and feature engineering instead.

**Importance**: Data augmentation is typically more relevant for image or text data. For tabular data, ensuring high-quality preprocessing and feature engineering is often more beneficial.

#### 4. How to Extract and Understand the Underlying Characteristics of the Dataset(s)?

**Exploratory Data Analysis (EDA)**:  
Conduct EDA to understand feature distributions and potential correlations. Generate summary statistics such as mean, median, variance, and standard deviation to identify patterns or anomalies. Use correlation matrices to explore relationships between features.

**Feature Engineering**:  
Implement Principal Component Analysis (PCA) to reduce dimensionality while retaining the most informative features. Determine the optimal number of components that explain a significant variance, typically 95%. Use Recursive Feature Elimination (RFE) to select the most relevant features for the classification task.

**Importance**: EDA and feature engineering are critical for understanding the dataset's structure and relationships, enabling more effective model training and improving interpretability and performance.

### Expected Outcomes and Results

**Quantitative Outcomes**:
- A dataset with normalized features, free of outliers, and properly encoded categorical variables.
- A reduced set of features through PCA that captures the majority of variance, facilitating efficient model training.
- Identification of the most relevant features through RFE, enhancing model accuracy.

**Qualitative Outcomes**:
- A comprehensive understanding of the dataset's structure and characteristics.
- Insights into feature importance and relationships, aiding in model interpretability.
- A well-prepared dataset that is ready for model training and evaluation, ensuring robust and reliable results.

By following these detailed steps, another data scientist can reproduce the preprocessing and analysis process, ensuring consistency and accuracy in the machine learning development plan.
```
- Plan Execution: **Model Agent**
```
### 1. Retrieving High-Performance Models

To retrieve high-performance models for the given task, we will focus on classical machine learning models known for their robustness and efficiency in handling tabular data, specifically for classification tasks. The models considered are k-Nearest Neighbors (kNN), Support Vector Machines (SVM), Random Forests, and Gradient Boosting Machines (GBM). These models are widely recognized for their ability to handle various data distributions and complexities.

**Model Candidates:**
- **k-Nearest Neighbors (kNN):** Known for its simplicity and effectiveness in capturing local data structures. It is particularly useful when the decision boundary is non-linear.
- **Support Vector Machines (SVM):** Effective in high-dimensional spaces and suitable for cases where the number of dimensions exceeds the number of samples.
- **Random Forests:** An ensemble method that combines multiple decision trees to improve robustness and accuracy. It is effective in handling overfitting.
- **Gradient Boosting Machines (GBM):** Known for its ability to handle various types of data distributions and its effectiveness in improving prediction accuracy through boosting.

**Automated Model Selection:**
To automate the selection process, we will utilize AutoML frameworks that can efficiently evaluate multiple models and configurations. These frameworks will help identify the best-performing models based on the weighted F1 score, which balances precision and recall across all classes.

### 2. Hyperparameter Optimization

To optimize the hyperparameters of the retrieved models, we will implement cross-validation techniques such as Grid Search or Random Search. These techniques will help fine-tune the parameters that significantly impact model performance.

**Optimization Strategy:**
- **k-Nearest Neighbors (kNN):**
  - Number of neighbors (`n_neighbors`): Optimal value estimated to be 5. This parameter controls the number of nearest neighbors considered for classification.
  - Distance metric (`metric`): Optimal value is 'euclidean'. This determines how distances between points are calculated.

- **Support Vector Machines (SVM):**
  - Kernel type (`kernel`): Optimal value is 'rbf'. The radial basis function kernel is effective for non-linear decision boundaries.
  - Regularization parameter (`C`): Optimal value estimated to be 1.0. This parameter controls the trade-off between achieving a low training error and a low testing error.

- **Random Forests:**
  - Number of trees (`n_estimators`): Optimal value estimated to be 100. This parameter determines the number of trees in the forest.
  - Maximum depth (`max_depth`): Optimal value estimated to be 10. This controls the depth of each tree, affecting the model's ability to generalize.

- **Gradient Boosting Machines (GBM):**
  - Learning rate (`learning_rate`): Optimal value estimated to be 0.1. This parameter controls the contribution of each tree to the final prediction.
  - Number of boosting stages (`n_estimators`): Optimal value estimated to be 100. This determines the number of boosting iterations.

**Cross-Validation:**
We will use k-fold cross-validation to ensure that the hyperparameter tuning process is robust and generalizable across different data splits. This approach helps in mitigating overfitting and provides a reliable estimate of model performance.

### 3. Understanding Model Characteristics

To extract and understand the characteristics of the candidate models, we will focus on computational complexity, memory usage, and inference latency.

**k-Nearest Neighbors (kNN):**
- **Computation Complexity:** O(n * d), where n is the number of samples and d is the number of features. This is due to the need to compute distances for each query.
- **Memory Usage:** High, as it stores all training data.
- **Inference Latency:** High, as it requires computing distances to all training samples for each prediction.

**Support Vector Machines (SVM):**
- **Computation Complexity:** O(n^2 * d) for training, where n is the number of samples and d is the number of features. This is due to the quadratic programming involved in training.
- **Memory Usage:** Moderate, as it primarily depends on the number of support vectors.
- **Inference Latency:** Moderate, as it depends on the number of support vectors used in decision making.

**Random Forests:**
- **Computation Complexity:** O(n * m * log(n)), where n is the number of samples and m is the number of trees. This is due to the construction of multiple decision trees.
- **Memory Usage:** Moderate, as it stores multiple trees.
- **Inference Latency:** Low, as predictions are made by averaging results from multiple trees.

**Gradient Boosting Machines (GBM):**
- **Computation Complexity:** O(n * m * log(n)), where n is the number of samples and m is the number of boosting stages. This is due to the iterative nature of boosting.
- **Memory Usage:** Moderate, as it stores multiple trees.
- **Inference Latency:** Moderate, as predictions are made by summing contributions from multiple trees.

### 4. Selecting the Top-3 Models

**Evaluation Metric:**
The weighted F1 score will be used as the primary evaluation metric to assess model performance. This metric provides a balanced measure of precision and recall across all classes.

**Performance Analysis:**
We will analyze confusion matrices and classification reports to gain insights into model performance and areas for improvement.

**Top-3 Selection:**
Based on the evaluation results, we will select the top-3 models that achieve the highest weighted F1 scores. We will consider both quantitative outcomes (e.g., F1 score) and qualitative insights (e.g., model interpretability, feature importance) when making the final selection.

### Expected Performance of Top-3 Models

1. **Random Forests:**
   - Weighted F1 Score: Estimated 0.92
   - Number of Parameters: Approximately 10,000
   - FLOPs: Estimated 1.5 million
   - Model Size: Approximately 50 MB
   - Training Time: Approximately 10 minutes
   - Inference Speed: Approximately 0.1 seconds per sample

2. **Gradient Boosting Machines (GBM):**
   - Weighted F1 Score: Estimated 0.90
   - Number of Parameters: Approximately 15,000
   - FLOPs: Estimated 2 million
   - Model Size: Approximately 60 MB
   - Training Time: Approximately 15 minutes
   - Inference Speed: Approximately 0.2 seconds per sample

3. **Support Vector Machines (SVM):**
   - Weighted F1 Score: Estimated 0.88
   - Number of Parameters: Depends on support vectors, estimated 5,000
   - FLOPs: Estimated 1 million
   - Model Size: Approximately 30 MB
   - Training Time: Approximately 20 minutes
   - Inference Speed: Approximately 0.15 seconds per sample

These estimates are based on typical performance metrics for these models on similar datasets and tasks.
```

#### Tabular Regression (`colleges`)
- Global Plan
```
To develop an end-to-end plan for building a predictive model for the `percent_pell_grant` target variable using the colleges dataset, the following structured plan is devised for AI agents to execute:

### 1. Data Retrieval and Understanding
- **Data Source Identification:** Retrieve the `colleges` dataset as specified in the user-upload source.
- **Data Summary and Initial Exploration:** Conduct a preliminary analysis to understand the dataset's structure, including the number of features, types of features, and the target variable.

### 2. Data Preprocessing
- **Handling Missing Values:** 
  - Implement imputation strategies to handle the 32.8% missing values. Evaluate methods such as mean, median, mode, KNN, or iterative imputation to fill missing entries.
  
- **Feature Scaling:**
  - Scale numerical features using MinMaxScaler or StandardScaler, ensuring feature magnitude consistency across the dataset.

- **Encoding Categorical Variables:**
  - Convert categorical features into numerical format using one-hot encoding or ordinal encoding.

### 3. Feature Engineering
- **Descriptive Analysis:**
  - Conduct in-depth analysis to discover patterns, anomalies, and relationships in the data.

- **Feature Selection:**
  - Utilize Recursive Feature Elimination (RFE) or other feature selection techniques to identify and retain the most relevant features.

- **Feature Transformation:**
  - Apply transformations like logarithmic or Box-Cox to manage skewness in data distributions.
  - Consider binning continuous variables to capture potential non-linear relationships.

- **Entity Embedding:**
  - Implement entity embeddings for categorical variables to capture intricate relationships and interactions.

### 4. Modeling
- **Model Selection:**
  - Choose classical machine learning models such as Linear Regression, Decision Trees, Random Forests, Gradient Boosting Machines (GBM), XGBoost, or LightGBM for their robustness in handling tabular data.

- **Automated Machine Learning:**
  - Leverage an AutoML platform to facilitate model selection and hyperparameter optimization, aligning with the user's medium expertise level.

- **Validation Strategy:**
  - Employ k-fold cross-validation to ensure model generalizability and avoid overfitting.

### 5. Model Evaluation and Interpretation
- **Performance Metric:**
  - Evaluate the model using Root Mean Square Error (RMSE) as the primary performance metric to quantify prediction accuracy.

- **Interpretability:**
  - Utilize SHAP (SHapley Additive exPlanations) to interpret model predictions, aiding in understanding the impact of each feature on the target variable.

### 6. Automation and Enhancement Tools
- **Automated Feature Engineering:**
  - Implement automated feature engineering tools, such as LLM-FE, to enhance feature discovery and transformation.

- **Feature Importance Analysis:**
  - Analyze feature importance scores from models like Random Forests and XGBoost to discern influential features.

### 7. Reporting and Documentation
- **Comprehensive Reporting:**
  - Compile a detailed report summarizing the data preprocessing steps, feature engineering techniques, model selection, evaluation results, and interpretability insights.

- **Documentation for Reproducibility:**
  - Ensure all steps and processes are well-documented to facilitate reproducibility and future enhancements.

By following this comprehensive plan, AI agents can effectively build a predictive model for the `percent_pell_grant` target variable, ensuring robust preprocessing, insightful feature engineering, and accurate model evaluation.
```
- Plan Execution: **Data Agent**
```
### Detailed Steps for Data Manipulation and Analysis

#### 1. Dataset Retrieval

- **Source Identification:** 
  The dataset named `colleges` has been uploaded by the user. It is accessible through the specified file paths:
  - `agent_workspace/datasets/colleges/split_dev.csv`
  - `agent_workspace/datasets/colleges/split_dev_target.csv`
  - `agent_workspace/datasets/colleges/split_dev_wo_target.csv`
  - `agent_workspace/datasets/colleges/split_test.csv`
  - `agent_workspace/datasets/colleges/split_test_target.csv`
  - `agent_workspace/datasets/colleges/split_test_wo_target.csv`
  - `agent_workspace/datasets/colleges/split_train.csv`

- **Initial Exploration:**
  Begin by loading the dataset using a data manipulation library such as Pandas. Inspect the dataset to understand its structure, focusing on:
  - Number of features and their data types (numerical and categorical).
  - Identify the target variable, which is `percent_pell_grant`.
  - Check for any obvious data quality issues, such as missing values or inconsistent data types.

#### 2. Data Preprocessing

- **Handling Missing Values:**
  Address the 32.8% missing values by implementing imputation strategies:
  - Use Pandas or Scikit-learn's `SimpleImputer` for mean, median, or mode imputation.
  - For more sophisticated imputation, consider using Scikit-learn's `KNNImputer` or `IterativeImputer` to predict missing values based on other features.
  - The choice of imputation method should depend on the nature of the data and the missingness mechanism.

- **Feature Scaling:**
  Normalize numerical features to ensure consistent feature magnitudes:
  - Use Scikit-learn's `MinMaxScaler` to scale features to a range between 0 and 1.
  - Alternatively, use `StandardScaler` to standardize features by removing the mean and scaling to unit variance.
  - This step is crucial for algorithms sensitive to feature scaling, such as KNN or gradient descent-based models.

- **Encoding Categorical Variables:**
  Convert categorical features to numerical format:
  - Use Pandas `get_dummies` or Scikit-learn's `OneHotEncoder` for one-hot encoding.
  - For ordinal data, use `OrdinalEncoder` to maintain the order of categories.
  - Encoding is necessary to convert categorical data into a format that can be provided to machine learning algorithms.

#### 3. Data Augmentation

- **Entity Embedding:**
  Implement entity embeddings for categorical variables:
  - Use deep learning frameworks like TensorFlow or PyTorch to create embeddings.
  - Entity embeddings can capture complex relationships and interactions that traditional encoding methods might miss.
  - This step is particularly useful if the dataset will be used with neural network models.

#### 4. Feature Engineering and Understanding

- **Descriptive Analysis:**
  Perform an in-depth analysis using Pandas and visualization libraries like Matplotlib or Seaborn:
  - Analyze distributions, correlations, and potential outliers.
  - Identify patterns, anomalies, and relationships within the data.
  - This step helps in understanding the data better and informs subsequent feature engineering.

- **Feature Selection:**
  Use techniques like Recursive Feature Elimination (RFE):
  - Implement RFE using Scikit-learn to select features by recursively considering smaller sets of features.
  - This helps in identifying and retaining the most relevant features for the predictive model.

- **Feature Transformation:**
  Apply transformations to manage skewness:
  - Use NumPy or Scipy to apply logarithmic or Box-Cox transformations.
  - Consider binning continuous variables using Pandas `cut` or `qcut` to capture potential non-linear relationships.
  - Transformations can improve model performance by stabilizing variance and normalizing distributions.

- **Feature Importance Analysis:**
  Analyze feature importance scores:
  - Use models like Random Forests or XGBoost from Scikit-learn or XGBoost library to compute feature importance.
  - This analysis helps in understanding which features have the most influence on the target variable.

### Expected Outcomes and Results

- **Quantitative Outcomes:**
  - A cleaned and preprocessed dataset with imputed missing values and scaled features.
  - Categorical variables converted into numerical format suitable for modeling.
  - A set of selected and transformed features ready for model training.

- **Qualitative Outcomes:**
  - Enhanced understanding of the dataset's structure and characteristics.
  - Identification of key features and their relationships with the target variable.
  - Insights into potential patterns and anomalies within the data.

By following these detailed steps, another data scientist can effectively preprocess the `colleges` dataset, ensuring robust data manipulation and insightful feature engineering. This preparation is crucial for building accurate and interpretable predictive models.
```
- Plan Execution: **Model Agent**
```
To effectively find the top-3 high-performance models or algorithms for the given task, we will follow a structured approach focusing on model selection, hyperparameter optimization, and understanding the characteristics of the candidate models. Here is a detailed explanation of each step:

### 1. Retrieve or Find the High-Performance Models

**Model Selection:**
- **Linear Regression:** A simple yet effective model for regression tasks, especially when the relationship between features and the target is linear.
- **Random Forests:** Known for handling non-linear relationships and interactions between features, which can be beneficial for tabular data.
- **XGBoost:** A powerful gradient boosting algorithm that often outperforms other models in tabular data tasks due to its ability to handle complex patterns.

**Rationale:** These models are chosen based on their robustness and ability to handle different data complexities, aligning with the user's requirement for tabular regression. They are well-suited for initial exploration due to their varying complexity and interpretability.

### 2. Optimize the Hyperparameters of the Retrieved Models

**Hyperparameter Tuning:**
- **Linear Regression:**
  - Regularization Strength (`alpha`): Optimize using a range from 0.01 to 1.0. Optimal value: 0.1.
  
- **Random Forests:**
  - Number of Trees (`n_estimators`): Test values between 100 and 500. Optimal value: 300.
  - Maximum Depth (`max_depth`): Explore depths from 5 to 50. Optimal value: 20.
  - Minimum Samples per Leaf (`min_samples_leaf`): Range from 1 to 10. Optimal value: 2.
  
- **XGBoost:**
  - Learning Rate (`eta`): Test values from 0.01 to 0.3. Optimal value: 0.1.
  - Maximum Depth (`max_depth`): Range from 3 to 10. Optimal value: 6.
  - Subsample Ratio (`subsample`): Values between 0.5 and 1.0. Optimal value: 0.8.
  - Number of Estimators (`n_estimators`): Range from 100 to 1000. Optimal value: 500.

**Rationale:** Hyperparameter tuning is essential to enhance model performance by finding the best configuration for each model. The chosen ranges and optimal values are based on typical values that balance model complexity and performance.

### 3. Extract and Understand the Characteristics of the Candidate Models

**Computation Complexity and Memory Usage:**
- **Linear Regression:** Low computational complexity with O(n) operations, where n is the number of features. Memory usage is minimal, as it only requires storing the coefficients.
  
- **Random Forests:** Moderate complexity due to ensemble learning, with O(nt) operations, where n is the number of trees and t is the depth of each tree. Memory usage is higher due to storing multiple decision trees.
  
- **XGBoost:** High computational complexity, especially with large datasets, due to boosting iterations and tree construction. Memory usage is significant, as it stores multiple boosted trees and requires additional memory for gradient and Hessian calculations.

**Inference Latency:**
- **Linear Regression:** Fast inference speed due to simple linear calculations.
- **Random Forests:** Moderate inference speed, as predictions are averaged over multiple trees.
- **XGBoost:** Slower inference speed compared to Random Forests, due to sequential tree evaluation.

**Rationale:** Understanding these characteristics helps in selecting models that not only perform well but also fit within the resource constraints of the deployment environment.

### 4. Select the Top-3 Models or Algorithms

**Performance Evaluation:**
- **Root Mean Square Error (RMSE):** Used as the primary metric to evaluate model accuracy.
- **Model Interpretability:** Considered using feature importance and SHAP values to ensure models are interpretable.

**Top-3 Models:**
1. **XGBoost:** Expected RMSE: 0.15, Number of Parameters: ~10,000, FLOPs: ~1.5 million, Model Size: ~50 MB, Training Time: ~2 hours, Inference Speed: ~50 ms per instance.
2. **Random Forests:** Expected RMSE: 0.18, Number of Parameters: ~5,000, FLOPs: ~1 million, Model Size: ~30 MB, Training Time: ~1.5 hours, Inference Speed: ~30 ms per instance.
3. **Linear Regression:** Expected RMSE: 0.25, Number of Parameters: Equal to the number of features, FLOPs: ~10,000, Model Size: ~1 MB, Training Time: ~10 minutes, Inference Speed: ~5 ms per instance.

**Rationale:** These models are selected based on their balance of performance, computational efficiency, and interpretability, ensuring they meet the user's requirements for the task.
```

---

### Senior Role with Instruction Steps
> You are an experienced {`job position`} of a automated machine learning project (AutoML). You have main responsibilities as follows.
> _...agent-specific steps..._

#### Tabular Classification (`mfeat-factors`)
- Global Plan
```
### End-to-End Plan for AI Agents: Tabular Classification using mfeat-factors Dataset

#### **Objective**
Develop a robust tabular classification model using the mfeat-factors dataset to predict the target column 'class' accurately. The objective is to maximize the f1 weighted score on the evaluation dataset.

#### **1. Data Retrieval and Understanding**
- **Data Source**: Retrieve the mfeat-factors dataset from the user-uploaded source.
- **Dataset Description**: Understand that the dataset includes 2,000 instances with 217 features, primarily numeric, representing handwritten numerals (0-9).

#### **2. Exploratory Data Analysis (EDA)**
- **Statistical Summary**: Compute descriptive statistics for numeric features to understand their distributions.
- **Visualization**: Generate visualizations such as histograms, box plots, and pair plots to explore feature distributions and relationships.
- **Correlation Analysis**: Calculate correlation coefficients to identify potential multicollinearity among features.

#### **3. Data Preprocessing**
- **Normalization/Standardization**: Apply Min-Max scaling or Z-score normalization to ensure all features contribute equally to the model.
- **Categorical Encoding**: Verify if symbolic features exist; if any, apply one-hot encoding for categorical conversion.
- **Handling Missing Values**: Confirm no missing values using dataset specifications; otherwise, apply imputation strategies.

#### **4. Feature Engineering**
- **Dimensionality Reduction**: Use Principal Component Analysis (PCA) to reduce dimensionality, retaining components that capture 95% of variance.
- **Feature Selection**: Apply techniques like Recursive Feature Elimination (RFE) or feature importance from tree-based models to select top features.

#### **5. Model Selection and Training**
- **Model Candidates**: Evaluate classical machine learning models such as:
  - Support Vector Machines (SVM)
  - Decision Trees
  - k-Nearest Neighbors (KNN)
  - Random Forests
  - Gradient Boosting Machines (GBM)
- **Ensemble Methods**: Consider Random Forests and XGBoost for leveraging ensemble techniques to improve accuracy.
- **Training**: Implement cross-validation (e.g., 5-fold) to train models while preventing overfitting.

#### **6. Hyperparameter Optimization**
- **Tuning**: Employ Grid Search or Random Search for hyperparameter tuning, focusing on parameters that influence model complexity and learning rate.
- **Metric Focus**: Optimize for the f1 weighted score, balancing precision and recall for multi-class classification.

#### **7. Model Evaluation**
- **Performance Measurement**: Evaluate models using the f1 weighted score on a validation dataset to ensure the model's robustness.
- **Iteration**: Based on evaluation, iterate on feature selection, preprocessing, and hyperparameter tuning to enhance performance.

#### **8. Deployment Considerations**
- **Model Efficiency**: Ensure the selected model is computationally efficient for deployment, considering potential resource constraints.
- **Implementation**: Prepare the model for deployment by serializing it with tools like joblib or pickle, ensuring compatibility with the intended inference engine.

#### **9. Documentation and Reporting**
- **Comprehensive Report**: Document the entire process, from EDA to model evaluation, including insights, decisions made, and performance metrics.
- **User Demonstration**: Develop a simple demonstration for users with medium expertise to understand and interact with the model, possibly using a web-based interface.

By following these structured steps, AI agents can effectively build a solution that meets the user's requirements for tabular classification in handwritten digit recognition using the mfeat-factors dataset.
```
- Plan Execution: **Data Agent**
```
### Detailed Steps for Data Manipulation and Analysis

#### 1. Retrieving the Dataset

To retrieve the mfeat-factors dataset, you need to access the user-uploaded source. The dataset is split into multiple CSV files located in the directory specified by the user: `agent_workspace/datasets/mfeat-factors/`. The relevant files are:

- `split_dev.csv`
- `split_dev_target.csv`
- `split_dev_wo_target.csv`
- `split_test.csv`
- `split_test_target.csv`
- `split_test_wo_target.csv`
- `split_train.csv`

These files contain the data for development, testing, and training purposes. Ensure you have access to this directory to collect the dataset. The dataset consists of 2,000 instances and 217 features, primarily numeric, representing handwritten numerals (0-9).

#### 2. Preprocessing the Dataset

**Normalization/Standardization**: 

- **Why**: Normalization or standardization is crucial for ensuring that each feature contributes equally to the model's performance. This is especially important for algorithms that rely on distance calculations, such as k-nearest neighbors or support vector machines.
- **How**: Use Min-Max scaling to transform the features to a range between 0 and 1, or apply Z-score normalization to center the data around a mean of 0 with a standard deviation of 1. Libraries such as scikit-learn provide functions like `MinMaxScaler` and `StandardScaler` to perform these transformations.

**Categorical Encoding**:

- **Why**: If there are any categorical features, they need to be converted into a numerical format to be used in machine learning models.
- **How**: Check the dataset for symbolic features. If present, apply one-hot encoding using libraries like pandas or scikit-learn's `OneHotEncoder` to convert these features into a binary matrix.

**Handling Missing Values**:

- **Why**: Missing values can lead to inaccurate model predictions and biases.
- **How**: Verify the dataset specifications to ensure there are no missing values. If missing values are present, use imputation strategies such as mean, median, or mode imputation, or more advanced techniques like K-nearest neighbors imputation, available in libraries like scikit-learn.

#### 3. Data Augmentation

**Dimensionality Reduction**:

- **Why**: Reducing the dimensionality of the dataset helps in decreasing computational complexity and can improve model performance by eliminating noise.
- **How**: Implement Principal Component Analysis (PCA) to reduce the number of features while retaining components that capture 95% of the variance. This can be done using scikit-learn's `PCA` class.

**Feature Selection**:

- **Why**: Selecting the most relevant features can enhance model interpretability and performance by focusing on the most informative parts of the data.
- **How**: Use Recursive Feature Elimination (RFE) or feature importance scores from tree-based models like Random Forests to identify and retain the most significant features. Scikit-learn provides tools like `RFE` and `RandomForestClassifier` for this purpose.

#### 4. Extracting and Understanding Dataset Characteristics

**Statistical Summary**:

- **Why**: Understanding the distribution, central tendencies, and variability of the features helps in identifying potential issues and informs preprocessing decisions.
- **How**: Compute descriptive statistics such as mean, median, standard deviation, and quartiles for all numeric features using pandas' `describe()` function.

**Correlation Analysis**:

- **Why**: Identifying multicollinearity among features is essential as it can affect the stability and performance of the model.
- **How**: Calculate correlation coefficients using pandas' `corr()` function to identify highly correlated features that may need to be removed or combined.

**Feature Relationships**:

- **Why**: Analyzing the relationships between features and the target variable can reveal significant patterns or insights that could inform model development.
- **How**: Use statistical tests or visual analysis (though not performed here) to explore these relationships.

### Expected Outcomes and Results

- **Quantitative**: The dataset will be transformed into a standardized, encoded, and dimensionally reduced form, ready for model training. Key features will be selected, and any multicollinearity issues will be identified and addressed.
  
- **Qualitative**: A comprehensive understanding of the dataset's characteristics will be achieved, including insights into feature distributions, relationships, and potential preprocessing needs. This understanding will guide effective model development and evaluation.
```
- Plan Execution: **Model Agent**
```
To effectively find the top-3 high-performance models or algorithms for the given dataset, we will follow a structured approach focusing on model retrieval, hyperparameter optimization, and understanding the underlying characteristics of the candidate models. The goal is to ensure that the selected models are not only accurate but also efficient in terms of computational resources.

### Step 1: Retrieve High-Performance Models

**Model Candidates**: 
- We will evaluate classical machine learning models suitable for tabular classification tasks. The models considered are Support Vector Machines (SVM), Decision Trees, k-Nearest Neighbors (KNN), Random Forests, and Gradient Boosting Machines (GBM).
- **Ensemble Methods**: Random Forests and XGBoost (a variant of GBM) will be included to leverage ensemble techniques, which often provide better generalization by combining multiple models.

**Model Evaluation**:
- Use 5-fold cross-validation to assess the performance of these models. This technique helps in obtaining a reliable estimate of model performance and reduces the risk of overfitting by ensuring that each data point is used for both training and validation.

### Step 2: Optimize Hyperparameters

**Hyperparameter Tuning**:
- **Support Vector Machines (SVM)**:
  - Kernel type: Optimize between 'linear', 'poly', 'rbf'. Optimal: 'rbf' for non-linear decision boundaries.
  - C (Regularization parameter): Range from 0.1 to 10. Optimal: 1.0 for a balance between margin width and classification error.
  - Gamma (Kernel coefficient): Range from 0.001 to 1. Optimal: 0.01 for controlling the influence of individual data points.
  
- **Random Forests**:
  - Number of trees: Range from 50 to 200. Optimal: 100 for a balance between performance and computation.
  - Max depth: Range from 5 to 30. Optimal: 15 to prevent overfitting while capturing complex patterns.
  - Min samples split: Range from 2 to 10. Optimal: 2 for allowing nodes to split with minimal samples.
  
- **Gradient Boosting Machines (XGBoost)**:
  - Number of trees: Range from 50 to 200. Optimal: 150 for capturing complex patterns.
  - Learning rate: Range from 0.01 to 0.3. Optimal: 0.1 for a balance between convergence speed and accuracy.
  - Max depth: Range from 3 to 10. Optimal: 6 to prevent overfitting while capturing complex patterns.

**Metric Focus**:
- Optimize for the f1 weighted score, which is crucial for multi-class classification tasks as it balances precision and recall.

### Step 3: Extract and Understand Characteristics

**Computation Complexity**:
- **SVM**: Complexity depends on the number of support vectors. Typically O(n^2) for training, where n is the number of samples.
- **Random Forests**: Complexity is O(m * log(n) * d), where m is the number of trees, n is the number of samples, and d is the number of features.
- **XGBoost**: Complexity is O(m * log(n) * d), similar to Random Forests, but with additional overhead due to boosting.

**Memory Usage**:
- **SVM**: High memory usage due to storing support vectors.
- **Random Forests**: Moderate memory usage, proportional to the number of trees and depth.
- **XGBoost**: Moderate to high memory usage due to storing multiple boosted trees.

**Inference Latency**:
- **SVM**: Fast inference for small datasets, but can be slower with many support vectors.
- **Random Forests**: Generally fast due to parallelizable nature.
- **XGBoost**: Slightly slower than Random Forests due to sequential boosting.

### Step 4: Select Top-3 Models

**Performance Measurement**:
- Evaluate models using the f1 weighted score on a validation dataset to ensure robustness.

**Iteration and Improvement**:
- Iterate on feature selection, preprocessing, and hyperparameter tuning based on evaluation results to enhance performance.

**Top-3 Selection**:
- Rank models based on their f1 weighted scores and select the top-3 models that demonstrate the best performance and efficiency for deployment.

### Expected Performance and Complexity Metrics

1. **XGBoost**:
   - f1 weighted score: ~0.92
   - Number of parameters: ~150 trees
   - FLOPs: ~1.5 million
   - Model size: ~50 MB
   - Training time: ~5 minutes
   - Inference speed: ~10 ms per sample

2. **Random Forests**:
   - f1 weighted score: ~0.90
   - Number of parameters: ~100 trees
   - FLOPs: ~1 million
   - Model size: ~40 MB
   - Training time: ~3 minutes
   - Inference speed: ~8 ms per sample

3. **SVM**:
   - f1 weighted score: ~0.88
   - Number of support vectors: ~500
   - FLOPs: ~500,000
   - Model size: ~30 MB
   - Training time: ~10 minutes
   - Inference speed: ~15 ms per sample

By following these steps, you can effectively retrieve, optimize, and select high-performance models that meet the user's requirements for tabular classification tasks.
```

#### Tabular Regression (`colleges`)
- Global Plan
```
Below is a comprehensive end-to-end actionable plan designed for AI agents to implement a solution for predicting the `percent_pell_grant` in the education domain using tabular regression:

### Step 1: Data Understanding and Retrieval

1. **Data Retrieval**:
   - Access the dataset described as a modified version for the AutoML benchmark, regrouping information for approximately 7800 different US colleges.

2. **Initial Data Exploration**:
   - Load the dataset and perform a preliminary inspection to understand its structure, focusing on the target variable `percent_pell_grant`.

### Step 2: Exploratory Data Analysis (EDA)

1. **Descriptive Statistics**:
   - Compute summary statistics for all features to understand data distribution, central tendency, and dispersion.

2. **Visualization**:
   - Generate histograms, box plots, and scatter plots to visualize distributions and relationships among features.
   - Use correlation matrices to identify potential linear relationships between features and the target variable.

### Step 3: Handling Missing Data

1. **Identify Missing Data**:
   - Analyze missing data patterns to determine if they are Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR).

2. **Imputation Strategies**:
   - Use appropriate imputation methods such as:
     - Mean/Median Imputation for numeric features.
     - K-Nearest Neighbors (KNN) for non-linear imputation.
     - Multiple Imputation by Chained Equations (MICE) for complex patterns.
   - Consider encoding missingness as a feature if it provides additional insights.

### Step 4: Feature Engineering

1. **Transformation**:
   - Normalize or standardize numeric features to ensure consistent scale.
   - Apply transformations like log or square root for skewed features.

2. **Feature Creation**:
   - Create interaction terms or polynomial features to capture non-linear relationships.

3. **Feature Selection**:
   - Use Recursive Feature Elimination (RFE) or feature importance scores from tree-based models to identify the most relevant features.

### Step 5: Model Selection and Training

1. **Model Selection**:
   - Choose classical machine learning models suitable for tabular data such as:
     - Linear Regression for baseline performance.
     - Decision Trees for interpretability.
     - Random Forests and Gradient Boosting Machines (GBM) for complex data patterns.
     - XGBoost for enhanced performance on imbalanced data.

2. **Training**:
   - Split the dataset into training and validation sets.
   - Train selected models using the training set.

### Step 6: Model Evaluation

1. **Performance Metrics**:
   - Evaluate models using Root Mean Square Error (RMSE) on the validation set to measure prediction accuracy.

2. **Cross-Validation**:
   - Implement k-fold cross-validation to ensure model robustness and generalizability.

### Step 7: Hyperparameter Tuning

1. **Optimization**:
   - Use grid search or random search to find optimal hyperparameters for the selected models.
   - Leverage cross-validation during the tuning process to validate hyperparameter efficacy.

### Step 8: Iterative Process and Refinement

1. **Model Iteration**:
   - Refine models based on evaluation results, iterating through feature selection, engineering, and hyperparameter tuning as needed.

2. **Final Model Selection**:
   - Choose the best-performing model based on RMSE and other relevant metrics.

### Step 9: Deployment Considerations

1. **Scalability**:
   - Prepare the model for deployment by considering scalability and integration into the user's system.
   - Ensure model inference capabilities are optimized for real-time predictions if necessary.

By following this detailed plan, AI agents can systematically develop, evaluate, and optimize a machine learning solution tailored to the educational domain for predicting the `percent_pell_grant` variable.
```
- Plan Execution: **Data Agent**
```
### Detailed Steps for Data Manipulation and Analysis

#### 1. How to Retrieve or Collect the Dataset(s)?

- **Dataset Access**: The dataset is user-uploaded and can be accessed directly from the specified file paths. The dataset is split into several CSV files located in the `agent_workspace/datasets/colleges/` directory. The files include:
  - `split_dev.csv`
  - `split_dev_target.csv`
  - `split_dev_wo_target.csv`
  - `split_test.csv`
  - `split_test_target.csv`
  - `split_test_wo_target.csv`
  - `split_train.csv`

  These files should be loaded using a data manipulation library such as Pandas, which provides robust tools for handling CSV files.

- **Initial Inspection**: Load the dataset into a Pandas DataFrame to perform a preliminary inspection. This involves checking the first few rows using `head()` to understand the structure, identifying the columns, and focusing on the target variable `percent_pell_grant`. This step is crucial to ensure that the dataset is correctly loaded and to get an initial understanding of its contents.

#### 2. How to Preprocess the Retrieved Dataset(s)?

- **Descriptive Statistics**: Use Pandas' `describe()` function to compute summary statistics for all features. This will provide insights into the data distribution, central tendency (mean, median), and dispersion (standard deviation, range). Understanding these statistics is essential for identifying any anomalies or outliers in the data.

- **Missing Data Handling**:
  - **Identify Missing Data**: Use Pandas functions like `isnull().sum()` to analyze missing data patterns. This helps determine whether the missing data is Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR).
  - **Imputation Strategies**:
    - For numeric features, apply mean or median imputation using Pandas' `fillna()` function to handle missing values.
    - For non-linear imputation, use the K-Nearest Neighbors (KNN) algorithm from the `sklearn.impute` module.
    - For complex patterns, consider Multiple Imputation by Chained Equations (MICE) using the `fancyimpute` library.
    - If missingness itself provides additional insights, encode it as a feature using Pandas' `isnull()` to create a new binary column indicating missingness.

- **Feature Transformation**:
  - Normalize or standardize numeric features using `StandardScaler` or `MinMaxScaler` from `sklearn.preprocessing` to ensure consistent scaling across features.
  - Apply transformations like log or square root using NumPy functions to address skewness in feature distributions. This can help in stabilizing variance and making the data more normally distributed.

#### 3. How to Efficiently Augment the Dataset(s)?

- **Feature Creation**:
  - Create interaction terms or polynomial features using `PolynomialFeatures` from `sklearn.preprocessing` to capture non-linear relationships between features. This can enhance the model's ability to learn complex patterns in the data.

- **Feature Selection**:
  - Use Recursive Feature Elimination (RFE) from `sklearn.feature_selection` or feature importance scores from tree-based models like Random Forests to identify the most relevant features. This step is crucial for reducing dimensionality and improving model performance by focusing on the most informative features.

#### 4. How to Extract and Understand the Underlying Characteristics of the Dataset(s)?

- **Correlation Analysis**: Compute a correlation matrix using Pandas' `corr()` function to identify potential linear relationships between features and the target variable `percent_pell_grant`. This helps in understanding which features are most strongly associated with the target and can guide feature selection.

- **Model Selection and Training**:
  - Although not performed here, the plan involves choosing classical machine learning models suitable for tabular data, such as Linear Regression, Decision Trees, Random Forests, Gradient Boosting Machines (GBM), and XGBoost. The dataset should be split into training and validation sets using `train_test_split` from `sklearn.model_selection`.

- **Model Evaluation**:
  - Evaluate models using Root Mean Square Error (RMSE) on the validation set. Implement k-fold cross-validation using `cross_val_score` from `sklearn.model_selection` to ensure model robustness and generalizability.

- **Hyperparameter Tuning**:
  - Use grid search or random search from `sklearn.model_selection` to find optimal hyperparameters for the selected models, leveraging cross-validation during the tuning process.

### Expected Outcomes and Results

- **Quantitative Outcomes**: 
  - A clean and preprocessed dataset with imputed missing values and transformed features ready for modeling.
  - A set of augmented features that capture non-linear relationships and improve model expressiveness.
  - A list of selected features that are most relevant to predicting the target variable, reducing dimensionality and enhancing model performance.

- **Qualitative Outcomes**:
  - A comprehensive understanding of the dataset's structure, including the distribution of features and their relationships with the target variable.
  - Insights into the nature of missing data and the effectiveness of different imputation strategies.
  - An understanding of the underlying characteristics of the dataset, guiding the selection of appropriate models and preprocessing techniques for the predictive task at hand.
```
- Plan Execution: **Model Agent**
```
### Step-by-Step Explanation for Finding the Top-3 High-Performance Models

#### 1. Retrieving High-Performance Models

**Model Selection**:  
For tabular regression in the education domain, we start with models known for their effectiveness on tabular data. The models include:

- **Linear Regression**: This serves as a baseline due to its simplicity and interpretability. It is essential to establish a benchmark for performance comparison.

- **Decision Trees**: Selected for their interpretability and ability to handle non-linear relationships. They provide insights into feature importance and decision paths.

- **Random Forests and Gradient Boosting Machines (GBM)**: These ensemble methods are chosen for their ability to capture complex patterns and reduce overfitting through bagging and boosting techniques.

- **XGBoost**: Known for its speed and performance, especially on imbalanced datasets, XGBoost is included for its advanced boosting capabilities.

**Initial Evaluation**:  
Each model is initially trained using the dataset, and performance is evaluated using Root Mean Square Error (RMSE) as the primary metric. RMSE provides a clear measure of prediction accuracy, which is crucial for regression tasks.

#### 2. Hyperparameter Optimization

**Hyperparameter Tuning**:  
To optimize the models, we employ hyperparameter tuning techniques:

- **Grid Search or Random Search**: These methods explore a predefined set of hyperparameters for each model. For instance:
  - **Linear Regression**: Regularization parameters such as L1 (Lasso) or L2 (Ridge) penalties are tuned. Optimal values might be L1 = 0.01 and L2 = 0.1.
  - **Decision Trees**: Parameters like `max_depth` (optimal value = 5), `min_samples_split` (optimal value = 10), and `min_samples_leaf` (optimal value = 4) are considered.
  - **Random Forests**: Tuning `n_estimators` (optimal value = 100), `max_features` (optimal value = 'sqrt'), and `max_depth` (optimal value = 10).
  - **GBM**: Parameters such as `learning_rate` (optimal value = 0.1), `n_estimators` (optimal value = 100), and `max_depth` (optimal value = 5) are optimized.
  - **XGBoost**: Key parameters include `eta` (optimal value = 0.1), `max_depth` (optimal value = 6), and `subsample` (optimal value = 0.8).

- **Cross-Validation**: Implement k-fold cross-validation (e.g., k=5) during tuning to ensure that the selected hyperparameters generalize well across different data subsets. This step is crucial for assessing model robustness.

- **Iterative Refinement**: Based on initial tuning results, hyperparameter ranges are adjusted, and models are re-evaluated to further enhance performance.

#### 3. Extracting and Understanding Model Characteristics

**Metadata Extraction and Profiling**:  
To understand the characteristics of each model, we assess:

- **Computation Complexity**: Evaluate the number of floating-point operations (FLOPs) required for training and inference. For example, XGBoost typically has higher FLOPs due to its iterative boosting process.

- **Memory Usage**: Analyze the memory footprint of each model. Random Forests may require more memory due to multiple decision trees.

- **Inference Latency**: Measure the time taken for a single prediction. Linear Regression generally has the lowest latency, while ensemble methods like Random Forests and XGBoost may have higher latency due to multiple trees.

Understanding these characteristics helps in selecting models that not only perform well but are also efficient in terms of resource usage and speed, which is essential for deployment considerations.

#### 4. Selecting the Top-k Models

**Model Evaluation and Selection**:

- **Performance Metrics**: Evaluate each model using RMSE on a validation set. Additional metrics like R-squared may be considered for comprehensive evaluation.

- **Cross-Validation**: Use k-fold cross-validation results to assess the robustness and generalizability of each model.

- **Iterative Process**: Refine models based on evaluation results, iterating through feature selection, engineering, and hyperparameter tuning as needed.

**Final Selection**:

- **Top-k Selection**: Rank the models based on RMSE and other relevant metrics. Select the top-3 models that demonstrate the best performance and generalizability.

- **Deployment Considerations**: Ensure the selected models are scalable and can be integrated into the user's system for deployment, considering real-time prediction capabilities if required.

### Names and Expected Quantitative Performance of the Top-3 Models

1. **XGBoost**:
   - RMSE: ~3.5
   - Number of Parameters: ~10,000
   - FLOPs: ~1 million
   - Model Size: ~50 MB
   - Training Time: ~30 minutes
   - Inference Speed: ~10 ms per instance

2. **Random Forest**:
   - RMSE: ~3.7
   - Number of Parameters: ~5,000
   - FLOPs: ~500,000
   - Model Size: ~40 MB
   - Training Time: ~20 minutes
   - Inference Speed: ~15 ms per instance

3. **Gradient Boosting Machine (GBM)**:
   - RMSE: ~3.8
   - Number of Parameters: ~7,500
   - FLOPs: ~750,000
   - Model Size: ~45 MB
   - Training Time: ~25 minutes
   - Inference Speed: ~12 ms per instance

These models are selected based on their balance of performance, computational efficiency, and suitability for the given task.
```

---

### World's Best Role with Instruction Steps
> You are the world's best {`job position`} of a automated machine learning project (AutoML). You have main responsibilities as follows.
> _...agent-specific steps..._

#### Tabular Classification (`mfeat-factors`)
- Global Plan
```
**End-to-End Actionable Plan for AI Agents: Handwritten Digit Recognition Using mfeat-factors Dataset**

---

### 1. **Data Preparation**

**1.1 Data Retrieval:**
- Retrieve the `mfeat-factors` dataset as provided by the user-upload source.
- Verify the integrity of the dataset and ensure there are no missing values as per the specifications.

**1.2 Data Analysis:**
- Conduct exploratory data analysis to understand the distribution and correlations of the features.
  - Generate summary statistics for each feature.
  - Plot histograms and pair plots to visualize distributions and potential correlations.
  - Identify any outliers or anomalies that may affect model performance.

**1.3 Data Preprocessing:**
- **Feature Scaling:**
  - Implement normalization or standardization to ensure features are on a similar scale.
- **Dimensionality Reduction:**
  - Apply Principal Component Analysis (PCA) to reduce dimensionality while maintaining most of the dataset's variance.
  - Determine an optimal number of components that balance computational efficiency and information retention.

### 2. **Feature Engineering**

**2.1 Feature Selection:**
- Use techniques like Recursive Feature Elimination (RFE) or feature importance from a Random Forest model to select influential features.

**2.2 Data Augmentation:**
- Although less common for tabular data, explore synthetic feature generation if beneficial patterns or interactions are observed during analysis.

### 3. **Model Development**

**3.1 Model Selection:**
- Begin with classical machine learning models suitable for tabular data:
  - **Support Vector Machines (SVM):** Focus on linear kernel initially.
  - **k-Nearest Neighbors (kNN):** Start with default parameters and adjust based on the dataset size.
  - **Decision Trees:** Utilize for initial model interpretability and feature importance insights.
- Explore ensemble methods for potential improvements:
  - **Random Forests:** For robust performance with default hyperparameters.
  - **Gradient Boosting Machines (GBM):** For improved accuracy, particularly in complex decision boundaries.

**3.2 Model Optimization:**
- Implement hyperparameter tuning using cross-validation:
  - Use grid search or randomized search strategies to optimize key parameters for each model.
  - Focus on maximizing the weighted F1 score during optimization.

### 4. **Model Evaluation and Validation**

**4.1 Data Splitting:**
- Partition the dataset into training (70%), validation (15%), and test (15%) sets to ensure generalization.
- Ensure stratified sampling to maintain class distribution across splits.

**4.2 Evaluation:**
- Use the weighted F1 score as the primary evaluation metric.
- Regularly assess model performance on the validation set during training phases.
- Final evaluation should be conducted on the test set to report final model performance.

### 5. **Considerations for Deployment**

**5.1 Computational Resources:**
- Prefer classical models due to their efficiency and lower computational demands.
- Ensure the models are interpretable, aligning with the medium expertise level of the user.

**5.2 Deployment Strategy:**
- Determine the target device for deployment and ensure model compatibility.
- Configure the deployment endpoint to facilitate easy access and integration with downstream systems.
- Prepare the inference engine to handle real-time predictions if necessary.

**5.3 Documentation and Reporting:**
- Document the entire process, detailing each step taken from data analysis to model deployment.
- Generate a comprehensive report presenting the weighted F1 score and other relevant performance metrics on the evaluation data.

---

By following this detailed plan, AI agents can construct a robust, efficient, and interpretable machine learning pipeline tailored to the task of handwritten digit recognition using the mfeat-factors dataset.
```
- Plan Execution: **Data Agent**
```
### Detailed Steps for Data Manipulation and Analysis

#### 1. Dataset Retrieval
- **Accessing the Dataset:** The `mfeat-factors` dataset is provided by the user and can be accessed from the specified paths: 
  - `agent_workspace/datasets/mfeat-factors/split_dev.csv`
  - `agent_workspace/datasets/mfeat-factors/split_dev_target.csv`
  - `agent_workspace/datasets/mfeat-factors/split_dev_wo_target.csv`
  - `agent_workspace/datasets/mfeat-factors/split_test.csv`
  - `agent_workspace/datasets/mfeat-factors/split_test_target.csv`
  - `agent_workspace/datasets/mfeat-factors/split_test_wo_target.csv`
  - `agent_workspace/datasets/mfeat-factors/split_train.csv`
- **Integrity Verification:** Before proceeding, ensure that there are no missing values in the dataset. This can be done using libraries like Pandas in Python, which provides functions such as `isnull()` and `sum()` to check for missing values. Ensuring data integrity is crucial for accurate model training and evaluation.

#### 2. Data Preprocessing
- **Feature Scaling:**
  - **Normalization or Standardization:** Use libraries like Scikit-learn to apply either Min-Max Scaling (Normalization) or Standard Scaling (Standardization). This step is essential because models like Support Vector Machines (SVMs) and k-Nearest Neighbors (kNN) are sensitive to the scale of input features. Normalization scales the features to a range of [0, 1], while standardization scales them to have a mean of 0 and a standard deviation of 1.
  
- **Dimensionality Reduction:**
  - **Principal Component Analysis (PCA):** Implement PCA using Scikit-learn to reduce the dimensionality of the dataset. Determine the number of components by retaining a high percentage of the variance (e.g., 95%). This step reduces computational complexity and helps in visualizing the data in lower dimensions while preserving essential information.

#### 3. Data Augmentation
- **Synthetic Feature Generation:** Although not typical for tabular data, consider generating synthetic features if exploratory data analysis (EDA) reveals potential beneficial interactions. This can involve creating polynomial features or interaction terms using Scikit-learn's PolynomialFeatures. This step is optional and should be guided by insights from EDA, potentially enhancing model performance by capturing non-linear relationships.

#### 4. Extracting and Understanding Dataset Characteristics
- **Exploratory Data Analysis (EDA):**
  - **Summary Statistics:** Use Pandas to generate descriptive statistics for each feature, such as mean, median, standard deviation, and quartiles. This helps in understanding the central tendencies and variability of the data.
  - **Outlier Detection:** Identify outliers using techniques like the Interquartile Range (IQR) or Z-score. Outliers can skew the results and affect model performance, so it's important to detect and decide on handling them appropriately.
  
- **Feature Selection:**
  - **Recursive Feature Elimination (RFE):** Use RFE with a model like a Random Forest to rank features based on their importance. Alternatively, use the feature importance scores directly from a trained Random Forest model. This step helps in focusing on the most relevant features, potentially improving model performance by reducing noise and overfitting.

### Expected Outcomes and Results

- **Quantitative Outcomes:**
  - A dataset that is free of missing values and outliers, ensuring data integrity.
  - Scaled features that enhance the performance of models sensitive to feature scales.
  - A reduced dataset with fewer dimensions, retaining most of the variance, leading to improved computational efficiency.
  - A set of selected features that are most relevant to the classification task, potentially improving model accuracy.

- **Qualitative Outcomes:**
  - A comprehensive understanding of the dataset's distribution, feature correlations, and potential interactions.
  - Insights into the dataset's underlying structure, aiding in better decision-making for model selection and tuning.
  - A robust dataset ready for model training, with enhanced interpretability and reduced complexity.

By following these steps, data scientists can effectively manipulate and analyze the `mfeat-factors` dataset, setting a strong foundation for building a robust model for handwritten digit recognition.
```
- Plan Execution: **Model Agent**
```
To effectively find the top-3 high-performance models or algorithms for the given task, we need to follow a structured approach that includes model selection, hyperparameter optimization, and understanding the characteristics of the candidate models. Below is a detailed explanation of each step:

### 1. Retrieving High-Performance Models

**Model Selection Strategy:**

- **Support Vector Machines (SVM):** SVMs are effective for high-dimensional spaces and are robust to overfitting, especially in cases where the number of dimensions exceeds the number of samples. They are suitable for binary and multi-class classification tasks.
  
- **k-Nearest Neighbors (kNN):** kNN is a simple, instance-based learning algorithm that is effective for small to medium-sized datasets. It is non-parametric and can capture complex decision boundaries.

- **Decision Trees:** Decision Trees are intuitive and easy to interpret. They can handle both numerical and categorical data and are robust to outliers.

- **Random Forests:** An ensemble of Decision Trees that improves accuracy by reducing overfitting. Random Forests are known for their high accuracy and robustness.

- **Gradient Boosting Machines (GBM):** GBM is an ensemble technique that builds models sequentially. It is powerful for tabular data and can capture complex patterns.

These models are selected based on their ability to handle the characteristics of tabular data effectively.

### 2. Hyperparameter Optimization

**Hyperparameter Tuning Approach:**

- **Support Vector Machines (SVM):**
  - **C (Regularization parameter):** Optimize between 0.1 and 10. Optimal value: 1.0
  - **Kernel:** Choose from 'linear', 'poly', 'rbf'. Optimal choice: 'rbf'
  - **Gamma (Kernel coefficient):** Optimize between 0.001 and 1. Optimal value: 0.01

- **k-Nearest Neighbors (kNN):**
  - **Number of Neighbors (k):** Optimize between 1 and 20. Optimal value: 5
  - **Weights:** Choose between 'uniform' and 'distance'. Optimal choice: 'distance'
  - **Algorithm:** Choose from 'auto', 'ball_tree', 'kd_tree', 'brute'. Optimal choice: 'auto'

- **Random Forests:**
  - **Number of Trees (n_estimators):** Optimize between 50 and 200. Optimal value: 100
  - **Max Depth:** Optimize between 5 and 20. Optimal value: 10
  - **Min Samples Split:** Optimize between 2 and 10. Optimal value: 2

- **Gradient Boosting Machines (GBM):**
  - **Learning Rate:** Optimize between 0.01 and 0.3. Optimal value: 0.1
  - **Number of Trees (n_estimators):** Optimize between 50 and 200. Optimal value: 100
  - **Max Depth:** Optimize between 3 and 10. Optimal value: 5

Cross-validation is used to ensure that the tuning process generalizes well to unseen data, focusing on maximizing the weighted F1 score.

### 3. Extracting and Understanding Model Characteristics

**Model Profiling:**

- **Support Vector Machines (SVM):**
  - **Computation Complexity:** High due to quadratic programming.
  - **Memory Usage:** Moderate, depends on the number of support vectors.
  - **Inference Latency:** Fast once trained.

- **k-Nearest Neighbors (kNN):**
  - **Computation Complexity:** High during inference as it requires distance computation for all training samples.
  - **Memory Usage:** High, as it stores all training data.
  - **Inference Latency:** Slow for large datasets.

- **Random Forests:**
  - **Computation Complexity:** Moderate due to parallel tree construction.
  - **Memory Usage:** Moderate, depends on the number of trees and depth.
  - **Inference Latency:** Fast due to parallel tree evaluation.

- **Gradient Boosting Machines (GBM):**
  - **Computation Complexity:** High due to sequential tree construction.
  - **Memory Usage:** Moderate, similar to Random Forests.
  - **Inference Latency:** Moderate, as trees are evaluated sequentially.

Understanding these characteristics helps in selecting models that not only perform well but are also efficient in terms of computation and memory.

### 4. Selecting the Top-3 Models

**Model Evaluation and Selection:**

- Split the dataset into training, validation, and test sets using stratified sampling.
- Evaluate models using the weighted F1 score on the validation set.
- Rank models based on their performance and select the top-3 models.

### Expected Performance of Top-3 Models

1. **Random Forests:**
   - **Weighted F1 Score:** ~0.92
   - **Number of Parameters:** ~10,000
   - **FLOPs:** ~1 million
   - **Model Size:** ~50 MB
   - **Training Time:** ~5 minutes
   - **Inference Speed:** ~10 ms per sample

2. **Gradient Boosting Machines (GBM):**
   - **Weighted F1 Score:** ~0.90
   - **Number of Parameters:** ~8,000
   - **FLOPs:** ~1.5 million
   - **Model Size:** ~40 MB
   - **Training Time:** ~10 minutes
   - **Inference Speed:** ~20 ms per sample

3. **Support Vector Machines (SVM):**
   - **Weighted F1 Score:** ~0.88
   - **Number of Parameters:** Depends on support vectors
   - **FLOPs:** ~500,000
   - **Model Size:** ~30 MB
   - **Training Time:** ~15 minutes
   - **Inference Speed:** ~5 ms per sample

By following these steps, we ensure that the selected models are not only high-performing but also efficient and suitable for deployment.
```

#### Tabular Regression (`colleges`)
- Global Plan
```
### End-to-End Actionable Plan for Building a Classical Machine Learning Model on Tabular Data

#### 1. Data Retrieval and Initial Setup
- **Objective**: Initiate the process by preparing the data environment for analysis.
- **Actions**:
  - Load the `colleges` dataset provided by the user.
  - Ensure the dataset is in a structured format suitable for analysis, such as a DataFrame.

#### 2. Comprehensive Data Analysis
- **Objective**: Understand the dataset's characteristics and the distribution of the target variable, `percent_pell_grant`.
- **Actions**:
  - Conduct Exploratory Data Analysis (EDA) to uncover feature distributions and potential relationships with the target.
  - Generate visualizations such as histograms, box plots, and correlation matrices to identify patterns and correlations.
  - Document observations about feature distributions and any initial insights regarding the target variable.

#### 3. Handling Missing Values
- **Objective**: Address missing data to ensure model robustness.
- **Actions**:
  - Identify features with missing values and quantify the extent of missingness.
  - Implement imputation strategies (mean, median, or KNN imputation) to fill missing values.
  - Consider creating binary indicators for features with missing values or removing features with excessive missingness.
  - Validate imputed values through visualization or statistical comparison with non-missing data.

#### 4. Feature Engineering
- **Objective**: Enhance the dataset with additional informative features and prepare it for modeling.
- **Actions**:
  - Scale numerical features using `MinMaxScaler` or `StandardScaler`.
  - Apply one-hot encoding to categorical variables.
  - Investigate polynomial features and interactions, particularly between geographical and demographic variables.
  - Evaluate potential new features for inclusion based on their correlation with the target variable.

#### 5. Feature Selection
- **Objective**: Reduce dimensionality and retain the most impactful features for modeling.
- **Actions**:
  - Use Recursive Feature Elimination (RFE) to assess feature importance.
  - Experiment with tree-based models for feature selection.
  - Apply dimensionality reduction techniques such as PCA if multicollinearity is detected among features.
  - Finalize a subset of features for model training.

#### 6. Model Selection and Training
- **Objective**: Train a classical machine learning model tailored to the task of tabular regression.
- **Actions**:
  - Evaluate classical algorithms including Linear Regression, Decision Trees, Random Forests, GBM, and XGBoost.
  - Split the dataset into training and validation sets.
  - Conduct hyperparameter tuning using grid search or random search to optimize model parameters.
  - Train selected models on the training data.

#### 7. Model Evaluation
- **Objective**: Assess model performance and ensure generalization to unseen data.
- **Actions**:
  - Implement k-fold cross-validation to validate model stability and performance.
  - Compute the Root Mean Square Error (RMSE) on the validation set as the primary performance metric.
  - Compare the performance of different models based on RMSE and select the best-performing model.

#### 8. Documentation and Reporting
- **Objective**: Compile a comprehensive report detailing the entire modeling process.
- **Actions**:
  - Document each step, including data preprocessing, feature engineering, model selection, and evaluation results.
  - Provide insights from EDA, justifications for preprocessing choices, feature engineering rationale, and RMSE results.
  - Highlight areas for potential improvement or future exploration.

By following this detailed plan, AI agents can effectively build, evaluate, and document a classical machine learning model for predicting `percent_pell_grant` using the `colleges` dataset, aligned with the user's medium expertise and specific requirements.
```
- Plan Execution: **Data Agent**
```
### Detailed Steps for Data Manipulation and Analysis

#### 1. Dataset Retrieval
- **Objective**: Obtain the dataset for analysis.
- **Action**: The dataset named `colleges` is provided by the user and can be retrieved from the following paths:
  - `agent_workspace/datasets/colleges/split_dev.csv`
  - `agent_workspace/datasets/colleges/split_dev_target.csv`
  - `agent_workspace/datasets/colleges/split_dev_wo_target.csv`
  - `agent_workspace/datasets/colleges/split_test.csv`
  - `agent_workspace/datasets/colleges/split_test_target.csv`
  - `agent_workspace/datasets/colleges/split_test_wo_target.csv`
  - `agent_workspace/datasets/colleges/split_train.csv`
  
  Load these CSV files into a DataFrame using a library such as Pandas. This ensures that the data is structured for easy manipulation and analysis. The main dataset for training is `split_train.csv`, while the others can be used for validation and testing.

#### 2. Data Preprocessing
- **Objective**: Prepare the dataset for model training by addressing data quality issues.

- **Missing Values**:
  - Identify features with missing values using Pandas' `isnull()` function combined with `sum()` to quantify the extent of missingness.
  - For imputation, consider using:
    - **Mean or Median Imputation**: Suitable for numerical features with a small percentage of missing values.
    - **KNN Imputation**: Useful for capturing relationships between features when missingness is more complex.
  - Optionally, create binary indicators for features with missing values to capture the information that a value was missing.
  - Remove features with excessive missingness (e.g., more than 50% missing values) as they might not provide reliable information.

- **Scaling**:
  - Normalize numerical features using `MinMaxScaler` or `StandardScaler` from the Scikit-learn library. This step is crucial to ensure that all features contribute equally to the model's performance, especially when using algorithms sensitive to feature scales.

- **Encoding**:
  - Apply one-hot encoding to categorical variables using Pandas' `get_dummies()` function. This converts categorical data into a numerical format, which is essential for most machine learning algorithms that require numerical input.

#### 3. Data Augmentation
- **Objective**: Enhance the dataset with additional features to improve model performance.

- **Actions**:
  - Investigate potential polynomial features and interactions, especially between geographical and demographic variables. This can be done using Scikit-learn's `PolynomialFeatures` to capture non-linear relationships that might not be apparent in the original features.
  - Evaluate new features by calculating their correlation with the target variable `percent_pell_grant`. Features with high correlation or those that provide new insights into the target variable should be considered for inclusion.

#### 4. Extracting and Understanding Dataset Characteristics
- **Objective**: Gain insights into the dataset's structure and relationships.

- **Actions**:
  - Conduct Exploratory Data Analysis (EDA) by generating summary statistics using Pandas' `describe()` function and correlation matrices using `corr()`. This helps in understanding feature distributions and their relationships with the target variable.
  - Document observations about feature distributions and initial insights regarding the target variable, focusing on patterns and correlations. This qualitative analysis can guide further feature engineering and model selection.
  - Use Recursive Feature Elimination (RFE) and tree-based models like Random Forests or Gradient Boosting to assess feature importance. These methods help in identifying the most impactful features and reducing dimensionality.
  - If multicollinearity is detected (e.g., high correlation between features), apply dimensionality reduction techniques such as Principal Component Analysis (PCA). PCA helps in retaining the most significant features while reducing redundancy, which is crucial for improving model performance and interpretability.

### Expected Outcomes and Results

- **Quantitative Outcomes**:
  - A clean and preprocessed dataset with imputed missing values, scaled numerical features, and encoded categorical variables.
  - An augmented dataset with additional polynomial and interaction features, potentially improving model performance.
  - A reduced feature set with the most impactful features identified through RFE and feature importance analysis.

- **Qualitative Outcomes**:
  - Enhanced understanding of the dataset's structure, including feature distributions and relationships with the target variable.
  - Insights into potential non-linear relationships captured through data augmentation.
  - A well-documented process of data preprocessing and augmentation, providing a solid foundation for subsequent modeling efforts.

These steps ensure that the dataset is well-prepared for building a classical machine learning model, aligning with best practices and the user's requirements.
```
- Plan Execution: **Model Agent**
```
To effectively find the top-3 high-performance models for the task of predicting `percent_pell_grant` using the `colleges` dataset, we will follow a structured approach focusing on model selection, hyperparameter optimization, and understanding the characteristics of the candidate models. Here is a detailed explanation of each step:

### 1. Retrieving High-Performance Models

**Model Selection:**
- We will focus on classical machine learning models suitable for tabular regression tasks. The models considered are Linear Regression, Decision Trees, Random Forests, Gradient Boosting Machines (GBM), and XGBoost. These models are chosen due to their ability to handle tabular data effectively and their widespread use in regression tasks.
- **Initial Evaluation:** Each model will be initially evaluated using default hyperparameters on the preprocessed dataset. This step is crucial to establish a baseline performance and identify models that inherently perform well in terms of Root Mean Square Error (RMSE).

### 2. Hyperparameter Optimization

**Hyperparameter Tuning:**
- **Grid Search or Random Search:** We will employ these techniques to systematically explore the hyperparameter space for each model. This involves defining a search space for each model's hyperparameters and using cross-validation to evaluate different combinations.
- **Specific Hyperparameters:**
  - **Linear Regression:** No hyperparameters to tune as it is a parameter-free model.
  - **Decision Trees:** Optimize `max_depth` (optimal value: 10), `min_samples_split` (optimal value: 5), and `min_samples_leaf` (optimal value: 2).
  - **Random Forests:** Optimize `n_estimators` (optimal value: 100), `max_depth` (optimal value: 15), and `min_samples_split` (optimal value: 4).
  - **Gradient Boosting Machines (GBM):** Optimize `n_estimators` (optimal value: 150), `learning_rate` (optimal value: 0.1), and `max_depth` (optimal value: 3).
  - **XGBoost:** Optimize `n_estimators` (optimal value: 200), `learning_rate` (optimal value: 0.05), `max_depth` (optimal value: 4), and `subsample` (optimal value: 0.8).
- The optimal values are determined based on empirical knowledge and typical settings for similar datasets.

### 3. Extracting and Understanding Model Characteristics

**Model Profiling:**
- **Computation Complexity:** Evaluate the number of floating-point operations (FLOPs) required for training and inference. For instance, XGBoost typically has higher FLOPs due to its ensemble nature.
- **Memory Usage:** Assess the memory footprint of each model. Random Forests and XGBoost tend to have larger memory usage due to storing multiple trees.
- **Inference Latency:** Measure the time taken for a single prediction. Linear Regression is usually the fastest, while ensemble methods like Random Forests and XGBoost might be slower.
- Understanding these characteristics helps in selecting models that not only perform well but also meet operational constraints such as speed and resource usage.

### 4. Selecting the Top-3 Models

**Model Evaluation and Selection:**
- Implement k-fold cross-validation to ensure the robustness and generalization capability of each model.
- Compute the RMSE for each model on the validation set and rank them based on performance.
- Select the top-3 models with the lowest RMSE values for further consideration or deployment.

### Expected Performance and Complexity Metrics

1. **XGBoost:**
   - Expected RMSE: 0.85
   - Number of Parameters: Approximately 200,000
   - FLOPs: 1.5 million
   - Model Size: 150 MB
   - Training Time: 30 minutes
   - Inference Speed: 10 ms per sample

2. **Random Forests:**
   - Expected RMSE: 0.90
   - Number of Parameters: Approximately 100,000
   - FLOPs: 1 million
   - Model Size: 100 MB
   - Training Time: 20 minutes
   - Inference Speed: 15 ms per sample

3. **Gradient Boosting Machines (GBM):**
   - Expected RMSE: 0.88
   - Number of Parameters: Approximately 150,000
   - FLOPs: 1.2 million
   - Model Size: 120 MB
   - Training Time: 25 minutes
   - Inference Speed: 12 ms per sample

These steps and metrics provide a comprehensive approach to identifying and optimizing the top-3 candidate models for the given task, ensuring they meet both performance and operational requirements.
```
