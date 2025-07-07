# Examples of Plan Knowledge Used for Planning

## Tabular Classification (`smoker-status`)
```
Here is a list of suggestions to address the user's requirements in building a machine learning model for predicting smoking status:

1. **Data Analysis and Preprocessing**:
   - Conduct thorough Exploratory Data Analysis (EDA) to understand feature distributions and identify any potential outliers.
   - Normalize or standardize the dataset to ensure all features contribute equally to the model's performance.
   - Since there are no missing values, focus on handling outliers through removal or replacement.

2. **Feature Engineering**:
   - Utilize techniques such as Recursive Feature Elimination (RFE) or leverage domain knowledge to select the most relevant features.
   - Consider creating polynomial features, interaction terms, or applying dimensionality reduction techniques like PCA to enhance model performance.
   - Use Partial Dependence Plots (PDP) to visualize relationships between features and the target variable, aiding in feature selection.

3. **Model Selection**:
   - Choose robust algorithms like Random Forest, Gradient Boosting Machines (GBM), XGBoost, or LightGBM, which are well-suited for large tabular datasets and binary classification tasks.
   - These models are interpretable and effective in handling feature interactions without extensive preprocessing.

4. **Model Evaluation**:
   - Emphasize the F1 score as the primary evaluation metric to balance precision and recall, crucial for minimizing false positives and false negatives in healthcare applications.
   - Implement cross-validation techniques to avoid overfitting and ensure the model's generalizability.

5. **Hyperparameter Tuning**:
   - Optimize model performance through hyperparameter tuning using methods like grid search, random search, or Bayesian optimization.

6. **Model Interpretability**:
   - Post-training, use model-agnostic interpretation tools like SHAP (SHapley Additive exPlanations) or LIME to understand feature importance and ensure the model's decisions align with domain knowledge.

7. **Model Validation**:
   - Evaluate the model on a separate test set to ensure its robustness and generalizability in predicting smoking status.

By following these suggestions, the user can build a robust and interpretable machine learning model tailored to predict smoking status effectively.
```
#### Corresponding Plan
```
# End-to-End Actionable Plan for AI Agents

## Objective
Develop a machine learning pipeline to predict smoking status using the provided "smoker-status" dataset, with a focus on achieving a high F1 score. The task involves tabular classification within the healthcare domain.

## 1. Data Understanding and Preparation

### 1.1 Data Retrieval
- **Task**: Retrieve the "smoker-status" dataset from the user-uploaded source.
- **Specification**: Ensure the dataset contains 143,330 instances and 23 numeric features.

### 1.2 Exploratory Data Analysis (EDA)
- **Task**: Conduct EDA to understand feature distributions, correlations, and detect potential outliers.
- **Instructions**:
  - Analyze statistical summaries (mean, median, variance) for each feature.
  - Visualize distributions using histograms and box plots.
  - Identify and document outliers using interquartile range (IQR).

### 1.3 Data Preprocessing
- **Task**: Prepare data for modeling.
- **Instructions**:
  - Normalize or standardize features to ensure equal contribution to model performance.
  - Handle outliers by either removing or replacing them with statistical measures (e.g., median).
  - Verify there are no missing values as specified.

## 2. Feature Engineering

### 2.1 Feature Selection
- **Task**: Select the most relevant features using domain knowledge and algorithmic techniques.
- **Instructions**:
  - Implement Recursive Feature Elimination (RFE) to rank features.
  - Consider feature importance scores from tree-based models for selection guidance.

### 2.2 Feature Enhancement
- **Task**: Enhance the feature set for potentially improved model performance.
- **Instructions**:
  - Generate polynomial features and interaction terms.
  - Apply PCA for dimensionality reduction if necessary, ensuring variance retention.
  - Visualize feature-target relationships using Partial Dependence Plots (PDP).

## 3. Model Development

### 3.1 Model Selection
- **Task**: Choose a suitable machine learning model for binary classification.
- **Instructions**:
  - Evaluate algorithms like Random Forest, Gradient Boosting Machines (GBM), XGBoost, and LightGBM.
  - Consider computational efficiency and interpretability for final selection.

### 3.2 Model Training
- **Task**: Train the selected model on the prepared dataset.
- **Instructions**:
  - Split data into training and validation sets.
  - Use cross-validation techniques (e.g., k-fold) to ensure model generalizability.

## 4. Model Evaluation

### 4.1 Performance Assessment
- **Task**: Evaluate model performance, focusing on the F1 score.
- **Instructions**:
  - Calculate precision, recall, and F1 score on validation data.
  - Use cross-validation results to assess model consistency.

### 4.2 Hyperparameter Tuning
- **Task**: Optimize model parameters for improved performance.
- **Instructions**:
  - Perform grid search, random search, or Bayesian optimization.
  - Focus on parameters that most affect model accuracy and F1 score.

### 4.3 Model Interpretability
- **Task**: Ensure model decisions are interpretable and align with domain knowledge.
- **Instructions**:
  - Use SHAP (SHapley Additive exPlanations) or LIME for feature importance analysis.
  - Document insights and ensure alignment with healthcare domain expectations.

## 5. Model Validation and Deployment

### 5.1 Final Validation
- **Task**: Evaluate the model on a separate test set for final assessment.
- **Instructions**:
  - Ensure robustness and generalizability in predicting smoking status.
  - Reevaluate key metrics, especially the F1 score.

### 5.2 Deployment Preparation
- **Task**: Prepare the model for deployment in a healthcare setting.
- **Instructions**:
  - Package the model and preprocessing pipeline.
  - Ensure compliance with healthcare data handling standards.
  - Develop an API endpoint for real-time inference (if applicable).

## Conclusion
Following this plan will enable AI agents to develop a robust, interpretable, and performant machine learning model that meets the user's requirements for predicting smoking status in the healthcare domain.
```


## Tabular Regression (`house-prices`)
```
Here is a list of suggestions to address the user's requirements for predicting 'SalePrice' in the real estate domain using the '05_house-prices-advanced-regression-techniques' dataset:

1. **Exploratory Data Analysis (EDA):**
   - Conduct EDA to understand data distribution, identify patterns, and detect outliers.
   - Assess feature distributions and identify missing values.

2. **Handling Missing Values:**
   - Use imputation techniques such as mean, median, mode substitution, or K-Nearest Neighbors imputation to handle missing data.
   - For categorical features, consider creating a separate category for missing values.

3. **Feature Engineering:**
   - Transform skewed distributions using log transformations to normalize data.
   - Encode categorical variables using one-hot encoding for better model interpretability.
   - Explore polynomial features or interaction terms to capture complex relationships.
   - Handle multicollinearity and select significant predictors using correlation matrices.

4. **Model Selection:**
   - Employ classical machine learning models like Ridge Regression, Linear Regression, Decision Trees, Random Forests, or Gradient Boosting Machines (GBM) such as XGBoost or LightGBM.
   - These models are well-suited for tabular data and can handle nonlinear relationships and missing values effectively.

5. **Model Evaluation:**
   - Use Root Mean Square Error (RMSE) as the performance metric to evaluate model accuracy.
   - Regularly validate model performance using cross-validation techniques to ensure generalizability and robustness across different data splits.

6. **Hyperparameter Tuning:**
   - Enhance model performance through hyperparameter tuning using techniques like grid search, random search, or Bayesian optimization.

7. **Model Interpretability and Robustness:**
   - Choose models that provide interpretability and are robust against overfitting.
   - Ensure the model's predictions closely align with actual values for accurate price prediction.

By following these suggestions, the user can build a robust and accurate model for predicting house prices in the real estate domain.
```
#### Corresponding Plan
```
Here is an actionable end-to-end plan for AI agents to build a robust model for predicting 'SalePrice' in the real estate domain using the '05_house-prices-advanced-regression-techniques' dataset:

### 1. Data Acquisition and Understanding

1. **Retrieve Dataset:**
   - Load the '05_house-prices-advanced-regression-techniques' dataset to initiate the process.

2. **Initial Data Inspection:**
   - Inspect the dataset's dimensions, feature types, and check for missing values.
   - Verify the target variable 'SalePrice' is included and assess its distribution.

### 2. Exploratory Data Analysis (EDA)

1. **Data Visualization:**
   - Generate visualizations such as histograms, box plots, and scatter plots to understand the distribution of features and target variable.
   - Create a correlation heatmap to identify relationships between features and the target variable.

2. **Outlier Detection:**
   - Identify outliers in numerical features using statistical methods like the IQR method or Z-score analysis.

### 3. Data Preprocessing

1. **Handling Missing Values:**
   - Quantify the missing values in each feature. For numerical features, consider mean or median imputation. For categorical features, use mode imputation or create a new category.

2. **Data Transformation:**
   - Apply log transformation on skewed numerical features to achieve normality.
   - Standardize or normalize numerical features to ensure consistent scale across all features.

3. **Encoding Categorical Variables:**
   - Use one-hot encoding for categorical features to convert them into numerical format suitable for machine learning models.

### 4. Feature Engineering

1. **Feature Selection:**
   - Use feature importance from models like Random Forest or Gradient Boosting to select significant predictors.
   - Handle multicollinearity by removing highly correlated features based on the correlation matrix.

2. **Feature Creation:**
   - Create polynomial features or interaction terms to capture complex relationships.
   - Consider domain-specific features, such as age of the house or renovation status, if applicable.

### 5. Model Development

1. **Model Selection:**
   - Choose classical machine learning models like Ridge Regression, Linear Regression, Decision Trees, Random Forest, or Gradient Boosting Machines (e.g., XGBoost, LightGBM) for this regression task.

2. **Model Training:**
   - Split the dataset into training and validation sets.
   - Train selected models on the training data using cross-validation to ensure model robustness.

3. **Hyperparameter Tuning:**
   - Conduct hyperparameter tuning using grid search or Bayesian optimization to find optimal model settings.

### 6. Model Evaluation

1. **Performance Metrics:**
   - Evaluate model performance using Root Mean Square Error (RMSE) as the primary metric on the validation set.
   - Compare RMSE across different models to identify the best-performing one.

2. **Model Validation:**
   - Implement k-fold cross-validation to ensure model generalizability and reduce risk of overfitting.

### 7. Model Deployment (Future Consideration)

1. **Target Device and Deployment Endpoint:**
   - Plan for deploying the model on a scalable platform that supports real-time inference.
   - Consider using cloud services like AWS SageMaker, Google Cloud AI Platform, or Azure ML for deployment.

2. **Inference Engine:**
   - Prepare the model for integration with an inference engine that supports batch and real-time prediction capabilities.

### 8. Reporting and Documentation

1. **Results Documentation:**
   - Document the entire process, from data exploration to final model selection and evaluation.
   - Include visualizations and model performance metrics in the report for clarity.

2. **Future Work Suggestions:**
   - Suggest further exploration of advanced techniques such as deep learning models, if the dataset size permits, or domain-specific enhancements for future work.

By following this plan, AI agents can build a comprehensive and effective model for predicting house prices, ensuring that all necessary steps from data preprocessing to model evaluation are thoroughly addressed.
```
