CONSTRAINT_PROMPTS = {
    "computer vision": {
        "image_classification": [
            "I need a highly accurate machine learning model developed to classify images within the Butterfly Image Classification dataset into their correct species categories. The dataset has been uploaded with its label information in the labels.csv file. Please use a convolutional neural network (CNN) architecture for this task, leveraging transfer learning from a pre-trained ResNet-50 model to improve accuracy. Optimize the model using cross-validation on the training split to fine-tune hyperparameters, and aim for an accuracy of at least 0.95 on the test split. Provide the final trained model, a detailed report of the training process, hyperparameter settings, accuracy metrics, and a confusion matrix to evaluate performance across different categories.",
            "Please provide a classification model that categorizes images into one of four clothing categories. The image path, along with its label information, can be found in the files train_labels.csv and test_labels.csv. The model should achieve at least 85% accuracy on the test set and be implemented using PyTorch. Additionally, please include data augmentation techniques and a confusion matrix in the evaluation.",
        ],
    },
    "natural language processing": {
        "text_classification": [
            """We require the development of an advanced neural network model for text classification tailored to the Ecommerce Text dataset, with the objective of achieving at least 0.95 classification accuracy.
            The model should be specifically trained to distinguish text into four defined categories: Electronics, Household, Books, and Clothing & Accessories.
            To facilitate this, we have uploaded the complete dataset in its entirety, without any prior division into training, validation, or test sets.""",
            """You are solving this machine learning task of classification: 
The dataset presented here (the Textual Entailment) comprises a series of labeled text pairs. Given two texts, your task is to predict the relationship of the text pair as neutral (0), contradiction (1), or entailment (2). The evaluation metric is accuracy.
Build a language model to get good performance, ensuring the model size does not exceed 200 million parameters and the inference time is less than 200 milliseconds per prediction.""",
        ],
    },
    "tabular": {
        "tabular_classification": [
            "Build a machine learning model, potentially XGBoost or LightGBM, to classify banana quality as Good or Bad based on their numerical information about bananas of different quality (size, weight, sweetness, softness, harvest time, ripeness, and acidity). We have uploaded the entire dataset for you here in the banana_quality.csv file. The model must achieve at least 0.98 accuracy.",
            """You are solving this data science task of binary classification: 
The dataset presented here (the Software Defects Dataset) comprises a lot of numerical features. Please split the dataset into three parts of train, valid, and test. Your task is to predict the defects item, which is a binary label with 0 and 1. The evaluation metric is the F1 score. 
Please train a binary classification model to get a good performance on this task, ensuring that the model training time does not exceed 30 minutes and the prediction time for each instance is under 5 milliseconds.""",
        ],
        "tabular_regression": [
            """You are solving this data science task of regression: 
The dataset presented here (the Crab Age Dataset) comprises a lot of both categorical and numerical features. Please split the dataset into three parts of train, valid, and test. Your task is to predict the age item. The evaluation metric is the RMSLE (root mean squared log error).
Now train a regression model to get a good performance on this task, ensuring that the model's training time does not exceed 30 minutes and that it can make predictions on the test set within 5 seconds.""",
            "I need an accurate regression model to predict crop prices based on features like soil composition, environmental factors, historical yield data, and crop management practices from the dataset I uploaded here. You should optimize the model to achieve RMSLE less than 1.0",
        ],
        "tabular_clustering": [
            """You are solving this data science task of unsupervised clustering:
The dataset presented here (the Smoker Status Dataset) comprises a lot of numerical features. Please use the features in test.csv. Your task is to create the clustered items, which is a binary label with 0 and 1 (two clusters). The evaluation metric is the Rand index or Rand score, which can be tested against 'smoking' labels. 
Now train an unsupervised clustering model to get a good performance on this task, ensuring that the Rand index is at least 0.75 and the model training time does not exceed 10 minutes.""",
            "I want an unsupervised clustering model to group student performances into eight groups. The dataset named 'Higher Education Students Performance Evaluation' (id=856) can be downloaded via ucimlrepo library. The clustering quality can be checked against the target variable OUTPUT Grade. The model should achieve a Rand Score of at least 0.8 and complete clustering within 10 minutes.",
        ],
    },
    "graph": {
        "node_classification": [
            "I want you to develop a node classification model using the Graph Convolutional Network (GCN) algorithm to predict the category of each paper in the Cora dataset. Start by importing the Cora dataset using the `Planetoid` dataset from the `torch_geometric.datasets` module in PyTorch Geometric. Ensure you preprocess the data to include node features and labels correctly. Train the model using a suitable optimizer and loss function. Then, evaluate its accuracy on the test set. The accuracy on the test set should be over 0.90.",
            "I want you to develop a node classification model using the Graph Convolutional Network (GCN) algorithm to predict the category of each paper in the Citeseer dataset. Start by importing the Citeseer dataset using the `Planetoid` dataset from the `torch_geometric.datasets` module in PyTorch Geometric. Ensure you preprocess the data to include node features and labels correctly. Train the model using a suitable optimizer and loss function. Then, evaluate its accuracy on the test set. The accuracy on the test set should be over 0.80.",
        ]
    },
    "time series": {
        "ts_forecasting": [
            "Build a state-of-the-art time-series forecasting model for the Weather dataset uploaded here, evaluating its accuracy with the RMSLE metric. Note that the input is a sequence of past observations with fixed size (INPUT_SEQ_LEN=96, INPUT_DIM=21). The model should predict the next future sequence with a fixed size (PRED_SEQ_LEN=96, PRED_DIM=21). We target the model to have RMSLE lower than 0.05.",
            """You are solving this machine learning task of time series forecasting:
The dataset presented here (the Electricity dataset) comprises real-world time series data. Please split the dataset into three parts of train, valid, and test. The input is a sequence of past observation with fixed size (INPUT_SEQ_LEN=96, INPUT_DIM=321). Your task is to predict the next future sequence with fixed size (PRED_SEQ_LEN=96, PRED_DIM=321). The evaluation metric is root mean squared log error (RMSLE).
Now train a time series forecasting model to get a good performance on the given fixed sequences. Ensure the model achieves an RMSLE of less than 0.1 and that the training time does not exceed 1 hour on a standard GPU.""",
        ],
    },
}
