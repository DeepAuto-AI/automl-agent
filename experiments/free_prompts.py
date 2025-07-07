FREE_PROMPTS = {
    "computer vision": {
        "image_classification": [
            "I need a very accurate model to classify images in the Butterfly Image Classification dataset into their respective categories. The dataset has been uploaded with its label information in the labels.csv file.",
            "Please provide a classification model that categorizes images into one of four clothing categories. The image path, along with its label information, can be found in the files train_labels.csv and test_labels.csv.",
        ],
    },
    "natural language processing": {
        "text_classification": [
            """We need a state-of-the-art model for text classification based on the Ecommerce Text dataset. 
            The model should be capable of accurately classifying text into four categories: Electronics, Household, Books, and Clothing & Accessories. 
            We have uploaded the entire dataset without splitting it here.""",
            """You are solving this machine learning tasks of classification: 
The dataset presented here (the Textual Entailment) comprises a series of labeled text pairs. Given two texts (text1 and text2), your task is to predict the relationship of the text pair of neutral (0), contradiction (1) or entailment (2). The evaluation metric is accuracy.
Build a language model to get a good performance. """,
        ],
    },
    "tabular": {
        "tabular_classification": [
            "Build a model to classify banana quality as Good or Bad based on their numerical information about bananas of different quality (size, weight, sweetness, softness, harvest time, ripeness, and acidity). We have uploaded the entire dataset for you here in the banana_quality.csv file.",
            """You are solving this data science tasks of binary classification: 
The dataset presented here (the Software Defects Dataset) comprises a lot of numerical features. Please split the dataset into three parts of train, valid and test. Your task is to predict the defects item, which is a binary label with 0 and 1. The evaluation metric is the F1 score.
Please train a binary classification model to get a good performance on this task.""",
        ],
        "tabular_regression": [
            """You are solving this data science tasks of regression: 
The dataset presented here (the Crab Age Dataset) comprises a lot of both categorical and numerical features. Pleae split the dataset into three parts of train, valid and test. Your task is to predict the age item. The evaluation metric is the RMSLE (root mean squared log error).
Now train a regression model to get a good performance on this task.""",
            "I need a regression model to predict crop prices based on features like soil composition, environmental factors, historical yield data, and crop management practices from the dataset I uploaded here.",
        ],
        "tabular_clustering": [
            """You are solving this data science tasks of unsupervised clustering: 
The dataset presented here (the Smoker Status Dataset) comprises a lot of numerical features. Please use the features in the test.csv file. Your task is to create the clustered items, which is a binary label with 0 and 1 (two clusters). The evaluation metric is the Rand index or Rand score, can be tested against 'smoking' labels.
Now train an unsupervised clustering model to get a good performance on this task.""",
            "I want an unsupervised clustering model to group student performances into eight groups. The dataset named 'Higher Education Students Performance Evaluation' (id=856) can be downloaded via ucimlrepo library. The clustering quality can be check against target variable OUTPUT Grade.",
        ],
    },
    "graph": {
        "node_classification": [
            "I want you to create a model for node classification on the Cora dataset to predict the category of each paper. You need to directly find the Cora dataset from a relevant library.",
            "I want you to create a model for node classification on the Citeseer dataset to predict the category of each paper. You need to directly find the Citeseer dataset from a relevant library.",
        ]
    },
    "time series": {
        "ts_forecasting": [
            "Build a model to perform time-series forecasting using the Weather dataset uploaded here, evaluating its accuracy with the RMSLE metric. Note that the input is a sequence of past observations with fixed size (INPUT_SEQ_LEN=96, INPUT_DIM=21). The model should predict the next future sequence with a fixed size (PRED_SEQ_LEN=96, PRED_DIM=21).",
            """You are solving this machine learning tasks of time series forecasting: 
The dataset presented here (the Electricity dataset) comprises real-world time series data. Please split the dataset into three parts of train, valid and test. The input is a sequence of past observation with fixed size (INPUT_SEQ_LEN=96, INPUT_DIM=321). Your task is to predict the next future sequence with fixed size (PRED_SEQ_LEN=96, PRED_DIM=321). The evaluation metric is root mean squared log error (RMSLE).
Now train a time series forecasting model to get a good performance on the given fixed sequences.""",
        ],
    },
}

AGENTK_PROMPTS = {
    "dogs-vs-cats-redux-kernels-edition": """This is a dogs-vs-cats-redux-kernels-edition dataset.
Your goal is to distinguish images of dogs from cats. This dataset is from Dogs vs. Cats Redux: Kernels Edition Kaggle Competition.
Make sure to additionally create a Submission File.
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:
```csv
id,label
1,0.5
2,0.5
3,0.5
...
```
""",
    "home-data-for-ml-course": """This is a home-data-for-ml-course dataset.
Your goal is to predict the target column `SalePrice`. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target.
Report rmse on the eval data. Do not plot or make any visualizations.
This dataset is from Housing Prices Competition for Kaggle Learn Users Competition.
Make sure to additionally create a Submission File. The file should contain a header and have the following format:
```csv
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221
...
```
""",
    "nlp1000-ml-challenge": """This is a nlp1000-ml-challenge dataset, A sentiment classification challenge using Filipino-English Twitter text data.
Your goal is to predict the target column `IsPositive`.

This dataset is from NLP1000 ML Challenge Kaggle Competition.
Make sure to additionally create a Submission File. The file should contain a header and have the following format:
For every author in the dataset, submission files should contain two columns: `DocumentId` and `IsPositive`. 
DocumentId is a unique identifier for each individual tweet. IsPositive is a Boolean value (i.e. True or False). True signals a document is positive in its sentiment, while False signals that a document is negative in its sentiment. When exporting your predictions, please ensure the output file is labeled as "submission.csv". You're also encouraged to use the following code for generating the CSV file if you have the predictions via a DataFrame: 
``` import pandas as pd import csv df.to_csv('submission.csv', quoting=csv.QUOTE_ALL) ``` The file should contain a header and have the following format: ``` "DocumentId", "IsPositive" "1","True" ```
""",
    "nlpsci": """This is a nlpsci dataset.
Your goal is to predict science categories in the target column `label` from title and abstract. From paper's title and abstract, predict science categories:
0: Computer Science
1: Physics
2: Mathematics
3: Statistics
4: Quantitative Biology
5: Quantitative Finance

This dataset is from Science nlp classification Kaggle Competition.
Make sure to additionally create a Submission File. The file should contain a header and have the following format:
```csv
ID,label
1,0
2,0
3,0
...
```
""",
    "sign-language-image-classification": """This is a sign-language-image-classification dataset.
The sign language database of hand gestures represents a multi-class problem with 20 classes of letters and numbers. 
Your goal is to predict the target column `label`. This dataset is from Sign Language Image Classification Kaggle Competition.
Make sure to additionally create a Submission File. The file should contain a header and have the following format:
```csv
id,predict
5f3540c43f86dab2c9a7c87834311ee3.jpg,0
7f14db51f1994bce42972c7fe111f5cd.jpg,0
092908960a3d4dccbe06a60b71dcdd98.jpg,0
...
```
""",
    "world-championship-2023-embryo-classification": """This is a world-championship-2023-embryo-classification dataset.
Your goal is to predict the target column ``. This year we introduce the Embryo classification challenge using on microscopic images by Hung Vuong Hospital. This is a remarkable opportunity to delve into the world of reproductive science and contribute to improving the accuracy of embryo quality assessment using cutting-edge deep learning techniques.

Embryo quality assessment is a critical aspect of assisted reproductive technology, aiding fertility specialists in selecting the most viable embryos for implantation. 
However, this process is both complex and subjective. In this competition, you're challenged to create a robust deep learning model capable of classifying embryos as 'good' or 'not good' based on their images at day-3 and day-5 of development.

The competition's objective is to harness the power of deep learning to develop an accurate classification model that can discern between 'good' and 'not good' embryos at both day-3 and day-5 stages. By achieving this, you'll help fertility specialists make more informed decisions, enhance the efficiency of assisted reproductive procedures, and ultimately contribute to the joy of prospective parents.

This dataset is from Embryo classification based on microscopic images Kaggle Competition.
Make sure to additionally create a Submission File.  For each embryo in the test set, you are required to predict a binary classification label representing the quality of the embryo. 
Use the provided sample_submission.csv as the template for your submission. The submission file should adhere to the following format:
```csv
ID,Class
1,1
2,0
3,1
4,0
5,1
```
""",
    "playground-series-s3e14": """Welcome to the 2023 Kaggle Playground Series! Thank you to everyone who participated in and contributed to Season 3 Playground Series so far!

Your Goal: For this Episode of the Series, your task is to use regression to predict the yield of wild blueberries. Good luck!
This dataset is from Regression with a Wild Blueberry Yield Dataset Kaggle Competition.
Make sure to additionally create a Submission File.  
For each id in the test set, you must predict the target yield. The file should contain a header and have the following format:
```csv
id,yield
15289,6025.194
15290,1256.223
15291,357.44
etc.
```
""",
    "restaurant-revenue-prediction": """With over 1,200 quick service restaurants across the globe, TFI is the company behind some of the world's most well-known brands: Burger King, Sbarro, Popeyes, Usta Donerci, and Arby’s. They employ over 20,000 people in Europe and Asia and make significant daily investments in developing new restaurant sites.

Right now, deciding when and where to open new restaurants is largely a subjective process based on the personal judgement and experience of development teams. This subjective data is difficult to accurately extrapolate across geographies and cultures. 

New restaurant sites take large investments of time and capital to get up and running. When the wrong location for a restaurant brand is chosen, the site closes within 18 months and operating losses are incurred. 

Finding a mathematical model to increase the effectiveness of investments in new restaurant sites would allow TFI to invest more in other important business areas, like sustainability, innovation, and training for new employees. Using demographic, real estate, and commercial data, this competition challenges you to predict the annual restaurant sales of 100,000 regional locations.

For every restaurant in the dataset, submission files should contain two columns: Id and Prediction. 

This dataset is from Regression with a Restaurant Revenue Prediction Kaggle Competition.
Make sure to additionally create a Submission File.  
The file should contain a header and have the following format:

```csv
Id,Prediction
0,1.0
1,1.0
2,1.0
etc.
```
""",
}

SELA_PROMPTS = {
    "house-prices": "This is a 05_house-prices-advanced-regression-techniques dataset.\
      \ Your goal is to predict the target column `SalePrice`.\nPerform data analysis,\
      \ data preprocessing, feature engineering, and modeling to predict the target.\
      \ \nReport rmse on the eval data. Do not plot or make any visualizations.\n",
    "click-prediction-small": "This is a Click_prediction_small dataset. Your goal is to predict\
      \ the target column `click`.\nPerform data analysis, data preprocessing, feature\
      \ engineering, and modeling to predict the target. \nReport f1 on the eval data.\
      \ Do not plot or make any visualizations.\n",
    "colleges": "This is a colleges dataset. Your goal is to predict the target\
      \ column `percent_pell_grant`.\nPerform data analysis, data preprocessing, feature\
      \ engineering, and modeling to predict the target. \nReport rmse on the eval\
      \ data. Do not plot or make any visualizations.\n",
    "mfeat-factors": "This is a mfeat-factors dataset. Your goal is to predict the\
      \ target column `class`.\nPerform data analysis, data preprocessing, feature\
      \ engineering, and modeling to predict the target. \nReport f1 weighted on the\
      \ eval data. Do not plot or make any visualizations.\n",
    "smoker-status": "This is a smoker-status dataset. Your goal is to predict the\
      \ target column `smoking`.\nPerform data analysis, data preprocessing, feature\
      \ engineering, and modeling to predict the target. \nReport f1 on the eval data.\
      \ Do not plot or make any visualizations.\n",
    "wine-quality-white": "This is a wine-quality-white dataset. Your goal is to predict\
      \ the target column `Class`.\nPerform data analysis, data preprocessing, feature\
      \ engineering, and modeling to predict the target. \nReport f1 weighted on the\
      \ eval data. Do not plot or make any visualizations.\n",
    "concrete-strength": "This is a concrete-strength dataset. Your goal is to predict\
      \ the target column `Strength`.\nPerform data analysis, data preprocessing,\
      \ feature engineering, and modeling to predict the target. \nReport rmse on\
      \ the eval data. Do not plot or make any visualizations.\n",
}
