from .free_prompts import FREE_PROMPTS
from .constraint_prompts import CONSTRAINT_PROMPTS
from glob import glob

data_infos = {
    "image_classification": [
        f'Note that the uploaded dataset is available at "_experiments/datasets/butterfly_image". This is the structure of the dataset directory: {glob("_experiments/datasets/butterfly_image/*")}',
        f'Note that the uploaded dataset is available at "_experiments/datasets/shopee". This is the structure of the dataset directory: {glob("_experiments/datasets/shopee/*")}.',
    ],
    "text_classification": [
        f'Note that the uploaded dataset is available at "_experiments/datasets/ecommerce_text/ecommerce_text.csv" in the csv format. The dataset file contains two titleless columns: the first column indicates class label, the second one indicates the text to be classified.',
        f'Note that the uploaded dataset is available at "_experiments/datasets/textual_entailment/train.csv" and "_experiments/datasets/textual_entailment/test.csv" in the csv format. Each file contains three named columns: text1 , text2, and label',
    ],
    "tabular_classification": [
        f'Note that the uploaded dataset is available at "_experiments/datasets/banana_quality/banana_quality.csv" in the csv format. The dataset file contains "Size,Weight,Sweetness,Softness,HarvestTime,Ripeness,Acidity,Quality" columns.',
        f'Note that the uploaded dataset is available at "_experiments/datasets/software_defects/train.csv" and "_experiments/datasets/software_defects/test.csv" in the csv format. Each dataset file contains "loc,v(g),ev(g),iv(g),n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount,defects" columns.',
    ],
    "tabular_regression": [
        f'Note that the uploaded dataset is available at "_experiments/datasets/crab_age/train.csv" and "_experiments/datasets/crab_age/test.csv" in the csv format. Each dataset file contains "Sex,Length,Diameter,Height,Weight,Shucked Weight,Viscera Weight,Shell Weight,Age" columns.',
        f'Note that the uploaded dataset is available at "_experiments/datasets/crop_yield_prediction/crop_yield_prediction.csv" in the csv format. The dataset file contains "Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH_Value,Rainfall,Crop,Yield" columns.',
    ],
    "tabular_clustering": [
        f'Note that the uploaded dataset is available at "_experiments/datasets/smoker_status/test.csv" in the csv format. The dataset file contains numerical columns, where the last column named "smoking" is the variable allowed only for final evaluation.',
    ],
    "ts_forecasting": [
        f'Note that the uploaded dataset is available at "_experiments/datasets/weather/weather.csv" in the csv format. The dataset file contains several numerical columns, where the last column named "OT" is the target variable for forecasting.',
        f'Note that the uploaded dataset is available at "_experiments/datasets/electricity/electricity.csv" in the csv format. The dataset file contains several numerical columns, where the last column named "OT" is the target variable for forecasting.',
    ],
}

agentk_data_infos = {
    "playground-series-s3e14": """The dataset for this competition (both train and test) was generated from a deep learning model trained on the Wild blueberry Yield Prediction Dataset. (Since this is Playground 3.14, it seems like we need a Blueberry Pie joke here?) Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
Files
train.csv - the training dataset; yield is the target
test.csv - the test dataset; your objective is to predict the yield given the other features
sample_submission.csv - a sample submission file in the correct format
""",
    "restaurant-revenue-prediction": """Dataset Description
TFI has provided a dataset with 137 restaurants in the training set, and a test set of 100000 restaurants. The data columns include the open date, location, city type, and three categories of obfuscated data: Demographic data, Real estate data, and Commercial data. The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. 

File descriptions
train.csv - the training set. Use this dataset for training your model. 
test.csv - the test set. To deter manual "guess" predictions, Kaggle has supplemented the test set with additional "ignored" data. These are not counted in the scoring.
sampleSubmission.csv - a sample submission file in the correct format
Data fields
Id : Restaurant id. 
Open Date : opening date for a restaurant
City : City that the restaurant is in. Note that there are unicode in the names. 
City Group: Type of the city. Big cities, or Other. 
Type: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
P1, P2 - P37: There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
Revenue: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. Please note that the values are transformed so they don't mean real dollar values. 
""",
    "dogs-vs-cats-redux-kernels-edition": """The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. 
For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).
""",
    "home-data-for-ml-course": """{
    "name": "home-data-for-ml-course",
    "description": "",
    "target_col": "SalePrice",
    "metadata": {
        "NumberOfClasses": 663,
        "NumberOfFeatures": 81,
        "NumberOfInstances": 1460,
        "NumberOfInstancesWithMissingValues": 1460,
        "NumberOfMissingValues": 7829,
        "NumberOfNumericFeatures": 38,
        "NumberOfSymbolicFeatures": 43
    },
    "df_head": " Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1 Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea PoolQC Fence MiscFeature  MiscVal  MoSold  YrSold SaleType SaleCondition  SalePrice\n  1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl    AllPub    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story            7            5       2003          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace       196.0        Gd        TA      PConc       Gd       TA           No          GLQ         706          Unf           0        150          856    GasA        Ex          Y      SBrkr       856       854             0       1710             1             0         2         1             3             1          Gd             8        Typ           0         NaN     Attchd       2003.0          RFn           2         548         TA         TA          Y           0           61              0          0            0         0    NaN   NaN         NaN        0       2    2008       WD        Normal     208500\n  2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl    AllPub       FR2       Gtl      Veenker      Feedr       Norm     1Fam     1Story            6            8       1976          1976     Gable  CompShg     MetalSd     MetalSd        NaN         0.0        TA        TA     CBlock       Gd       TA           Gd          ALQ         978          Unf           0        284         1262    GasA        Ex          Y      SBrkr      1262         0             0       1262             0             1         2         0             3             1          TA             6        Typ           1          TA     Attchd       1976.0          RFn           2         460         TA         TA          Y         298            0              0          0            0         0    NaN   NaN         NaN        0       5    2007       WD        Normal     181500\n  3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl    AllPub    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story            7            5       2001          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace       162.0        Gd        TA      PConc       Gd       TA           Mn          GLQ         486          Unf           0        434          920    GasA        Ex          Y      SBrkr       920       866             0       1786             1             0         2         1             3             1          Gd             6        Typ           1          TA     Attchd       2001.0          RFn           2         608         TA         TA          Y           0           42              0          0            0         0    NaN   NaN         NaN        0       9    2008       WD        Normal     223500\n  4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl    AllPub    Corner       Gtl      Crawfor       Norm       Norm     1Fam     2Story            7            5       1915          1970     Gable  CompShg     Wd Sdng     Wd Shng        NaN         0.0        TA        TA     BrkTil       TA       Gd           No          ALQ         216          Unf           0        540          756    GasA        Gd          Y      SBrkr       961       756             0       1717             1             0         1         0             3             1          Gd             7        Typ           1          Gd     Detchd       1998.0          Unf           3         642         TA         TA          Y           0           35            272          0            0         0    NaN   NaN         NaN        0       2    2006       WD       Abnorml     140000\n  5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl    AllPub       FR2       Gtl      NoRidge       Norm       Norm     1Fam     2Story            8            5       2000          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace       350.0        Gd        TA      PConc       Gd       TA           Av          GLQ         655          Unf           0        490         1145    GasA        Ex          Y      SBrkr      1145      1053             0       2198             1             0         2         1             4             1          Gd             9        Typ           1          TA     Attchd       2000.0          RFn           3         836         TA         TA          Y         192           84              0          0            0         0    NaN   NaN         NaN        0      12    2008       WD        Normal     250000"
}
""",
    "nlp1000-ml-challenge": """Dataset Description
The following describes the files present and the columns included.

Files
train.csv - the training set
test.csv - the test set (also includes a validation set; validation -> public scores, test -> private scores)
sample_submission.csv - a sample submission file in the correct format
Columns
DocumentId - unique identifier of a tweet
Text - the text of the tweet
IsPositive - a Boolean value describing whether a tweet is positive (i.e. True) or negative (i.e. False)

When loading in your data, please consider loading in the csv files as such
```python
import pandas as pd
import csv

data_df = pd.read_csv('path/to/file.csv', quoting=csv.QUOTE_ALL, encoding="utf-8", index_col=0)
```
""",
    "nlpsci": """Dataset Description
train data: 15972

test data : 5000

class: 6

0 Computer Science

1 Physics

2 Mathematics

3 Statistics

4 Quantitative Biology

5 Quantitative Finance

The column names are: "ID", "TITLE", "ABSTRACT", "label".
""",
    "sign-language-image-classification": """The sign language database of hand gestures represents a multi-class problem with 20 classes of letters and numbers (excluding some classes which require motion).
Dataset Description
File descriptions
train.csv - the training set
test.csv - the test set
submit.csv - a sample submission file in the correct format
Data fields
id - an image name
label - a label of the image
predict - a result of the image prediction
""",
    "world-championship-2023-embryo-classification": """Provided by Hung Vuong Hospital in Ho Chi Minh City, this dataset offers a unique observation into the early stages of embryo development. 
It comprises images of embryos at two crucial time points, day-3 and day-5. Each image is labeled as either 'good' or 'not good,' reflecting the embryo's potential for successful implantation.

Dataset Description
Welcome to the "Hung Vuong Hospital Embryo Classification" dataset. This page provides a comprehensive overview of the data files, their formats, and the essential columns you'll encounter in this competition. Taking a moment to understand the data will help you navigate the challenge effectively and make informed decisions during your analysis and modeling.

The dataset comprises the following key files:

train folder - Contains images of embryos at day-3 and day-5 for training purposes.
test folder - Contains images of embryos at day-3 and day-5 for testing purposes.
train.csv - Contains information about the training set.
test.csv - Contains information about the test set.
sample_submission.csv - A sample submission file that demonstrates the correct submission format.
Data Format Expectations

The embryo images are arranged within subfolders under the train and test directories. Each image is saved in JPG format and is labeled with a prefix. Images corresponding to day-3 embryos have the prefix D3 while images related to day-5 embryos bear the prefix D5. This prefix-based categorization allows for easy identification of the embryo's developmental stage.

Expected Output

Your task in this competition is to create a deep learning model that can accurately classify embryo images as 1 for good or 0 for not good for both day-3 and day-5 stages. The model should be trained on the training set and then used to predict the embryo quality in the test set. The ID column assigns an ID to each image. You will create the Class column as the result of model classification. The submission file contains only 2 columns: ID and Class (See the sample submission file)

Columns

You will encounter the following columns throughout the dataset:

ID - Refers to the ID of the images in the test set.
Image - Refers to the file name of the embryo images in the train or test folder.
Class - Represents the evaluation of the embryo images. This column provides the ground truth label for each image, indicating whether the embryo is classified as 'good' or 'not good'.
We encourage you to explore, analyze, and preprocess the provided data to build a robust model for accurate embryo quality classification. Good luck, and may your innovative solutions contribute to advancements in reproductive science!
""",
}

sela_data_infos = {
    "mfeat-factors": """{
    "name": "mfeat-factors",
    "description": "**Author**: Robert P.W. Duin, Department of Applied Physics, Delft University of Technology  \n\n**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Multiple+Features) - 1998  \n\n**Please cite**: [UCI](https://archive.ics.uci.edu/ml/citation_policy.html)   \n\n\n\n**Multiple Features Dataset: Factors**  \n\nOne of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps. Corresponding patterns in different datasets correspond to the same original character. 200 instances per class (for a total of 2,000 instances) have been digitized in binary images. \n\n\n\n### Attribute Information  \n\nThe attributes represent 216 profile correlations. No more information is known.\n\n\n\n### Relevant Papers  \n\nA slightly different version of the database is used in  \n\nM. van Breukelen, R.P.W. Duin, D.M.J. Tax, and J.E. den Hartog, Handwritten digit recognition by combined classifiers, Kybernetika, vol. 34, no. 4, 1998, 381-386.\n\n \n\nThe database as is is used in:  \n\nA.K. Jain, R.P.W. Duin, J. Mao, Statistical Pattern Recognition: A Review, IEEE Transactions on Pattern Analysis and Machine Intelligence archive, Volume 22 Issue 1, January 2000",
    "target_col": "class",
    "metadata": {
        "NumberOfClasses": 10.0,
        "NumberOfFeatures": 217.0,
        "NumberOfInstances": 2000.0,
        "NumberOfInstancesWithMissingValues": 0.0,
        "NumberOfMissingValues": 0.0,
        "NumberOfNumericFeatures": 216.0,
        "NumberOfSymbolicFeatures": 1.0,
        "AutoCorrelation": 0.9954977488744372,
        "CfsSubsetEval_DecisionStumpAUC": 0.9360315277777779,
        "CfsSubsetEval_DecisionStumpErrRate": 0.1335,
        "CfsSubsetEval_DecisionStumpKappa": 0.8516666666666667,
        "CfsSubsetEval_NaiveBayesAUC": 0.9360315277777779,
        "CfsSubsetEval_NaiveBayesErrRate": 0.1335,
        "CfsSubsetEval_NaiveBayesKappa": 0.8516666666666667,
        "CfsSubsetEval_kNN1NAUC": 0.9360315277777779,
        "CfsSubsetEval_kNN1NErrRate": 0.1335,
        "CfsSubsetEval_kNN1NKappa": 0.8516666666666667,
        "ClassEntropy": 3.321928094887362,
        "DecisionStumpAUC": 0.7207469444444443,
        "DecisionStumpErrRate": 0.8065,
        "DecisionStumpKappa": 0.10388888888888889,
        "Dimensionality": 0.1085,
        "EquivalentNumberOfAtts": NaN,
        "J48.00001.AUC": 0.9277102777777778,
        "J48.00001.ErrRate": 0.1375,
        "J48.00001.Kappa": 0.8472222222222223,
        "J48.0001.AUC": 0.9277102777777778,
        "J48.0001.ErrRate": 0.1375,
        "J48.0001.Kappa": 0.8472222222222223,
        "J48.001.AUC": 0.9277102777777778,
        "J48.001.ErrRate": 0.1375,
        "J48.001.Kappa": 0.8472222222222223,
        "MajorityClassPercentage": 10.0,
        "MajorityClassSize": 200.0,
        "MaxAttributeEntropy": NaN,
        "MaxKurtosisOfNumericAtts": 1.7012862999932459,
        "MaxMeansOfNumericAtts": 1141.5704999999996,
        "MaxMutualInformation": NaN,
        "MaxNominalAttDistinctValues": 10.0,
        "MaxSkewnessOfNumericAtts": 0.8630705307139119,
        "MaxStdDevOfNumericAtts": 153.51158138902838,
        "MeanAttributeEntropy": NaN,
        "MeanKurtosisOfNumericAtts": -0.315188200704828,
        "MeanMeansOfNumericAtts": 318.2703888888889,
        "MeanMutualInformation": NaN,
        "MeanNoiseToSignalRatio": NaN,
        "MeanNominalAttDistinctValues": 10.0,
        "MeanSkewnessOfNumericAtts": 0.027560516903782382,
        "MeanStdDevOfNumericAtts": 49.58131318395632,
        "MinAttributeEntropy": NaN,
        "MinKurtosisOfNumericAtts": -1.3893209589478623,
        "MinMeansOfNumericAtts": 6.7674999999999965,
        "MinMutualInformation": NaN,
        "MinNominalAttDistinctValues": 10.0,
        "MinSkewnessOfNumericAtts": -1.0738237890694697,
        "MinStdDevOfNumericAtts": 1.9854850353031415,
        "MinorityClassPercentage": 10.0,
        "MinorityClassSize": 200.0,
        "NaiveBayesAUC": 0.9919074999999999,
        "NaiveBayesErrRate": 0.0765,
        "NaiveBayesKappa": 0.915,
        "NumberOfBinaryFeatures": 0.0,
        "PercentageOfBinaryFeatures": 0.0,
        "PercentageOfInstancesWithMissingValues": 0.0,
        "PercentageOfMissingValues": 0.0,
        "PercentageOfNumericFeatures": 99.53917050691244,
        "PercentageOfSymbolicFeatures": 0.4608294930875576,
        "Quartile1AttributeEntropy": NaN,
        "Quartile1KurtosisOfNumericAtts": -0.6970467475463484,
        "Quartile1MeansOfNumericAtts": 14.118125000000006,
        "Quartile1MutualInformation": NaN,
        "Quartile1SkewnessOfNumericAtts": -0.2190263674003187,
        "Quartile1StdDevOfNumericAtts": 5.542689317909613,
        "Quartile2AttributeEntropy": NaN,
        "Quartile2KurtosisOfNumericAtts": -0.40269116562873153,
        "Quartile2MeansOfNumericAtts": 146.4797499999999,
        "Quartile2MutualInformation": NaN,
        "Quartile2SkewnessOfNumericAtts": -0.014119361298266794,
        "Quartile2StdDevOfNumericAtts": 30.290065676946117,
        "Quartile3AttributeEntropy": NaN,
        "Quartile3KurtosisOfNumericAtts": 0.01138630535997398,
        "Quartile3MeansOfNumericAtts": 684.5027499999998,
        "Quartile3MutualInformation": NaN,
        "Quartile3SkewnessOfNumericAtts": 0.36651259782511425,
        "Quartile3StdDevOfNumericAtts": 91.59687126428517,
        "REPTreeDepth1AUC": 0.9486284722222221,
        "REPTreeDepth1ErrRate": 0.1575,
        "REPTreeDepth1Kappa": 0.8250000000000001,
        "REPTreeDepth2AUC": 0.9486284722222221,
        "REPTreeDepth2ErrRate": 0.1575,
        "REPTreeDepth2Kappa": 0.8250000000000001,
        "REPTreeDepth3AUC": 0.9486284722222221,
        "REPTreeDepth3ErrRate": 0.1575,
        "REPTreeDepth3Kappa": 0.8250000000000001,
        "RandomTreeDepth1AUC": 0.8927777777777776,
        "RandomTreeDepth1ErrRate": 0.193,
        "RandomTreeDepth1Kappa": 0.7855555555555556,
        "RandomTreeDepth2AUC": 0.8927777777777776,
        "RandomTreeDepth2ErrRate": 0.193,
        "RandomTreeDepth2Kappa": 0.7855555555555556,
        "RandomTreeDepth3AUC": 0.8927777777777776,
        "RandomTreeDepth3ErrRate": 0.193,
        "RandomTreeDepth3Kappa": 0.7855555555555556,
        "StdvNominalAttDistinctValues": 0.0,
        "kNN1NAUC": 0.9753261111111112,
        "kNN1NErrRate": 0.044,
        "kNN1NKappa": 0.951111111111111
    },
    "df_head": " att1  att2  att3  att4  att5  att6  att7  att8  att9  att10  att11  att12  att13  att14  att15  att16  att17  att18  att19  att20  att21  att22  att23  att24  att25  att26  att27  att28  att29  att30  att31  att32  att33  att34  att35  att36  att37  att38  att39  att40  att41  att42  att43  att44  att45  att46  att47  att48  att49  att50  att51  att52  att53  att54  att55  att56  att57  att58  att59  att60  att61  att62  att63  att64  att65  att66  att67  att68  att69  att70  att71  att72  att73  att74  att75  att76  att77  att78  att79  att80  att81  att82  att83  att84  att85  att86  att87  att88  att89  att90  att91  att92  att93  att94  att95  att96  att97  att98  att99  att100  att101  att102  att103  att104  att105  att106  att107  att108  att109  att110  att111  att112  att113  att114  att115  att116  att117  att118  att119  att120  att121  att122  att123  att124  att125  att126  att127  att128  att129  att130  att131  att132  att133  att134  att135  att136  att137  att138  att139  att140  att141  att142  att143  att144  att145  att146  att147  att148  att149  att150  att151  att152  att153  att154  att155  att156  att157  att158  att159  att160  att161  att162  att163  att164  att165  att166  att167  att168  att169  att170  att171  att172  att173  att174  att175  att176  att177  att178  att179  att180  att181  att182  att183  att184  att185  att186  att187  att188  att189  att190  att191  att192  att193  att194  att195  att196  att197  att198  att199  att200  att201  att202  att203  att204  att205  att206  att207  att208  att209  att210  att211  att212  att213  att214  att215  att216  class\n   98   236   531   673   607   647     2     9     3      6      8      5    225    517    652    624    628    994      7     22     28     13     10     19    305    481    667    663   1009    727     38     28     18     11     20     10    287    567    651    742    824    900     26     34     30      8     16     13    248    556    631    796    926    748     39     34     18      9     17     12    248    540    506    814   1051    728     38     28      5     13     16      8    246    518    751    579    699   1062     13     30     28     10     16     16    276    344    682    500    709    916     10     30     23     17     15     14    357    435    829     610     745     994      20       7      24      12      10       9     355     409     477     886     976     723      30      24      14       7       7       8     290     352     435     753     894     751      29      29       2      13      13      14     260     286     562     698     665     757      11       8      15      14       9       9     238     292     586     698     733     707       9       6      15      11      10      16     294     406     654     644     741    1000      18      23      17       9      20      11     302     418     561     709     961     776      21      25      12      12      19      10     360     328     607     984    1186     599      29       7      14       6       9       9     362     314     924     733     601    1216       4       8      20      10       9       5     251     421     474     536     628     632      18      36       8      15      12      13      1\n  121   193   607   611   585   665     7     9     2      4      3      7    214    514    690    548    630   1006      2     18     31     15      5     15    260    344    655    605    993    743     29     32     25     11     17      6    308    412    719    700    784    902     31     32     37      8     13     15    259    429    699    758    900    762     34     30     25      9     14     10    207    445    558    784   1037    750     29     30     12     11     17      6    199    427    791    553    707   1074     10     26     31     12     17     10    253    345    726    440    731    932      7     26     30     19     10     12    326    452    795     540     701    1012      17       7      31      12      11      11     320     264     563     796     972     741      29      20      17       7       8       6     243     255     467     711     896     769      26      25       9       9      12      10     265     171     648     626     687     777      16      10      12      10       4       5     279     269     634     628     737     727      14      10      12      13       5      14     253     317     680     614     755    1016      15      19      24      11      17      11     255     321     617     651     979     792      20      29      19       8      18       8     351     249     687     922    1194     617      26      11      11       4      10       7     347     439     950     675     577    1222      13       6      27      10      10       3     224     354     520     458     570     634      15      32      11      13      15      11      1\n  115   141   590   605   557   627    12     6     3      3      5      4    196    404    611    560    594    986      7     21     28     14      7     12    278    406    670    579    991    703     24     29     18     12     19      9    378    468    668    654    786    900     32     31     28     11     17     12    299    495    650    698    902    726     29     31     18     12     16      5    249    389    465    722   1033    704     24     27      5     12     17     11    229    377    700    483    675   1056      5     29     28     11     17     15    247    309    645    444    697    898      6     29     21     18     12      7    312    328    774     550     689     980      14       6      20      11       9       6     360     294     568     790     956     733      24      23      18       6       8      11     235     259     436     665     876     719      21      28       0      10      14      13     253     111     653     638     653     737      17       9      15      11       6      10     297     185     637     638     715     677       9       9      15      12       7       9     271     335     591     542     719     970      10      22      13      12      19       6     293     335     536     613     943     746      15      26      10       9      20      13     415     323     672     896    1168     595      21       8      14       3      10      12     343     365     877     663     547    1192      18       7      20      13      10       2     196     348     535     498     572     656      20      35      16      14      13       6      1\n   90   122   627   692   607   642     0     6     4      5      3      5    201    445    664    629    626    945      5     21     27     16      7      9    273    397    703    662    987    710     36     29     21      8     15     10    331    459    731    765    818    865     24     29     33      9     15     11    256    510    707    801    900    735     37     29     21     10     16      2    246    438    534    817   1031    721     36     27      8     10     15      8    248    390    767    586    693   1013     11     29     27     13     15     12    278    312    702    509    707    873      8     29     26     18      8      8    361    347    831     631     739     955      18       8      27      13      13       9     369     293     589     879     962     718      28      23      13       8      10       8     268     364     449     754     884     728      27      28       5       8      10      14     232     172     674     713     663     722       9       9      14       9       2       9     244     202     682     719     725     684       7       9      16      14       5       6     260     412     640     651     735     973      16      22      20      12      15       3     276     416     601     712     959     757      19      26      15       7      16      10     380     354     711     987    1180     606      27       8      15       5      12       9     400     358     918     756     603    1171       6       7      23      11      12       5     263     419     576     549     628     621      16      35       7      12      15       9      1\n  157   167   681   666   587   666     8     6     1      4      5      5    212    386    740    613    614    997      5     21     30     11      7     17    248    390    703    662    949    744     28     29     20     13     19     10    244    468    797    743    750    909     32     29     32     10     15     11    231    479    773    799    850    769     37     29     20     11     14     10    189    361    620    819    995    743     32     27      7     15     17      8    183    365    851    588    701   1063     13     29     30      8     17     14    257    237    782    501    715    925     10     31     25     15     12     12    314    334    847     599     689    1003      16       8      26       8       9      13     310     300     633     871     936     752      32      23      16       7       8       8     257     233     511     756     860     758      29      30       4      11      14      14     265     137     718     687     671     768      17       9      13      10       6       9     253     169     704     687     707     714      15       9      13       9       7      14     251     283     714     651     733    1009      18      22      19       7      19       9     239     281     685     716     955     785      23      26      14       8      20       6     337     303     763     989    1156     628      29       8      12       2      10       7     321     337     988     726     577    1225      14       7      22       8      10       5     276     342     594     525     568     653      16      35      10      15      13      13      1"
}""",
    "colleges": """{
    "name": "colleges",
    "description": "Modified version for the automl benchmark.\nRegroups information for about 7800 different US colleges. Including geographical information, stats about the population attending and post graduation career earnings.",
    "target_col": "percent_pell_grant",
    "metadata": {
        "NumberOfClasses": 0.0,
        "NumberOfFeatures": 45.0,
        "NumberOfInstances": 7063.0,
        "NumberOfInstancesWithMissingValues": 7063.0,
        "NumberOfMissingValues": 104249.0,
        "NumberOfNumericFeatures": 33.0,
        "NumberOfSymbolicFeatures": 12.0,
        "AutoCorrelation": 0.7872184225431891,
        "Dimensionality": 0.006371230355373071,
        "MajorityClassPercentage": NaN,
        "MajorityClassSize": NaN,
        "MinorityClassPercentage": NaN,
        "MinorityClassSize": NaN,
        "NumberOfBinaryFeatures": 0.0,
        "PercentageOfBinaryFeatures": 0.0,
        "PercentageOfInstancesWithMissingValues": 100.0,
        "PercentageOfMissingValues": 32.79972312677962,
        "PercentageOfNumericFeatures": 73.33333333333333,
        "PercentageOfSymbolicFeatures": 26.666666666666668
    },
    "df_head": "      city state        zip  latitude  longitude  admission_rate  sat_verbal_midrange  sat_math_midrange  sat_writing_midrange  act_combined_midrange  act_english_midrange  act_math_midrange  act_writing_midrange  sat_total_average  undergrad_size  percent_white  percent_black  percent_hispanic  percent_asian  percent_part_time  average_cost_academic_year  average_cost_program_year  tuition_(instate)  tuition_(out_of_state)  spend_per_student  faculty_salary  percent_part_time_faculty  percent_pell_grant  completion_rate predominant_degree highest_degree         ownership                                                     region gender                        carnegie_basic_classification                                                                                  carnegie_undergraduate                                                  carnegie_size religious_affiliation  percent_female  agege24   faminc  mean_earnings_6_years  median_earnings_6_years  mean_earnings_10_years  median_earnings_10_years\n    Normal    AL      35762   34.7834   -86.5685          0.8989                410.0              400.0                   NaN                   17.0                  17.0               17.0                   NaN              823.0          4051.0         0.0279         0.9501            0.0089         0.0022             0.0622                     18888.0                        NaN             7182.0                 12774.0             7459.0          7079.0                     0.8856              0.7115           0.2914          Bachelors       Graduate            Public Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)   COED Master's Colleges and Universities (larger programs)                                                                          Full-time four-year, inclusive             Medium 4-year, highly residential (3,000 to 9,999)                   NaN            0.53     0.08 40211.22                26100.0                  22800.0                 35300.0                   31400.0\nBirmingham    AL 35294-0110   33.5022   -86.8092          0.8673                580.0              585.0                   NaN                   25.0                  26.0               23.0                   NaN             1146.0         11200.0         0.5987         0.2590            0.0258         0.0518             0.2579                     19990.0                        NaN             7206.0                 16398.0            17208.0         10170.0                     0.9106              0.3505           0.5377          Bachelors       Graduate            Public Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)   COED  Research Universities (very high research activity)                                               Medium full-time four-year, selective, higher transfer-in            Large 4-year, primarily nonresidential (over 9,999)                   NaN            0.65     0.26 49894.65                37400.0                  33200.0                 46300.0                   40300.0\nMontgomery    AL 36117-3553   32.3626   -86.1740             NaN                  NaN                NaN                   NaN                    NaN                   NaN                NaN                   NaN                NaN           322.0         0.2919         0.4224            0.0093         0.0031             0.3727                     12300.0                        NaN             6870.0                  6870.0             5123.0          3849.0                     0.6721              0.6839           0.6667          Bachelors       Graduate Private nonprofit Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)   COED              Baccalaureate Colleges--Arts & Sciences Medium full-time four-year, inclusivestudents with varying levels academic preparation and achievement) Very small 4-year, primarily nonresidential (less than 1,000,)    Churches of Christ            0.51     0.83 38712.18                38500.0                  32800.0                 42100.0                   38100.0\nHuntsville    AL      35899   34.7228   -86.6384          0.8062                575.0              580.0                   NaN                   26.0                  26.0               25.0                   NaN             1180.0          5525.0         0.7012         0.1310            0.0338         0.0364             0.2395                     20306.0                        NaN             9192.0                 21506.0             9352.0          9341.0                     0.6555              0.3281           0.4835          Bachelors       Graduate            Public Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)   COED  Research Universities (very high research activity)                                               Medium full-time four-year, selective, higher transfer-in       Medium 4-year, primarily nonresidential (3,000 to 9,999)                   NaN            0.55     0.29 54155.40                39300.0                  36700.0                 52700.0                   46600.0\nMontgomery    AL 36104-0271   32.3643   -86.2957          0.5125                430.0              425.0                   NaN                   17.0                  17.0               17.0                   NaN              830.0          5354.0         0.0161         0.9285            0.0114         0.0015             0.0902                     17400.0                        NaN             8720.0                 15656.0             7393.0          6557.0                     0.6641              0.8265           0.2517          Bachelors       Graduate            Public Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)   COED Master's Colleges and Universities (larger programs)                                                                          Full-time four-year, inclusive          Medium 4-year, primarily residential (3,000 to 9,999)                   NaN            0.57     0.11 31846.99                21200.0                  19300.0                 30700.0                   27800.0"
}""",
    "smoker-status": """{
    "name": "smoker-status",
    "description": "",
    "target_col": "smoking",
    "metadata": {
        "NumberOfClasses": 2,
        "NumberOfFeatures": 23,
        "NumberOfInstances": 143330,
        "NumberOfInstancesWithMissingValues": 0,
        "NumberOfMissingValues": 0,
        "NumberOfNumericFeatures": 23,
        "NumberOfSymbolicFeatures": 0
    },
    "df_head": " age  height(cm)  weight(kg)  waist(cm)  eyesight(left)  eyesight(right)  hearing(left)  hearing(right)  systolic  relaxation  fasting blood sugar  Cholesterol  triglyceride  HDL  LDL  hemoglobin  Urine protein  serum creatinine  AST  ALT  Gtp  dental caries  smoking\n  35         175          80       84.0             0.8              1.0              1               1       122          70                  100          188           238   44   96        16.0              1               0.7   31   22   32              0        1\n  35         185          80       88.0             1.2              1.2              1               1       100          64                   92          154           122   49   80        15.0              1               0.9   22   22   17              0        1\n  20         160          60       76.0             1.2              1.2              1               1       130          85                   90          194           134   34  134        16.9              1               0.8   24   32   41              1        1\n  60         155          60       87.8             0.7              0.2              1               1       119          62                   91          261           103   73  167        13.2              1               0.9   21   16   14              0        0\n  60         165          70       85.0             0.8              0.6              1               1       142          93                  108          179           109   49  108        15.5              1               0.7   27   40   61              0        0"
}""",
    "wine-quality-white": """{
    "name": "wine-quality-white",
    "description": "Citation Request:\n  This dataset is public available for research. The details are described in [Cortez et al., 2009]. \n  Please include this citation if you plan to use this database:\n\n  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. \n  Modeling wine preferences by data mining from physicochemical properties.\n  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.\n\n  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016\n                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf\n                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib\n\n1. Title: Wine Quality \n\n2. Sources\n   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009\n   \n3. Past Usage:\n\n  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. \n  Modeling wine preferences by data mining from physicochemical properties.\n  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.\n\n  In the above reference, two datasets were created, using red and white wine samples.\n  The inputs include objective tests (e.g. PH values) and the output is based on sensory data\n  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality \n  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model\n  these datasets under a regression approach. The support vector machine model achieved the\n  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),\n  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity\n  analysis procedure).\n \n4. Relevant Information:\n\n   The two datasets are related to red and white variants of the Portuguese &quot;Vinho Verde&quot; wine.\n   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].\n   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables \n   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).\n\n   These datasets can be viewed as classification or regression tasks.\n   The classes are ordered and not balanced (e.g. there are munch more normal wines than\n   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent\n   or poor wines. Also, we are not sure if all input variables are relevant. So\n   it could be interesting to test feature selection methods. \n\n5. Number of Instances: red wine - 1599; white wine - 4898. \n\n6. Number of Attributes: 11 + output attribute\n  \n   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of\n   feature selection.\n\n7. Attribute information:\n\n   For more information, read [Cortez et al., 2009].\n\n   Input variables (based on physicochemical tests):\n   1 - fixed acidity\n   2 - volatile acidity\n   3 - citric acid\n   4 - residual sugar\n   5 - chlorides\n   6 - free sulfur dioxide\n   7 - total sulfur dioxide\n   8 - density\n   9 - pH\n   10 - sulphates\n   11 - alcohol\n   Output variable (based on sensory data): \n   12 - quality (score between 0 and 10)\n\n8. Missing Attribute Values: None",
    "target_col": "Class",
    "metadata": {
        "NumberOfClasses": 7.0,
        "NumberOfFeatures": 12.0,
        "NumberOfInstances": 4898.0,
        "NumberOfInstancesWithMissingValues": 0.0,
        "NumberOfMissingValues": 0.0,
        "NumberOfNumericFeatures": 11.0,
        "NumberOfSymbolicFeatures": 1.0,
        "AutoCorrelation": 0.4124974474167858,
        "ClassEntropy": 1.8617149332355558,
        "Dimensionality": 0.002449979583503471,
        "EquivalentNumberOfAtts": NaN,
        "MajorityClassPercentage": 44.87545937117191,
        "MajorityClassSize": 2198.0,
        "MaxAttributeEntropy": NaN,
        "MaxKurtosisOfNumericAtts": 37.564599706679964,
        "MaxMeansOfNumericAtts": 138.36065741118807,
        "MaxMutualInformation": NaN,
        "MaxNominalAttDistinctValues": 7.0,
        "MaxSkewnessOfNumericAtts": 5.0233306827597035,
        "MaxStdDevOfNumericAtts": 42.49806455414295,
        "MeanAttributeEntropy": NaN,
        "MeanKurtosisOfNumericAtts": 7.066218779083282,
        "MeanMeansOfNumericAtts": 18.432687072385747,
        "MeanMutualInformation": NaN,
        "MeanNoiseToSignalRatio": NaN,
        "MeanNominalAttDistinctValues": 7.0,
        "MeanSkewnessOfNumericAtts": 1.3004201639647581,
        "MeanStdDevOfNumericAtts": 6.105775283618253,
        "MinAttributeEntropy": NaN,
        "MinKurtosisOfNumericAtts": -0.6984253303455836,
        "MinMeansOfNumericAtts": 0.045772356063699476,
        "MinMutualInformation": NaN,
        "MinNominalAttDistinctValues": 7.0,
        "MinSkewnessOfNumericAtts": 0.3907098416536844,
        "MinStdDevOfNumericAtts": 0.002990906916936935,
        "MinorityClassPercentage": 0.10208248264597795,
        "MinorityClassSize": 5.0,
        "NumberOfBinaryFeatures": 0.0,
        "PercentageOfBinaryFeatures": 0.0,
        "PercentageOfInstancesWithMissingValues": 0.0,
        "PercentageOfMissingValues": 0.0,
        "PercentageOfNumericFeatures": 91.66666666666666,
        "PercentageOfSymbolicFeatures": 8.333333333333332,
        "Quartile1AttributeEntropy": NaN,
        "Quartile1KurtosisOfNumericAtts": 0.571853233353457,
        "Quartile1MeansOfNumericAtts": 0.33419150673744386,
        "Quartile1MutualInformation": NaN,
        "Quartile1SkewnessOfNumericAtts": 0.48734199235899067,
        "Quartile1StdDevOfNumericAtts": 0.10079454842486446,
        "Quartile2AttributeEntropy": NaN,
        "Quartile2KurtosisOfNumericAtts": 3.4698201025636317,
        "Quartile2MeansOfNumericAtts": 3.1882666394446715,
        "Quartile2MutualInformation": NaN,
        "Quartile2SkewnessOfNumericAtts": 0.9777730048688807,
        "Quartile2StdDevOfNumericAtts": 0.15100059961506782,
        "Quartile3AttributeEntropy": NaN,
        "Quartile3KurtosisOfNumericAtts": 9.793806910763827,
        "Quartile3MeansOfNumericAtts": 10.514267046957942,
        "Quartile3MutualInformation": NaN,
        "Quartile3SkewnessOfNumericAtts": 1.4067449205303153,
        "Quartile3StdDevOfNumericAtts": 5.072057784014863,
        "StdvNominalAttDistinctValues": 0.0
    },
    "df_head": " V1   V2   V3   V4    V5   V6    V7     V8   V9  V10  V11  Class\n7.0 0.27 0.36 20.7 0.045 45.0 170.0 1.0010 3.00 0.45  8.8      4\n6.3 0.30 0.34  1.6 0.049 14.0 132.0 0.9940 3.30 0.49  9.5      4\n8.1 0.28 0.40  6.9 0.050 30.0  97.0 0.9951 3.26 0.44 10.1      4\n7.2 0.23 0.32  8.5 0.058 47.0 186.0 0.9956 3.19 0.40  9.9      4\n7.2 0.23 0.32  8.5 0.058 47.0 186.0 0.9956 3.19 0.40  9.9      4"
}""",
    "click-prediction-small": """{
    "name": "Click_prediction_small",
    "description": "This is the same data as version 5 (OpenML ID = 1220) with '_id' features coded as nominal factor variables.",
    "target_col": "click",
    "metadata": {
        "NumberOfClasses": 2.0,
        "NumberOfFeatures": 12.0,
        "NumberOfInstances": 39948.0,
        "NumberOfInstancesWithMissingValues": 0.0,
        "NumberOfMissingValues": 0.0,
        "NumberOfNumericFeatures": 5.0,
        "NumberOfSymbolicFeatures": 7.0,
        "AutoCorrelation": 0.7170250582021178,
        "Dimensionality": 0.00030039050765995795,
        "MajorityClassPercentage": 83.15810553719835,
        "MajorityClassSize": 33220.0,
        "MinorityClassPercentage": 16.841894462801644,
        "MinorityClassSize": 6728.0,
        "NumberOfBinaryFeatures": 1.0,
        "PercentageOfBinaryFeatures": 8.333333333333332,
        "PercentageOfInstancesWithMissingValues": 0.0,
        "PercentageOfMissingValues": 0.0,
        "PercentageOfNumericFeatures": 41.66666666666667,
        "PercentageOfSymbolicFeatures": 58.333333333333336
    },
    "df_head": " click  impression     url_hash    ad_id  advertiser_id  depth  position  query_id  keyword_id  title_id  description_id  user_id\n     0         1.0 1.071003e+19  8343295          11700      3         3 7702266.0       21264     27892            1559        0\n     1         1.0 1.736385e+19 20017077          23798      1         1   93079.0       35498         4           36476   562934\n     0         1.0 8.915473e+18 21348354          36654      1         1   10981.0       19975     36105           33292 11621116\n     0         1.0 4.426693e+18 20366086          33280      3         3       0.0        5942      4057            4390  8778348\n     0         1.0 1.157260e+19  6803526          10790      2         1 9881978.0       60593     25242            1679 12118311"
}""",
    "house-prices": """{
    "name": "05_house-prices-advanced-regression-techniques",
    "description": "",
    "target_col": "SalePrice",
    "metadata": {
        "NumberOfClasses": 663,
        "NumberOfFeatures": 81,
        "NumberOfInstances": 1460,
        "NumberOfInstancesWithMissingValues": 1460,
        "NumberOfMissingValues": 7829,
        "NumberOfNumericFeatures": 38,
        "NumberOfSymbolicFeatures": 43
    },
    "df_head": " Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1 Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea PoolQC Fence MiscFeature  MiscVal  MoSold  YrSold SaleType SaleCondition  SalePrice\n  1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl    AllPub    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story            7            5       2003          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace       196.0        Gd        TA      PConc       Gd       TA           No          GLQ         706          Unf           0        150          856    GasA        Ex          Y      SBrkr       856       854             0       1710             1             0         2         1             3             1          Gd             8        Typ           0         NaN     Attchd       2003.0          RFn           2         548         TA         TA          Y           0           61              0          0            0         0    NaN   NaN         NaN        0       2    2008       WD        Normal     208500\n  2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl    AllPub       FR2       Gtl      Veenker      Feedr       Norm     1Fam     1Story            6            8       1976          1976     Gable  CompShg     MetalSd     MetalSd        NaN         0.0        TA        TA     CBlock       Gd       TA           Gd          ALQ         978          Unf           0        284         1262    GasA        Ex          Y      SBrkr      1262         0             0       1262             0             1         2         0             3             1          TA             6        Typ           1          TA     Attchd       1976.0          RFn           2         460         TA         TA          Y         298            0              0          0            0         0    NaN   NaN         NaN        0       5    2007       WD        Normal     181500\n  3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl    AllPub    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story            7            5       2001          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace       162.0        Gd        TA      PConc       Gd       TA           Mn          GLQ         486          Unf           0        434          920    GasA        Ex          Y      SBrkr       920       866             0       1786             1             0         2         1             3             1          Gd             6        Typ           1          TA     Attchd       2001.0          RFn           2         608         TA         TA          Y           0           42              0          0            0         0    NaN   NaN         NaN        0       9    2008       WD        Normal     223500\n  4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl    AllPub    Corner       Gtl      Crawfor       Norm       Norm     1Fam     2Story            7            5       1915          1970     Gable  CompShg     Wd Sdng     Wd Shng        NaN         0.0        TA        TA     BrkTil       TA       Gd           No          ALQ         216          Unf           0        540          756    GasA        Gd          Y      SBrkr       961       756             0       1717             1             0         1         0             3             1          Gd             7        Typ           1          Gd     Detchd       1998.0          Unf           3         642         TA         TA          Y           0           35            272          0            0         0    NaN   NaN         NaN        0       2    2006       WD       Abnorml     140000\n  5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl    AllPub       FR2       Gtl      NoRidge       Norm       Norm     1Fam     2Story            8            5       2000          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace       350.0        Gd        TA      PConc       Gd       TA           Av          GLQ         655          Unf           0        490         1145    GasA        Ex          Y      SBrkr      1145      1053             0       2198             1             0         2         1             4             1          Gd             9        Typ           1          TA     Attchd       2000.0          RFn           3         836         TA         TA          Y         192           84              0          0            0         0    NaN   NaN         NaN        0      12    2008       WD        Normal     250000"
    }""",
    "concrete-strength": """{
    "name": "concrete-strength",
    "description": "",
    "target_col": "Strength",
    "metadata": {
        "NumberOfClasses": 842,
        "NumberOfFeatures": 9,
        "NumberOfInstances": 4866,
        "NumberOfInstancesWithMissingValues": 0,
        "NumberOfMissingValues": 0,
        "NumberOfNumericFeatures": 9,
        "NumberOfSymbolicFeatures": 0
    },
    "df_head": " CementComponent  BlastFurnaceSlag  FlyAshComponent  WaterComponent  SuperplasticizerComponent  CoarseAggregateComponent  FineAggregateComponent  AgeInDays  Strength\n           255.0               0.0              0.0           192.0                        0.0                     889.8                   845.0          3     32.40\n           397.0               0.0              0.0           192.0                        0.0                     940.6                   785.6          7     18.20\n           212.5               0.0            121.6           172.3                        8.7                    1043.6                   759.3        100     44.52\n           339.0               0.0            158.0           196.0                       10.6                     949.0                   784.0         28     22.93\n           255.0               0.0              0.0           192.0                        0.0                     889.8                   945.0         90     21.50"
}""",
}
