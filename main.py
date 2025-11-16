import tensorflow as tf
#import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

train_file_path = "data/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.head(3)))

#dropping data with too many missing values
columns_to_drop = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage', 'GarageType', 'GarageYrBlt', 
                   'GarageFinish', 'GarageQual', 'GarageCond']
dataset_df = dataset_df.drop(columns_to_drop, axis=1)

#new features
dataset_df['TotalSqFt'] = dataset_df['GrLivArea'] + dataset_df['TotalBsmtSF']
dataset_df['TotalBaths'] = dataset_df['FullBath'] + (0.5 * dataset_df['HalfBath']) + dataset_df['BsmtFullBath'] + (0.5 * dataset_df['BsmtHalfBath'])


# one hot encoding
columns_to_encode = ['MSZoning', 'Street', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                     'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'PavedDrive', 'SaleType', 'SaleCondition'
]
dataset_df = pd.get_dummies(dataset_df, columns=columns_to_encode, drop_first=True)
print(f"Dataset shape after one hot encoding: {dataset_df.shape}")

#standardization

columns_to_standardize = ['GarageArea', 'OpenPorchSF', '1stFlrSF', 'TotalSqFt']
scaler = StandardScaler()
dataset_df[columns_to_standardize] = scaler.fit_transform(dataset_df[columns_to_standardize])

#function to visualize all of the attributes
def visualizeData():

    print(dataset_df['SalePrice'].describe())
    plt.figure(figsize=(9, 8))
    sns.histplot(dataset_df['SalePrice'], color='g', bins=100, alpha=0.4)
    plt.title("Distribution of SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.show()


    list(set(dataset_df.dtypes.tolist()))

    df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 12), bins=50, xlabelsize=8, ylabelsize=8)
    plt.suptitle("Distributions of Numeric Features", fontsize=12)
    plt.tight_layout(pad=1.0)
    plt.show()

#visualizeData();
    
