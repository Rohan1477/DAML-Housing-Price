import tensorflow as tf
#import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_file_path = "data/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.head(3)))

dataset_df = dataset_df.drop('Id', axis=1)

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

