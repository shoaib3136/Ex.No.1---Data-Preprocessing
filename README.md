# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Name:Shaik Shoaib Nawaz
Reg no:212222240094

# Importing Libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset
df=pd.read_csv('Churn_Modelling.csv')
df

#Checking for null values
df.isnull().sum()

#Checking for dulpicated values
df.duplicated()

#Dropping unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df

#Normalising using MinMaxScaler
ms=MinMaxScaler()
df2=pd.DataFrame(ms.fit_transform(df))
df2

#Splitting the dataset - x
X=df2.iloc[:,:-1].values
X

#Splitting the dataset - y
y=df2.iloc[:,-1].values
y

# Training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
```

## OUTPUT:
i.)Read the Dataset:
![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/31cf0f1d-43c5-4d26-8683-023094b42542)







ii.) Checking Null values:

![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/2ac109f2-fb56-4229-a4be-f712391e6d0c)





iii.) Dropping Unwanted columns:

![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/1821919d-b779-4df6-b6e2-109fc7906a24)







iv.) Normalising using MinMaxscaler:

![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/dcac4acd-f870-4f6f-b2ae-b2ec74042be8)







v.) Splitting the dataset-X:

![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/6106e7b2-a0a3-4e2e-942a-4a3d520213bb)





vi.) Splitting the dataset-Y:

![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/57731d65-be73-46d7-a4fd-386fb33aa881)



vii.)Training the dataset:

![image](https://github.com/shoaib3136/Ex.No.1---Data-Preprocessing/assets/117919362/bc080be1-9128-4d71-9ad5-03a8f079172b)







## RESULT:
The data set downloaded from kaggle is successfully processed.
