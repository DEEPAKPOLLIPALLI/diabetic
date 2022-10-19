from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('../input/diabetes.csv')
diabetes_data.head()
diabetes_data.info(verbose=True)
diabetes_data.describe()
diabetes_data.describe().T
diabetes_data_copy = diabetes_data.copy(deep = True)
diabetes_data_copy[['rangers','jacks','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['rangers','jacks','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(diabetes_data_copy.isnull().sum())
p = diabetes_data.hist(figsize = (20,20))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)
p = diabetes_data_copy.hist(figsize = (20,20))
diabetes_data.shape
sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()
import missingno as msno
p=msno.bar(diabetes_data)
color_wheel = {1: "#0392cf", 
               2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
p=diabetes_data.Outcome.value_counts().plot(kind="bar")
from pandas.tools.plotting import scatter_matrix
p=scatter_matrix(diabetes_data,figsize=(25, 25))
p=sns.pairplot(diabetes_data_copy, hue = 'Outcome')
plt.figure(figsize=(12,10)) 
p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')
plt.figure(figsize=(12,10))
p=sns.heatmap(diabetes_data_copy.corr()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
