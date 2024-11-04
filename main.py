#%% Importar datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz, plot_tree
from six import StringIO
from IPython.display import Image  
  
import pydotplus
from sklearn import tree
#%% Importar datos
df_datos = pd.read_csv('Tema_14.csv')
# %%
df_datos.describe()
df_datos.columns
# %%
#for i in df_datos.columns:
#    print(df_datos[i].value_counts(dropna=False))
df_datos['gill-attachment'].value_counts(dropna=False)

# %%
resultado = df_datos.groupby('gill-attachment',  dropna=False)['class'].value_counts()
resultado
# %%

df_datos['stem-root'].value_counts(dropna=False)

# %%
resultado = df_datos.groupby('stem-root',  dropna=False)['class'].value_counts()
resultado
# %%

df_datos['veil-type'].value_counts(dropna=False)

# %%
resultado = df_datos.groupby('veil-type',  dropna=False)['class'].value_counts()
resultado
# %%

df_datos['ring-type'].value_counts(dropna=False)

# %%
resultado = df_datos.groupby('ring-type',  dropna=False)['class'].value_counts()
resultado
# %% analisis 'cap-color', 'stem-color', 'veil-color', 'season'
columns_of_interest = ['cap-color', 'stem-color', 'veil-color', 'season']
null_summary = df_datos[columns_of_interest].isnull().sum()
print("Missing values per column:\n", null_summary)
# Fill missing values in 'cap-color', 'stem-color', and 'season' with the mode
df_datos['cap-color'] = df_datos['cap-color'].fillna(df_datos['cap-color'].mode()[0])
df_datos['stem-color'] = df_datos['stem-color'].fillna(df_datos['stem-color'].mode()[0])
df_datos['season'] = df_datos['season'].fillna(df_datos['season'].mode()[0])
# Optionally, drop 'veil-color' if necessary
#df_datos.drop(columns=['veil-color'], inplace=True)
# Create dummy variables
data_dummies = pd.get_dummies(df_datos[['cap-color', 'stem-color', 'season']], 
                              prefix=['cap-color', 'stem-color', 'season'])

# Combine dummies with the original data if desired
df_datos = pd.concat([df_datos, data_dummies], axis=1) # queremos modificar el original
# Verify no missing values in the transformed data
print("Missing values after imputation:\n", df_datos[columns_of_interest].isnull().sum())

# Preview the first few rows of the dummy variables
print(data_dummies.head())
# Drop original columns after dummies are created
df_datos.drop(columns=['cap-color', 'stem-color', 'season'], inplace=True)# %%

# %%
