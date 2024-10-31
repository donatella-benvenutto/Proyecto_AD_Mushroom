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
for i in df_datos.columns:
    print(df_datos[i].value_counts(dropna=False))

resultado = df_datos.groupby('spore-print-color',  dropna=False)['class'].value_counts()
resultado
# %%
