#%% Importar datos

import numpy as np
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from scipy.stats import chi2_contingency
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
#%%[markdown]
## Importar Datos

df_datos = pd.read_csv('Tema_14.csv')
#%%[markdown]
#Columnas
df_datos.columns
#%%[markdown]
#### class
columnas_a_revisar = ['class']
df_datos = df_datos.dropna(subset=columnas_a_revisar)
df_datos['class'].value_counts(dropna=False)

#%%[markdown]
#### cap-diameter
df_datos['cap-diameter'] = pd.to_numeric(df_datos['cap-diameter'], errors='coerce')
#count=0
#for i in df_datos['cap-diameter']:
#    if(i>= 623.40):
#        count=count+1

median_value = df_datos.loc[df_datos['cap-diameter'] < 623.40, 'cap-diameter'].median()
df_datos.loc[df_datos['cap-diameter'] >= 623.40, 'cap-diameter'] = median_value



#%%
print(df_datos['cap-diameter'].value_counts(dropna=False))
df_datos['cap-diameter'] = np.where(df_datos['cap-diameter'].isna(), '0', df_datos['cap-diameter'])

#%%
print(df_datos['cap-diameter'].value_counts(dropna=False))

#%%[markdown]
#### cap-shape
df_datos['cap-shape'].value_counts(dropna=False)
#%%
columnas_a_revisar = ['cap-shape']
df_datos = df_datos.dropna(subset=columnas_a_revisar)
#%%
df_datos['cap-shape'].value_counts(dropna=False)

#%%[markdown]
#### cap-surface
df_datos = df_datos.drop(columns=['cap-surface'])

#%%
df_datos.columns

#%%[markdown]
#### cap-color
df_datos['cap-color'].value_counts(dropna=False)
#%%
df_datos['cap-color'] = df_datos['cap-color'].fillna(df_datos['cap-color'].mode()[0])
df_datos['cap-color'].value_counts(dropna=False)

#%%[markdown]
#### does-bruise-or-bleed
df_datos['does-bruise-or-bleed'].value_counts(dropna=False)
#%%
df_datos['does-bruise-or-bleed'] = df_datos['does-bruise-or-bleed'].fillna(df_datos['does-bruise-or-bleed'].mode()[0])
df_datos['does-bruise-or-bleed'].value_counts(dropna=False)

#%%[markdown]
#### gill-attachment
# Esta columna indica como se encuentra conectado las laminas al tallo del hongo
df_datos['gill-attachment'].value_counts(dropna=False)
#%%[markdown]
# Tiene valores como
#Adnato (a), Adnexo (x), Decurrente (d), Libres (e), Sinuadas (s), Poros (p), Ninguna (f)
# %%
resultado = df_datos.groupby('gill-attachment',  dropna=False)['class'].value_counts()
resultado
#%%[markdown]
# Lo primero que notamos es su gran cantidad de nulos, se podria completar por la moda.
# Al fijarnos en su relacion con la variable obejtivo obtenemos que no hay nunguna tendencia importante, es decir, el gill-atachment no parece tener relacion directa con la clase
# Es por esto que decidimos eliminar la columna
#%%
df_datos = df_datos.drop(columns=['gill-attachment'])
#%%
df_datos.columns

#%%[markdown]
#### gill-spacing

df_datos['gill-spacing'].value_counts(dropna=False)

#%%
df_datos = df_datos.drop(columns=['gill-spacing'])
#%%
df_datos.columns
#%%[markdown]
#### gill-color
df_datos['gill-color'].value_counts(dropna=False)
#%%
df_datos['gill-color'] = df_datos['gill-color'].fillna(df_datos['gill-color'].mode()[0])
df_datos['gill-color'].value_counts(dropna=False)

#%%[markdown]
#### stem-height
df_datos['stem-height'].value_counts(dropna=False)
#%%
df_datos['stem-height'] = pd.to_numeric(df_datos['stem-height'], errors='coerce')
#count=0
#for i in df_datos['stem-height']:
#    if(i>= 339.20):
#        count=count+1
#count
#%%
median_value = df_datos.loc[df_datos['stem-height'] < 339.20, 'stem-height'].median()

df_datos.loc[df_datos['stem-height'] >= 339.20, 'stem-height'] = median_value

#%%
df_datos['stem-height'] = df_datos['stem-height'].fillna(median_value)
#%%[markdown]
#### stem-width
df_datos['stem-width'].value_counts(dropna=False)
#%%
df_datos['stem-width'] = pd.to_numeric(df_datos['stem-width'], errors='coerce')
#count=0
#for i in df_datos['stem-width']:
#    if(i>= 1039.10):
#        count=count+1
#count
#%%
median_value = df_datos.loc[df_datos['stem-width'] < 1039.10, 'stem-width'].median()

df_datos.loc[df_datos['stem-width'] >= 1039.10, 'stem-width'] = median_value

#%%
df_datos['stem-width'] = df_datos['stem-width'].fillna(median_value)
#%%[markdown]
#### stem-root
# Esta variable separa en categorias la característica principal de la base del tallo: bulboso (b), hinchado (s), en forma de club (c), en forma de copa (u), igual (e), rizomorfos (z), enraizado (r).

# %%

df_datos['stem-root'].value_counts(dropna=False)
#%%[markdown]
# Se puede notar una gran cantidad de nulos, 50.000 de 61.000 por lo que no parece ser una buena columna a mantener 
# %%
resultado = df_datos.groupby('stem-root',  dropna=False)['class'].value_counts()
resultado

resultado.plot(kind='bar', stacked=True)
plt.title('Distribución de Stem Root por Edibilidad')
plt.xlabel('Stem Root')
plt.ylabel('Frecuencia')
plt.legend(title='Edibilidad', labels=['Edible', 'Poisonous'])
plt.show()
#%%[markdown]
# Realizando una agrupacion en base al tipo de base de tallo y la clase (venenoso, comestible) se puede notar que a pesar de tener alguna relacion interesante, la mayoria de datos son nulos y no nos ayudan para el posterior modelo.
# Por esto decidimos eliminar la columna
#%%
df_datos = df_datos.drop(columns=['stem-root'])

#%%[markdown]
#### stem-surface
df_datos['stem-surface'].value_counts(dropna=False)
#%%
df_datos = df_datos.drop(columns=['stem-surface'])
#%%[markdown]
#### stem-color
df_datos['stem-color'].value_counts(dropna=False)
#%%
df_datos['stem-color'] = df_datos['stem-color'].fillna(df_datos['stem-color'].mode()[0])
#%%[markdown]
#### veil-type
# Esta variable indica que tipo de velo tiene el hongo: universal (u) y parcial (p)
# %%

df_datos['veil-type'].value_counts(dropna=False)
#%%[markdown]
# En esta columna tambien tenemos un numero exagerado de nulos, ni siquiera hay valores p
# %%
resultado = df_datos.groupby('veil-type',  dropna=False)['class'].value_counts()
resultado
resultado.plot(kind='bar', stacked=True)
plt.title('Distribución de Stem Root por Edibilidad')
plt.xlabel('Stem Root')
plt.ylabel('Frecuencia')
plt.legend(title='Edibilidad', labels=['Edible', 'Poisonous'])
plt.show()
#%%[markdown]
# En cuanto a su distribucion en comparacion con la clase se puede observar que no hay una relacion directa lo que nos da aun mas motivos para eliminar esta columna
#%%
df_datos = df_datos.drop(columns=['veil-type'])
#%%[markdown]
#### veil-color

df_datos['veil-color'].value_counts(dropna=False)
#%%
df_datos = df_datos.drop(columns=['veil-color'])
#%%[markdown]
#### has-ring
df_datos['has-ring'].value_counts(dropna=False)
#%%
tieneanillo = df_datos.groupby('has-ring',  dropna=False)['ring-type'].value_counts()
tieneanillo
#%%
df_datos['has-ring'] = np.where(df_datos['has-ring'].isna() & (df_datos['ring-type'] == 'f'), 'f', df_datos['has-ring'])
df_datos['has-ring'] = np.where(df_datos['has-ring'].isna(), 't', df_datos['has-ring'])
#%%[markdown]
#### ring-type
# Esta variable cuenta forma y caracteristica del anillo de algunos tallos
# %%
df_datos['ring-type'].value_counts(dropna=False)
#%%[markdown]
#Esta varaible cuenta con los siguientes valores enredado (c), efímero (e), en forma de ala (r), con surcos (g), grande (l), colgante (p), envolvente (s), en forma de zona (z), escamoso (y), móvil (m), ninguno (f).
#Tambien cuenta con valores nulos, pero en menor medida

# %%
tieneanillo = df_datos.groupby('ring-type',  dropna=False)['has-ring'].value_counts()
tieneanillo

#%%[markdown]
#Como se puede obvservar a excepcion de f todos tienen el has-ring en true. Podriamos dividir los nulos en que si has-ring es f se ponga el valor f en ring-type y si es t se ponga en e, que es el tipo de anillo mas comun
#%%
df_datos['ring-type'] = np.where(df_datos['ring-type'].isna() & (df_datos['has-ring'] == 'f'), 'f', df_datos['ring-type'])
df_datos['ring-type'] = np.where(df_datos['ring-type'].isna() & (df_datos['has-ring'] == 't'), 'e', df_datos['ring-type'])

#%% Comprobacion
tieneanillo = df_datos.groupby('ring-type',  dropna=False)['has-ring'].value_counts()
tieneanillo

#%%[markdown]
#### spore-print-color
df_datos['spore-print-color'].value_counts(dropna=False)
#%%
df_datos = df_datos.drop(columns=['spore-print-color'])
#%%[markdown]
#### habitat
df_datos['habitat'].value_counts(dropna=False)
#%%
df_datos['habitat'] = np.where(df_datos['habitat'].isna(), 'd', df_datos['habitat'])
#%%[markdown]
#### season
df_datos['season'].value_counts(dropna=False)
#%%
df_datos['season'] = df_datos['season'].fillna(df_datos['season'].mode()[0])

#%%
df_datos.columns
#%%[markdown]
# Ahora pasaremos a dummies esta columna
#ring_type_dummies = pd.get_dummies(df_datos['ring-type'], prefix='ring_type', drop_first=False)
# %% analisis nulos 'veil-color'

resultado = df_datos.groupby('veil-type',  dropna=False)['class'].value_counts()
resultado
resultado.plot(kind='bar', stacked=True)
plt.title('Distribución de Stem Root por Edibilidad')
plt.xlabel('Stem Root')
plt.ylabel('Frecuencia')
plt.legend(title='Edibilidad', labels=['Edible', 'Poisonous'])
plt.show()


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





 













