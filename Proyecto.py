
#%%
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#%% Cargar el archivo CSV para su análisis
file_path = 'Tema_14.csv'
data = pd.read_csv(file_path)

#%% Mostrar las primeras filas para una vista preliminar
data.head()

#%% Resumen de valores nulos y tipos de datos para cada columna
missing_values = data.isnull().sum()
data_types = data.dtypes

# Descripción estadística inicial para entender el rango y distribución de valores numéricos
description = data.describe()

missing_values, data_types, description

#%%# 1. Eliminar columnas con más del 50% de datos faltantes, ya que será difícil imputarlas adecuadamente
threshold = 0.5 * len(data)
data_cleaned = data.dropna(thresh=threshold, axis=1)

# 2. Imputar valores nulos restantes para columnas categóricas (rellenar con la moda)
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
data_cleaned[categorical_cols] = imputer_cat.fit_transform(data_cleaned[categorical_cols])

# 3. Imputar valores nulos restantes para columnas numéricas (rellenar con la mediana)
numerical_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
imputer_num = SimpleImputer(strategy='median')
data_cleaned[numerical_cols] = imputer_num.fit_transform(data_cleaned[numerical_cols])

# 4. Convertir variables categóricas a variables numéricas con Label Encoding
encoder = LabelEncoder()
for col in categorical_cols:
    data_cleaned[col] = encoder.fit_transform(data_cleaned[col])

# Verificar el resultado del pretratamiento
data_cleaned.info(), data_cleaned.head()

#%%