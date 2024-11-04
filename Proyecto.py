import pandas as pd

# Cargar el archivo CSV
file_path = 'Tema_14.csv'
data = pd.read_csv(file_path)

# Preprocesamiento: Revisar valores nulos
missing_values = data.isnull().sum()
print("Nulos por columna:\n", missing_values)

# Eliminar filas solo si tienen valores nulos en ciertas columnas
columnas_a_revisar = ['class']
df_sin_nulos_columnas = data.dropna(subset=columnas_a_revisar).copy()  # Asegura que es una copia del DataFrame original

# Verificar los nulos restantes

#print("Nulos después de eliminar filas en columnas específicas:\n", df_sin_nulos_columnas.isnull().sum())

# Contar la cantidad de cada categoría en cada columna y mostrarla
for column in ['cap-shape', 'cap-surface', 'spore-print-color']:
    print(f"Cantidad de cada categoría en la columna {column}:")
    print(data[column].value_counts())
    print("\n")

# Imputación de valores nulos usando la moda para columnas categóricas
data['cap-shape'] = data['cap-shape'].fillna(data['cap-shape'].mode()[0])
data['cap-surface'] = data['cap-surface'].fillna(data['cap-surface'].mode()[0])


# Verificar nulos después de la imputación
print("Nulos después de la imputación:\n", data.isnull().sum())

# Calcular y mostrar los porcentajes de "poisonous" y "edible" en las categorías seleccionadas
for column in ['cap-shape', 'cap-surface', 'spore-print-color']:
    print(f"Porcentajes de 'poisonous' y 'edible' en la columna {column}:")
    porcentaje_clase = data.groupby([column, 'class']).size().unstack(fill_value=0)
    porcentaje_clase = porcentaje_clase.div(porcentaje_clase.sum(axis=1), axis=0) * 100
    print(porcentaje_clase)
    print("\n")

# Crear una tabla de solo dummies (sin combinarlas con el DataFrame principal)
dummies_table = pd.get_dummies(data[['cap-shape', 'cap-surface', 'spore-print-color']], 
                               columns=['cap-shape', 'cap-surface', 'spore-print-color'], 
                               prefix=['cap_shape', 'cap_surface', 'spore_print_color'])

# Mostrar el resultado de las primeras filas de la tabla de solo dummies
print("Tabla de solo dummies:\n", dummies_table.head())
