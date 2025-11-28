import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Cargar base de datos
Base = pd.read_excel('C:/Users/Angel/Desktop/UAM/4o/TFG/Datos/Base final.xlsx')
Base['Fecha'] = pd.to_datetime(Base['Fecha'])  # Convertir a datetime

# Reemplazar comas por puntos en todas las columnas que son strings
for col in Base.columns:
    if Base[col].dtype == 'object':
        Base[col] = Base[col].str.replace(',', '.')

# Convertir a numérico las columnas especificadas
numeric_cols = ['Turbidez', 'Temperatura', 'Salinidad', 'Oxigeno', 'Clorofila', 'Nitratos', 'Fosfatos']
Base[numeric_cols] = Base[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores NaN resultantes de la conversión
Base = Base.dropna()

# Variables para predicción
X = Base.drop(['Fecha', 'Clorofila'], axis=1)
y = Base['Clorofila']
fecha = Base['Fecha']

# Configuración del modelo RandomForest con los mejores hiperparámetros
RF_optimized = RandomForestRegressor(
    n_estimators=100, 
    max_depth=20, 
    min_samples_split=2, 
    min_samples_leaf=2, 
    max_features='log2', 
    bootstrap=True
)

# Configuración de TimeSeriesSplit para validación cruzada temporal con 200 splits
tscv = TimeSeriesSplit(n_splits=10)

# Inicializar listas para almacenar los errores y puntuaciones en cada split
train_mse_scores = []
test_mse_scores = []
train_mae_scores = []
test_mae_scores = []
train_r2_scores = []
test_r2_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Ajustar el modelo en los datos de entrenamiento
    RF_optimized.fit(X_train, y_train)
    
    # Predecir en los conjuntos de entrenamiento y prueba
    y_train_pred = RF_optimized.predict(X_train)
    y_test_pred = RF_optimized.predict(X_test)
    
    # Calcular MSE, MAE y R² para ambos conjuntos
    train_mse = root_mean_squared_error(y_train, y_train_pred)
    test_mse = root_mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Guardar los resultados
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_mae_scores.append(train_mae)
    test_mae_scores.append(test_mae)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

# Graficar las pérdidas (MSE) en entrenamiento y validación

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_mse_scores) + 1), train_mse_scores, label='Train', color='blue')
plt.plot(range(1, len(test_mse_scores) + 1), test_mse_scores, label='Test', color='red')
plt.title('Curva de Pérdidas (Train vs Test)')
plt.xlabel('Número de Splits')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Graficar MAE de entrenamiento y validación
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_mae_scores) + 1), train_mae_scores, label='MAE Entrenamiento', color='blue')
plt.plot(range(1, len(test_mae_scores) + 1), test_mae_scores, label='MAE Validación', color='red')
plt.title('MAE en Entrenamiento y Validación')
plt.xlabel('Número de Splits')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.show()

# Graficar R² de entrenamiento y validación
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_r2_scores) + 1), train_r2_scores, label='R² Entrenamiento', color='blue')
plt.plot(range(1, len(test_r2_scores) + 1), test_r2_scores, label='R² Validación', color='red')
plt.title('R² en Entrenamiento y Validación')
plt.xlabel('Número de Splits')
plt.ylabel('R² Score')
plt.legend()
plt.grid(True)
plt.show()

# Calcular y mostrar las métricas finales (promedio de todos los splits)
print(f"MSE medio en Entrenamiento: {np.mean(train_mse_scores)}")
print(f"MSE medio en Validación: {np.mean(test_mse_scores)}")
print(f"MAE medio en Entrenamiento: {np.mean(train_mae_scores)}")
print(f"MAE medio en Validación: {np.mean(test_mae_scores)}")
print(f"R² medio en Entrenamiento: {np.mean(train_r2_scores)}")
print(f"R² medio en Validación: {np.mean(test_r2_scores)}")
