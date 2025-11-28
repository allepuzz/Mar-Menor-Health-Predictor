import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
X = Base.drop(['Fecha', 'Nitratos'], axis=1)
y = Base['Nitratos']
fecha = Base['Fecha']

# Definir la configuración de la búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Grid Search para encontrar los mejores hiperparámetros
RF = RandomForestRegressor()
tscv = TimeSeriesSplit(n_splits=256)
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X, y)
best_params = grid_search.best_params_
print(f"Mejores hiperparámetros para Nitratos: {best_params}")

# Configuración del modelo RandomForest con los mejores hiperparámetros
RF_optimized = RandomForestRegressor(**best_params)

# Configuración de TimeSeriesSplit para validación cruzada temporal con 257 splits
tscv = TimeSeriesSplit(n_splits=257)

# Realizar la validación cruzada de series temporales
mse_scores = []
real = []
predict = []
fechasall = []
primeraFecha = []
feature_importances = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    fechas_test = fecha.iloc[test_index]
    
    RF_optimized.fit(X_train, y_train)
    predictions = RF_optimized.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mse_scores.append(mse)
    
    real.extend(y_test)
    predict.extend(predictions)
    fechasall.extend(fechas_test)
    primeraFecha.append(fechas_test.iloc[0])
    
    feature_importances.append(RF_optimized.feature_importances_)

print(f"MSE por split para Nitratos: {mse_scores}")
print(f"MSE medio para Nitratos: {np.mean(mse_scores)}")

# Crear un DataFrame para visualizar los resultados
results = pd.DataFrame({'Fecha': fechasall, 
                        'Nitratos_Real': real, 'Nitratos_Predicho': predict})

# Definir umbrales
umbral = 12.90

# Función para graficar
def plot_results(variable, real_label, pred_label, umbral, color_real, color_pred, umbral_color):
    plt.figure(figsize=(15, 7))
    plt.plot(results['Fecha'], results[real_label], label=f'{variable} Real', color=color_real, marker='o')
    plt.plot(results['Fecha'], results[pred_label], label=f'{variable} Predicho', color=color_pred, linestyle='--', marker='x')
    
    plt.axhline(y=umbral, color=umbral_color, linestyle='-', linewidth=2, label=f'Umbral {variable}')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  
    plt.gcf().autofmt_xdate()  

    # Agregar líneas verticales para cada fecha de inicio de split
    for fecha in primeraFecha:
        plt.axvline(x=fecha, color='grey', linestyle='--', linewidth=1)

    plt.title(f'Comparación de Valores Reales vs. Predichos de {variable}')
    plt.xlabel('Fecha')
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{variable}_prediccion.svg', format='svg')
    plt.show()

# Graficar resultados
plot_results('Nitratos', 'Nitratos_Real', 'Nitratos_Predicho', umbral, 'green', 'orange', 'magenta')

# Graficar importancias de características
feature_names = X.columns
plt.figure(figsize=(10, 6))
importances = np.mean(feature_importances, axis=0)
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importancia Relativa')
plt.title('Importancia de las Características para Nitratos')
plt.savefig('./Nitratos_importancia.svg', format='svg')
plt.show()
