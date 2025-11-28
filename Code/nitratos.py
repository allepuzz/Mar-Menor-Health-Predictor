import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

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

# Extraer el día, mes y año de la fecha
Base['Dia'] = Base['Fecha'].dt.day
Base['Mes'] = Base['Fecha'].dt.month
Base['Año'] = Base['Fecha'].dt.year

# Variables numéricas, incluyendo Día, Mes y Año
numeric_cols = ['Turbidez', 'Temperatura', 'Salinidad','Conductividad', 'Caudal', 'Oxigeno', 'Clorofila', 'Nitratos', 'Fosfatos', 'Dia', 'Mes', 'Año']

# Análisis de correlación
plt.figure(figsize=(12, 8))
correlation_matrix = Base[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación')
plt.savefig('./matriz_correlacion_nitratos.svg', format='svg')
plt.show()

# Análisis de Tendencias y Estacionalidad
# Asegurarse de que la variable Clorofila esté en un índice temporal
Base.set_index('Fecha', inplace=True)

# Descomposición de la serie temporal
decomposition = seasonal_decompose(Base['Nitratos'], model='additive', period=30)  # Ajusta el período según tus datos

# Graficar los componentes
plt.figure(figsize=(15, 10))
decomposition.plot()
plt.suptitle('Descomposición de la serie temporal de Nitratos', fontsize=16)
plt.savefig('./descomposicion_nitratos.svg', format='svg')
plt.show()

# Variables para predicción
X = Base.drop(['Nitratos'], axis=1)  # Incluir Día, Mes, Año
y = Base['Nitratos']
fecha = Base.index  # Usa el índice como fecha

# Configuración de la búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'bootstrap': [False]
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

# Inicializar listas para almacenar los errores y puntuaciones en cada split
mse_scores = []
real = []
predict = []
fechasall = []
primeraFecha = []
feature_importances = []

train_mse_scores = []
test_mse_scores = []
train_mae_scores = []
test_mae_scores = []
train_r2_scores = []
test_r2_scores = []

# Validación cruzada de series temporales
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    fechas_test = fecha[test_index]
    
    RF_optimized.fit(X_train, y_train)
    predictions = RF_optimized.predict(X_test)
    
    mse = root_mean_squared_error(y_test, predictions)
    mse_scores.append(mse)
    
    real.extend(y_test)
    predict.extend(predictions)
    fechasall.extend(fechas_test)
    primeraFecha.append(fechas_test[0])
    
    feature_importances.append(RF_optimized.feature_importances_)

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

print(f"MSE por split para Nitratos: {mse_scores}")
print(f"MSE medio para Nitratos: {np.mean(mse_scores)}")

# Crear un DataFrame para visualizar los resultados
results = pd.DataFrame({'Fecha': fechasall, 
                        'Nitratos_Real': real, 'Nitratos_Predicho': predict})

# Definir umbrales
umbral = 12.9

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
plot_results('Nitratos', 'Nitratos_Real', 'Nitratos_Predicho', umbral, 'green', 'red', 'violet')


plt.title('Comparación de Valores Reales vs. Predichos de Nitratos')
plt.xlabel('Fecha')
plt.ylabel('Nitratos')
plt.legend()
plt.grid(True)
plt.savefig('./Nitratos_prediccion.svg', format='svg')
plt.show()

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

