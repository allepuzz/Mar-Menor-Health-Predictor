import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
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
numeric_cols = ['Turbidez', 'Conductividad', 'Temperatura', 'Salinidad', 'Oxigeno', 'Clorofila', 'Nitratos', 'Fosfatos']
Base[numeric_cols] = Base[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores NaN resultantes de la conversión
Base = Base.dropna()

Base = Base[Base['Fosfatos'] > 0]
# Extraer el día, mes y año de la fecha
Base['Dia'] = Base['Fecha'].dt.day
Base['Mes'] = Base['Fecha'].dt.month
Base['Año'] = Base['Fecha'].dt.year

# Variables numéricas, incluyendo Día, Mes y Año
numeric_cols = ['Turbidez', 'Temperatura', 'Conductividad', 'Salinidad', 'Oxigeno', 'Clorofila', 'Nitratos', 'Fosfatos', 'Dia', 'Mes', 'Año']

# B. Análisis de correlación actualizado
plt.figure(figsize=(12, 8))
correlation_matrix = Base[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación')
plt.savefig('./matriz_correlacion.svg', format='svg')
plt.show()

# Análisis de Tendencias y Estacionalidad
Base.set_index('Fecha', inplace=True)
decomposition = seasonal_decompose(Base['Fosfatos'], model='additive', period=30)  # Ajusta el período según tus datos

plt.figure(figsize=(15, 10))
decomposition.plot()
plt.suptitle('Descomposición de la serie temporal de Fosfatos', fontsize=16)
plt.savefig('./descomposicion_fosfatos.svg', format='svg')
plt.show()

# Variables para predicción
X = Base.drop(['Fosfatos'], axis=1)  # Incluir Día, Mes, Año
y = Base['Fosfatos']
fecha = Base.index  # Usa el índice como fecha

# Definir el modelo de regresión lineal
model = LinearRegression()

# Validación cruzada de series temporales
tscv = TimeSeriesSplit(n_splits=135)

# Inicializar listas para almacenar los errores y puntuaciones en cada split
mse_scores = []
real = []
predict = []
fechasall = []
primeraFecha = []

train_mse_scores = []
test_mse_scores = []
train_mae_scores = []
test_mae_scores = []
train_r2_scores = []
test_r2_scores = []

# Realizar la validación cruzada de series temporales
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    fechas_test = fecha[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = root_mean_squared_error(y_test, predictions)
    mse_scores.append(mse)
    
    real.extend(y_test)
    predict.extend(predictions)
    fechasall.extend(fechas_test)
    primeraFecha.append(fechas_test[0])

    # Predecir en los conjuntos de entrenamiento y prueba
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular MSE, MAE y R² para ambos conjuntos
    train_mse = root_mean_squared_error(y_train, y_train_pred)
    test_mse = root_mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_mae_scores.append(train_mae)
    test_mae_scores.append(test_mae)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

print(f"MSE por split para Fosfatos: {mse_scores}")
print(f"MSE medio para Fosfatos: {np.mean(mse_scores)}")

# Crear un DataFrame para visualizar los resultados
results = pd.DataFrame({'Fecha': fechasall, 
                        'Fosfatos_Real': real, 'Fosfatos_Predicho': predict})

# Definir umbrales
umbral = 0.08

# Función para graficar
def plot_results(variable, real_label, pred_label, umbral, color_real, color_pred, umbral_color):
    plt.figure(figsize=(10, 6))
    plt.plot(results['Fecha'], results[real_label], label=f'{variable} Real', color=color_real, marker='o')
    plt.plot(results['Fecha'], results[pred_label], label=f'{variable} Predicho', color=color_pred, linestyle='--', marker='x')
    
    plt.axhline(y=umbral, color=umbral_color, linestyle='-', linewidth=2, label=f'Umbral {variable}')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  
    plt.gcf().autofmt_xdate()  

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
plot_results('Fosfatos', 'Fosfatos_Real', 'Fosfatos_Predicho', umbral, 'purple', 'gold', 'red')

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
