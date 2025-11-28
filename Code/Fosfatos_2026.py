import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime

# Cargar los datos
Base = pd.read_excel('C:/Users/Angel/Desktop/UAM/4o/TFG/Datos/Base final.xlsx', parse_dates=['Fecha'])
Base.set_index('Fecha', inplace=True)
print(Base)

# Función para estructurar los datos en series de tiempo supervisadas
def create_supervised_data(data, n_lags=12):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, n_lags + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.dropna(inplace=True)
    df.columns = [f'lag_{i}' for i in range(n_lags, 0, -1)] + ['target']
    return df

# Preparar los datos
n_lags = 12
df_supervised = create_supervised_data(Base['Fosfatos'].values, n_lags)
X, y = df_supervised.iloc[:, :-1].values, df_supervised.iloc[:, -1].values

# Configuración del modelo optimizado y de la validación temporal
model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, max_features='log2', bootstrap=False)
tscv = TimeSeriesSplit(n_splits=245)
predictions = []
true_values = []
dates = []

# Entrenamiento y predicción fuera de muestra (out-of-sample)
for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Entrenar el modelo y mostrar mensaje de depuración
    model.fit(X_train, y_train)
    
    # Hacer predicciones y almacenar
    y_pred = model.predict(X_test)
    predictions.extend(y_pred)
    true_values.extend(y_test)
    
    # Guardar las fechas correspondientes para graficarlas
    test_dates = Base.index[n_lags:][test_index]
    dates.extend(test_dates)

# Calcular el error
mse = mean_squared_error(true_values, predictions)
print(f'Mean Squared Error: {mse}')

# Crear predicciones futuras hasta mediados de 2025
last_window = df_supervised.iloc[-n_lags:, :-1].values
future_predictions = []

for i in range(34):  # Predicciones para 18 meses hasta mediados de 2025
    pred = model.predict(last_window)  # Realizar la predicción
    future_predictions.append(pred[0])  # Almacenar la predicción
    
    # Actualizar la ventana deslizante para la próxima predicción
    last_window = np.column_stack([last_window[:, 1:], pred])

# Crear las fechas correspondientes a las predicciones futuras
fechas_predicciones = pd.date_range(start=Base.index.max(), periods=34, freq='ME')

# Imprimir las fechas y los valores de las predicciones futuras
for fecha, pred in zip(fechas_predicciones, future_predictions):
    print(f"Fecha: {fecha.date()}, Predicción de Fosfatos: {pred}")

# Visualización de los resultados
plt.figure(figsize=(14, 7))

# Graficar los valores reales y las predicciones para el periodo histórico
plt.plot(dates, true_values, label='Valores Reales de Fosfatos', color='purple')
plt.plot(dates, predictions, label='Predicciones del Modelo (historicas)', color='gold')

# Graficar las predicciones futuras
plt.plot(fechas_predicciones, future_predictions, label='Predicciones Futuras de Fosfatos (hasta finales de 2026)', color='red')

# Añadir el umbral de 12.9
plt.axhline(y=0.072, color='blue', linestyle='--', label='Umbral de Fosfatos = 0.072 mg/L')


# Convertir la fecha a un objeto datetime
fecha_inicio_predicciones = datetime.strptime('30-03-2024', '%d-%m-%Y')

# Añadir la línea vertical en la fecha correcta
plt.axvline(x=fecha_inicio_predicciones, color='purple', linestyle='--', linewidth=2, label='Inicio Predicciones')


# Configuración de la gráfica
plt.xlabel('Fecha')
plt.ylabel('Concentración de Fosfatos')
plt.title('Predicciones de Fosfatos: Históricas y Futuras hasta finales de 2026')
plt.legend()
plt.grid(True)
plt.show()
