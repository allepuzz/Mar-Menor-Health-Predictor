import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

# Cargar base de datos
Base = pd.read_excel('C:/Users/Angel/Desktop/UAM/4o/TFG/Datos/Base final.xlsx')
Base['Fecha'] = pd.to_datetime(Base['Fecha'])  # Convertir a datetime

# Reemplazar comas por puntos en columnas de texto y convertir a numérico
for col in Base.columns:
    if Base[col].dtype == 'object':
        Base[col] = Base[col].str.replace(',', '.')
numeric_cols = ['Turbidez', 'Conductividad', 'Temperatura', 'Salinidad', 'Oxigeno', 'Clorofila', 'Nitratos', 'Fosfatos']
Base[numeric_cols] = Base[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Eliminar filas con NaN
Base = Base.dropna()

# Extraer día, mes y año de la fecha
Base['Dia'] = Base['Fecha'].dt.day
Base['Mes'] = Base['Fecha'].dt.month
Base['Año'] = Base['Fecha'].dt.year

# Establecer la fecha como índice
Base.set_index('Fecha', inplace=True)

# Variable dependiente
y = Base['Nitratos']

# Descomposición de la serie temporal
decomposition = seasonal_decompose(Base['Nitratos'], model='additive', period=30)  # Ajusta el período según tus datos

# Graficar los componentes
plt.figure(figsize=(15, 10))
decomposition.plot()
plt.suptitle('Descomposición de la serie temporal de Nitratos', fontsize=16)
plt.savefig('./descomposicion_nitratos.svg', format='svg')
plt.show()

# Rango de valores para los parámetros
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
s = [12]  # Estacionalidad (mensual)

# Generar todas las combinaciones de parámetros
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], season) for x in itertools.product(P, D, Q) for season in s]

# Buscar el mejor modelo basado en el AIC
best_aic = float("inf")
best_params = None
best_seasonal_params = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            mod = SARIMAX(y, order=param, seasonal_order=seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_seasonal_params = seasonal_param
        except:
            continue

print(f"Mejores parámetros: {best_params} con parámetros estacionales: {best_seasonal_params} y AIC: {best_aic}")

# Entrenar el modelo SARIMA con los mejores parámetros
sarima_model = SARIMAX(y, order=best_params, seasonal_order=best_seasonal_params, enforce_stationarity=False, enforce_invertibility=False)
sarima_results = sarima_model.fit(disp=False)

# Predicción
y_pred = sarima_results.get_prediction(start=0, dynamic=False)
y_forecast = sarima_results.get_forecast(steps=30)
y_pred_ci = y_pred.conf_int()
y_forecast_ci = y_forecast.conf_int()

# Graficar los resultados de predicción
plt.figure(figsize=(15, 7))
plt.plot(y, label='Nitratos Real', color='green')
plt.plot(y_pred.predicted_mean, label='Nitratos Predicha', color='orange', linestyle='--')
plt.fill_between(y_pred_ci.index, y_pred_ci.iloc[:, 0], y_pred_ci.iloc[:, 1], color='violet', alpha=0.3)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gcf().autofmt_xdate()

plt.title('Comparación de Valores Reales vs. Predichos de Nitratos (SARIMA)')
plt.xlabel('Fecha')
plt.ylabel('Nitratos')
plt.legend()
plt.grid(True)
# Limitar el eje y 
plt.ylim(0, 250)
plt.show()

# Calcular métricas de rendimiento
y_true = y.values
y_pred_values = y_pred.predicted_mean.values
rmse = mean_squared_error(y_true, y_pred_values, squared=True)
mae = mean_absolute_error(y_true, y_pred_values)
r2 = r2_score(y_true, y_pred_values)

print(f"MSE para Nitratos (SARIMA): {rmse}")
print(f"MAE para Nitratos (SARIMA): {mae}")
print(f"R² para Nitratos (SARIMA): {r2}")

print(f"Mejores parámetros: {best_params} con parámetros estacionales: {best_seasonal_params} y AIC: {best_aic}")
