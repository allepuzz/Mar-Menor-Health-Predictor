import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import griddata

Base = pd.read_csv("C:/Users/Angel/Desktop/UAM/4o/TFG/Datos/Base.csv", delimiter=';')
Base['Fecha'] = pd.to_datetime(Base['Fecha'])  # Convertir a datetime
print("Columnas del DataFrame:", Base.columns)


for col in Base.columns:
    if Base[col].dtype == 'object':  
        Base[col] = Base[col].str.replace(',', '.')  

numeric_cols = ['Turbidez', 'Temperatura', 'Salinidad', 'Oxigeno', 'Clorofila'] 
Base[numeric_cols] = Base[numeric_cols].apply(pd.to_numeric)

#print("Columnas convertidas y tipos de datos:")
#print(Base.dtypes)  # Mostrar los tipos de datos 

#Predecir Clorofila, porque tengo el umbral 
X = Base.drop(['Clorofila', 'Fecha'], axis=1)
y = Base['Clorofila']
fecha = Base['Fecha']

# Configuración del modelo RandomForest
RF = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True
)

# Configuración de TimeSeriesSplit para validación cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)  # Define el número de splits

# Realizar la validación cruzada de series temporales
mse_scores = []
real = []
predict = []
fechasall = []
primeraFecha = []


for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    fechas_test = fecha.iloc[test_index]
    
    RF.fit(X_train, y_train)  
    predictions = RF.predict(X_test)  
    mse = mean_squared_error(y_test, predictions)  
    mse_scores.append(mse)  

    real.extend(y_test)
    predict.extend(predictions)
    fechasall.extend(fechas_test)

    # Guardar la primera fecha del split actual
    primeraFecha.append(fechas_test.iloc[0]
)

# Mostrar las primeras fechas de cada split
for i, fechaaux in enumerate(primeraFecha, 1):
    print(f"Primera fecha del Split {i}: {fechaaux.strftime('%Y-%m-%d')}")

print("MSE por split: ", mse_scores)
print("MSE medio: ", np.mean(mse_scores))

# Crear un DataFrame para visualizar los resultados
results = pd.DataFrame({'Fecha': fechasall, 'Real': real, 'Predicho': predict})

#visualizar los splits
plt.figure(figsize=(10, 5))
plt.plot(mse_scores, marker='o', linestyle='-', color='b')
plt.title('MSE por Split de Validación Cruzada de Series Temporales')
plt.xlabel('Número de Split')
plt.ylabel('MSE')
plt.grid(True)
plt.xticks(range(len(mse_scores)), [f"Split {i+1}" for i in range(len(mse_scores))])  # Etiquetas personalizadas para cada split
plt.show()

# Grafica comparativa entre valores reales y predichos de la clorofila
plt.figure(figsize=(15, 7))
plt.plot(results['Fecha'], results['Real'], label='Real', color='blue', marker='o')
plt.plot(results['Fecha'], results['Predicho'], label='Predicho', color='red', linestyle='--', marker='x')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))  
plt.gcf().autofmt_xdate()  

# Agregar líneas verticales para cada fecha de inicio de split
for fecha in primeraFecha:
    plt.axvline(x=fecha, color='green', linestyle='--', linewidth=1)

plt.title('Comparación de Valores Reales vs. Predichos de Clorofila')
plt.xlabel('Fecha')
plt.ylabel('Clorofila')
plt.legend()
plt.grid(True)
plt.show()