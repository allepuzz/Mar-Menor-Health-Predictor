import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Crear un DataFrame con los datos proporcionados
data = {
    'Año': [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    'Invierno (31-01)': [None, 6.769, 2.470637, 0.773392, 0.840360, 0.695789, 0.8261, 0.914],
    'Primavera (30-04)': [None, 2.334, 0.600162, 1.279426, 0.338282, 0.6365, 0.6913, 0.926],
    'Verano (31-07)': [None, 2.115, 1.581151, 0.721205, 0.368148, 0.7765, 0.9407, 1.233],
    'Otoño (31-10)': [17.701, 1.132, 2.348362, 1.197043, 1.182232, 1.167, 1.238, 1.208]  # Añadido valor de 2026
}

df = pd.DataFrame(data)

# Crear una lista para almacenar los valores secuenciales
valores = []
etiquetas = []
colores = []

# Definir colores para cada estación
color_estaciones = {
    'Invierno (31-01)': 'blue',
    'Primavera (30-04)': 'green',
    'Verano (31-07)': 'orange',
    'Otoño (31-10)': 'red'
}

# Crear una secuencia para cada año y cada estación
for index, row in df.iterrows():
    for estacion in ['Invierno (31-01)', 'Primavera (30-04)', 'Verano (31-07)', 'Otoño (31-10)']:
        # Verificar si el valor es finito
        if pd.notna(row[estacion]) and np.isfinite(row[estacion]):  # Solo agregar valores finitos
            valores.append(row[estacion])
            # Cambiar el formato de la etiqueta a día-mes-año
            fecha_etiqueta = f"{estacion.split()[0]} {int(row['Año'])}"
            etiquetas.append(fecha_etiqueta)
            
            # Definir colores, coloreando solo predicciones futuras a partir de Primavera 2024
            if row['Año'] >= 2024 and estacion == 'Primavera (30-04)':
                color = 'purple'  # Color para predicciones futuras
            else:
                color = color_estaciones[estacion]  # Color histórico para otros datos
                
            colores.append(color)

# Comprobación adicional si después de filtrar hay valores no finitos
if not all(np.isfinite(valores)):
    raise ValueError("Después de filtrar, hay valores no finitos (NaN o inf) en los datos de clorofila.")

# Crear el gráfico
plt.figure(figsize=(14, 7))

# Graficar la línea que conecta los puntos
plt.plot(etiquetas, valores, linestyle='-', marker='o', markersize=8, color='black', alpha=0.6)

# Mostrar los valores de clorofila en cada punto
for i, txt in enumerate(valores):
    plt.text(etiquetas[i], valores[i], f'{txt:.2f}', ha='center', va='bottom', fontsize=10)

# Añadir una línea horizontal en y=0.9
plt.axhline(y=0.9, color='gray', linestyle='--', linewidth=2, label='Clorofila = 0.9')

# Personalizar el gráfico
plt.title('Concentración de Clorofila (Chl-a) por Año y Estación', fontsize=16, fontweight='bold')
plt.xlabel('Fecha (Estación y Año)', fontsize=12)
plt.ylabel('Concentración de Clorofila (Chl-a)', fontsize=12)

# Hacer los ejes X más legibles
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)

# Añadir fondo animado
plt.gca().set_facecolor('#f7f7f7')

# Añadir una cuadrícula suave
plt.grid(True, linestyle='--', alpha=0.7)

# Identificar el índice de la primavera de 2024
fecha_predicciones = 'Primavera 2024'

# Buscar el índice del punto correspondiente (en este caso, Primavera 2024)
indice_fecha_predicciones = None
for i, etiqueta in enumerate(etiquetas):
    if 'Primavera' in etiqueta and '2024' in etiqueta:
        indice_fecha_predicciones = i
        break

# Añadir línea vertical en la fecha de inicio de las predicciones futuras
if indice_fecha_predicciones is not None:
    plt.axvline(x=indice_fecha_predicciones, color='purple', linestyle=':', linewidth=2, label="Inicio Predicciones Futuras")
else:
    print("No se encontró la fecha de inicio de las predicciones futuras.")

# Añadir la línea de la leyenda
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()
