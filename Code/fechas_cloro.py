import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Crear un DataFrame con los datos de clorofila históricos
data = {
    'Año': [2019, 2020, 2021, 2022, 2023, 2024],  # Incluir el año 2024 para Primavera
    'Invierno (31-01)': [None, 6.769, 2.470637, 0.773392, 0.840360, 0.313],  # No cambiar Invierno 2024
    'Primavera (30-04)': [None, 2.334, 0.600162, 1.279426, 0.338282, 0.6365],  # Establecer un valor para Primavera 2024
    'Verano (31-07)': [None, 2.115, 1.581151, 0.721205, 0.368148, None],  # No cambiar Verano 2024
    'Otoño (31-10)': [17.701, 1.132, 2.348362, 1.197043, 1.182232, None]  # No cambiar Otoño 2024
}

df = pd.DataFrame(data)

# Datos de predicción de clorofila
predicciones = {
    'Fecha': [
        '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31', '2024-09-30', '2024-10-31', '2024-11-30',
        '2024-12-31', '2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30', '2025-05-31', '2025-06-30',
        '2025-07-31', '2025-08-31', '2025-09-30', '2025-10-31', '2025-11-30', '2025-12-31', '2026-01-31',
        '2026-02-28', '2026-03-31', '2026-04-30', '2026-05-31', '2026-06-30', '2026-07-31', '2026-08-31',
        '2026-09-30', '2026-10-31', '2026-11-30', '2026-12-31'
    ],
    'Predicción de Clorofila': [
        0.712, 0.846, 0.833, 0.746, 0.773, 1.215, 1.299, 1.057, 0.878, 0.774, 0.751, 0.725,0.888,
        0.974, 0.897, 0.975, 1.161, 1.355, 1.283, 1.118, 0.938, 0.738, 0.795, 0.902, 1.024, 1.059,
        1.069, 1.154, 1.225, 1.350, 1.260, 1.083
    ]
}

# Convertir las fechas de predicción a formato datetime
predicciones_df = pd.DataFrame(predicciones)
predicciones_df['Fecha'] = pd.to_datetime(predicciones_df['Fecha'])

# Crear listas para almacenar los valores secuenciales
valores = []
etiquetas = []
colores = []

# Definir colores para cada estación en los datos históricos
color_estaciones = {
    'Invierno (31-01)': 'blue',
    'Primavera (30-04)': 'blue',
    'Verano (31-07)': 'blue',
    'Otoño (31-10)': 'blue'
}

# Añadir los datos históricos al gráfico
for index, row in df.iterrows():
    for estacion in ['Invierno (31-01)', 'Primavera (30-04)', 'Verano (31-07)', 'Otoño (31-10)']:
        # Verificar si el valor es finito
        if pd.notna(row[estacion]) and np.isfinite(row[estacion]):  # Solo agregar valores finitos
            valores.append(row[estacion])
            # Cambiar el formato de la etiqueta a día-mes-año
            fecha_etiqueta = f"{estacion.split()[0]} {int(row['Año'])}"
            etiquetas.append(fecha_etiqueta)
            
            # Asignar color verde a los datos históricos (no futuros)
            color = color_estaciones[estacion]
                
            colores.append(color)

# Añadir los datos de predicción al gráfico
for i, row in predicciones_df.iterrows():
    valores.append(row['Predicción de Clorofila'])
    etiquetas.append(row['Fecha'].strftime('%d-%m-%Y'))  # Convertir la fecha al formato dd-mm-aaaa
    colores.append('red')  # Color para las predicciones (naranja)

# Datos reales de clorofila (ya procesados)
fechas_reales = ['31-05-2024', '30-06-2024', '31-07-2024']
valores_reales = [0.333, 0.499, 0.450]

# Añadir los datos reales de clorofila al gráfico
for i, valor in enumerate(valores_reales):
    # Buscar la fecha de la predicción más cercana para colocar el valor real en la posición correcta
    fecha_real = fechas_reales[i]
    indice_fecha = predicciones_df[predicciones_df['Fecha'].dt.strftime('%d-%m-%Y') == fecha_real].index[0]
    
    valores.append(valor)
    etiquetas.append(predicciones_df.loc[indice_fecha, 'Fecha'].strftime('%d-%m-%Y'))  # Usar la fecha de la predicción
    colores.append('blue')  # Color verde para los valores reales (no futuros)

# Crear el gráfico
plt.figure(figsize=(14, 7))

# Graficar los puntos de todos los datos
plt.scatter(etiquetas, valores, c=colores, marker='o', s=100, alpha=0.9)

# Graficar las líneas de los puntos
# Línea de los datos históricos (verdes)
historicos = [v for c, v in zip(colores, valores) if c == 'blue']
etiquetas_historicos = [e for c, e in zip(colores, etiquetas) if c == 'blue']
plt.plot(etiquetas_historicos, historicos, linestyle='-', color='blue', label='Datos Históricos')

# Línea de las predicciones (naranjas)
predicciones = [v for c, v in zip(colores, valores) if c == 'red']
etiquetas_predicciones = [e for c, e in zip(colores, etiquetas) if c == 'red']
plt.plot(etiquetas_predicciones, predicciones, linestyle='-', color='red', label='Predicciones Futuras')

# Añadir una línea horizontal en y=0.08 (como referencia)
plt.axhline(y=0.08, color='green', linestyle='-', linewidth=2, label='Umbral clorofila = 0.9 ug / L')


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

# Personalizar el gráfico
plt.title('Concentración de Clorofila por Año y Estación', fontsize=16, fontweight='bold')
plt.xlabel('Izq. de la linea violeta, fechas historicas estacionales - Dch. de la línea violeta, Fechas historicas e futuras mensuales', fontsize=12)
plt.ylabel('Concentración de Clorofila (ug / L)', fontsize=12)

# Hacer los ejes X más legibles
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)

# Añadir fondo animado
plt.gca().set_facecolor('#f7f7f7')

# Añadir una cuadrícula suave
plt.grid(True, linestyle='--', alpha=0.9)

# Añadir la línea de la leyenda
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.show()
