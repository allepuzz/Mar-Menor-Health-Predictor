import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Crear un DataFrame con los datos de fosfatos históricos
data = {
    'Año': [2019, 2020, 2021, 2022, 2023, 2024],  # Incluir el año 2024 para Primavera
    'Invierno (31-01)': [None, None, None, 0.300, 0.328, 0.283],  # No cambiar Invierno 2024
    'Primavera (30-04)': [None, None, None, 0.300, 0.119, 0.182],  # Establecer un valor para Primavera 2024
    'Verano (31-07)': [None, None, None, 2.275, 0.443, None],  # No cambiar Verano 2024
    'Otoño (31-10)': [None, None, 0.411, 0.442, 0.199, None]  # No cambiar Otoño 2024
}

df = pd.DataFrame(data)

# Datos de predicción de fosfatos
predicciones = {
    'Fecha': [
        '2024-05-31', '2024-06-30', '2024-07-31', '2024-08-31', '2024-09-30', '2024-10-31', '2024-11-30',
        '2024-12-31', '2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30', '2025-05-31', '2025-06-30',
        '2025-07-31', '2025-08-31', '2025-09-30', '2025-10-31', '2025-11-30', '2025-12-31', '2026-01-31',
        '2026-02-28', '2026-03-31', '2026-04-30', '2026-05-31', '2026-06-30', '2026-07-31', '2026-08-31',
        '2026-09-30', '2026-10-31', '2026-11-30', '2026-12-31'
    ],
    'Predicción de Fosfatos': [
        0.235, 0.217, 0.167, 0.186, 0.207, 0.198, 0.198, 0.231, 0.210, 0.215, 0.241, 0.179, 
        0.224, 0.189, 0.193, 0.198, 0.215, 0.200, 0.210, 0.218, 0.193, 0.210, 0.245, 0.188,
        0.184, 0.208, 0.195, 0.200, 0.193, 0.210, 0.195, 0.188
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
    'Invierno (31-01)': 'purple',
    'Primavera (30-04)': 'purple',
    'Verano (31-07)': 'purple',
    'Otoño (31-10)': 'purple'
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
    valores.append(row['Predicción de Fosfatos'])
    etiquetas.append(row['Fecha'].strftime('%d-%m-%Y'))  # Convertir la fecha al formato dd-mm-aaaa
    colores.append('gold')  # Color para las predicciones (naranja)

# Datos reales de fosfatos (ya procesados)
fechas_reales = ['31-05-2024', '30-06-2024', '31-07-2024', '31-08-2024', '30-09-2024', '31-10-2024']
valores_reales = [0.83, 0.68, 0.78, 0.73, 0.68, 1.49]

# Añadir los datos reales de fosfatos al gráfico
for i, valor in enumerate(valores_reales):
    # Buscar la fecha de la predicción más cercana para colocar el valor real en la posición correcta
    fecha_real = fechas_reales[i]
    indice_fecha = predicciones_df[predicciones_df['Fecha'].dt.strftime('%d-%m-%Y') == fecha_real].index[0]
    
    valores.append(valor)
    etiquetas.append(predicciones_df.loc[indice_fecha, 'Fecha'].strftime('%d-%m-%Y'))  # Usar la fecha de la predicción
    colores.append('purple')  # Color verde para los valores reales (no futuros)

# Crear el gráfico
plt.figure(figsize=(14, 7))

# Graficar los puntos de todos los datos
plt.scatter(etiquetas, valores, c=colores, marker='o', s=100, alpha=0.9)

# Graficar las líneas de los puntos
# Línea de los datos históricos (verdes)
historicos = [v for c, v in zip(colores, valores) if c == 'purple']
etiquetas_historicos = [e for c, e in zip(colores, etiquetas) if c == 'purple']
plt.plot(etiquetas_historicos, historicos, linestyle='-', color='purple', label='Datos Históricos')

# Línea de las predicciones (naranjas)
predicciones = [v for c, v in zip(colores, valores) if c == 'gold']
etiquetas_predicciones = [e for c, e in zip(colores, etiquetas) if c == 'gold']
plt.plot(etiquetas_predicciones, predicciones, linestyle='-', color='gold', label='Predicciones Futuras')

# Añadir una línea horizontal en y=0.08 (como referencia)
plt.axhline(y=0.08, color='blue', linestyle='-', linewidth=2, label='Umbral fosfatos = 0.072 mg / L')

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
plt.title('Concentración de Fosfatos por Año y Estación', fontsize=16, fontweight='bold')
plt.xlabel('Izq. de la linea violeta, fechas historicas estacionales - Dch. de la línea violeta, Fechas historicas e futuras mensuales', fontsize=12)
plt.ylabel('Concentración de Fosfatos (ug / L)', fontsize=12)

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
