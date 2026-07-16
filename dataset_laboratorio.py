import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("--- Paso 2 y 3: Procesamiento de Datos de Laboratorio ---")

# 1. Simulación de una base de datos hospitalaria real
datos_hospital = {
    'ID_Paciente': list(range(1, 5)),         # Genera los números del 1 al 4 automáticamente
    'Glob_Blancos_uL': [16500.0, 6200.0, 11200.0, 7100.0],  # Normal: 4500 - 11000
    'PCR_mg_L': [85.2, 2.1, 40.5, 1.8],       # Normal: < 5.0
    'Neutrofilos_porc': [82.5, 55.0, 71.0, 58.2] # Normal: 40% - 70%
}

df_laboratorios = pd.DataFrame(datos_hospital)
print("\n1. Tabla de datos crudos extraída del hospital (Pandas DataFrame):")
print(df_laboratorios)

# 2. El paso más importante en IA: NORMALIZACIÓN
escalador = StandardScaler()
columnas_medicas = ['Glob_Blancos_uL', 'PCR_mg_L', 'Neutrofilos_porc']
valores_normalizados = escalador.fit_transform(df_laboratorios[columnas_medicas])

print("\n2. Valores bioquímicos normalizados (Escala óptima para la IA):")
print(valores_normalizados)

# 3. Conversión final a Tensores de PyTorch
tensores_sangre = torch.tensor(valores_normalizados, dtype=torch.float32)
print("\n3. Tensores de PyTorch listos para conectar a la capa 'encoder_sangre':")
print(tensores_sangre)
print(f"Dimensión final del lote de laboratorio real: {tensores_sangre.shape}")
