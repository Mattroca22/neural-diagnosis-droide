import os
import random
import numpy as np
import pandas as pd
from PIL import Image

print("--- Paso 12: Generando Datos Clínicos con Ruido y Solapamiento Real ---")

os.makedirs("data_hospital/imagenes", exist_ok=True)

# Vocabulario médico realista y desordenado para mezclar de forma aleatoria
sintomas_respiratorios = ["tos seca", "tos con flema", "expectoracion verdosa", "disnea ligera", "falta de aire al caminar"]
signos_graves = ["fiebre de 39C", "escalofrios nocturnos", "dolor punzante en el costado", "saturacion de oxigeno al 88%"]
notas_rutina = ["paciente asintomatico", "asiste por chequeo general", "adecuado estado de salud", "pulmones limpios"]

registros = []

print("Generando 1000 expedientes médicos con variables cruzadas y ruido...")
for i in range(1, 1001):
    clase = random.choice([0, 1, 2]) # 0: Neumonía, 1: Derrame, 2: Sano
    
    # Añadimos un 15% de probabilidad de que el paciente sea un "Caso Atípico" (Ruido biológico)
    es_caso_atipico = random.random() < 0.15

    if clase == 0:  # NEUMONÍA
        # Si es atípico, sus laboratorios se ven normales a pesar de estar enfermo
        blancos = random.uniform(5000.0, 9500.0) if es_caso_atipico else random.uniform(10500.0, 18500.0)
        pcr = random.uniform(2.0, 15.0) if es_caso_atipico else random.uniform(35.0, 120.0)
        neutrofilos = random.uniform(50.0, 65.0) if es_caso_atipico else random.uniform(72.0, 88.0)
        
        texto = f"Paciente ingresa. Presenta {random.choice(sintomas_respiratorios)} y {random.choice(signos_graves)}. Alerta por posible infeccion."
        brillo_base = random.randint(150, 220)

    elif clase == 1:  # DERRAME PLEURAL
        blancos = random.uniform(4500.0, 8500.0) if es_caso_atipico else random.uniform(8500.0, 13500.0)
        pcr = random.uniform(1.0, 10.0) if es_caso_atipico else random.uniform(15.0, 55.0)
        neutrofilos = random.uniform(48.0, 62.0) if es_caso_atipico else random.uniform(66.0, 76.0)
        
        texto = f"Refiere {random.choice(sintomas_respiratorios)}. Reporta dolor toracico. Murmullo vesicular disminuido. Sugiere liquido pleuritico."
        brillo_base = random.randint(120, 170)

    else:  # PACIENTE SANO
        # Si es atípico, el paciente sano tiene los laboratorios alterados por otra causa ajena
        blancos = random.uniform(11000.0, 14500.0) if es_caso_atipico else random.uniform(4500.0, 9800.0)
        pcr = random.uniform(20.0, 45.0) if es_caso_atipico else random.uniform(0.1, 4.5)
        neutrofilos = random.uniform(70.0, 78.0) if es_caso_atipico else random.uniform(45.0, 68.0)
        
        texto = f"Control medico. {random.choice(notas_rutina)}. Signos estables. Niega sintomatologia respiratoria aguda."
        brillo_base = random.randint(70, 120)

    # --- INYECCIÓN DE RUIDO VISUAL ---
    # Creamos una matriz base y le sumamos ruido aleatorio píxel por píxel (simulando artefactos mecánicos del Rayos X)
    matriz_pizeles = np.full((250, 250), brillo_base, dtype=np.uint8)
    ruido_gaussiano = np.random.normal(0, 15, (250, 250)).astype(np.int16)
    matriz_ruidosa = np.clip(matriz_pizeles + ruido_gaussiano, 0, 255).astype(np.uint8)
    
    ruta_img = f"data_hospital/imagenes/paciente_{i}.jpg"
    img = Image.fromarray(matriz_ruidosa)
    img.save(ruta_img)

    registros.append({
        'ID_Paciente': i,
        'Glob_Blancos_uL': blancos,
        'PCR_mg_L': pcr,
        'Neutrofilos_porc': neutrofilos,
        'Nota_Clinica': texto,
        'Ruta_Imagen': ruta_img,
        'Diagnostico_Real': clase
    })

df_total = pd.DataFrame(registros)
df_total.to_csv("data_hospital/index_clinico.csv", index=False)
print("\n✔ ¡Base de datos ruidosa y compleja sobrescrita con éxito!")
