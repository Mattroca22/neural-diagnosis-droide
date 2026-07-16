import torch
from model_multimodal import RedMultimodalMedica

# 1. Instanciar nuestro modelo multimodal recién creado
modelo = RedMultimodalMedica(num_clases=3)
modelo.eval() # Modo de evaluación para desactivar capas de Dropout de entrenamiento

# 2. Simular un lote (Batch) de 4 pacientes artificiales simultáneos
# Cada paciente ingresa con sus 3 exámenes correspondientes al mismo tiempo:
print("--- Generando datos sintéticos de prueba para 4 pacientes ---")

# Examen 1: Imágenes de Rayos X (Formato: [Pacientes, Canales RGB, Alto, Ancho])
# El codificador espera imágenes cuadradas estándar de 3 canales y 224x224 píxeles
imagenes_fake = torch.randn(4, 3, 224, 224) 

# Examen 2: Valores de laboratorio de Sangre (Formato: [Pacientes, Variables numéricas])
# Pasamos las 3 variables clave elegidas (Glóbulos Blancos, PCR, Neutrófilos)
sangre_fake = torch.randn(4, 3) 

# Examen 3: Embeddings de Texto Clínico (Formato: [Pacientes, Características de BERT])
# Simula el vector denso que entregará ClinicalBERT tras procesar los reportes escritos
texto_fake = torch.randn(4, 768) 

print(f"Dimensión del lote de Rayos X: {imagenes_fake.shape}")
print(f"Dimensión del lote de Laboratorio: {sangre_fake.shape}")
print(f"Dimensión del lote de Reportes de texto: {texto_fake.shape}\n")

# 3. Pasar los exámenes simulados a través de la Red Neuronal
print("--- Procesando exámenes a través del modelo multimodal ---")
with torch.no_grad(): # Desactiva el cálculo de gradientes para ahorrar memoria ram
    veredicto_final = modelo(imagenes_fake, sangre_fake, texto_fake)

# 4. Mostrar el resultado de salida de la inteligencia artificial
print(f"Dimensión de la predicción de salida: {veredicto_final.shape}")
print("\nMatriz de predicción bruta (Logits) obtenida por paciente:")
print(veredicto_final)

# Obtener la clase con la probabilidad más alta para cada uno de los 4 pacientes
diagnosticos_id = torch.argmax(veredicto_final, dim=1)
mapeo_clases = {0: "Neumonía Bacteriana", 1: "Derrame Pleural", 2: "Paciente Sano"}

print("\nVeredicto diagnóstico simulado para cada paciente:")
for i, idx in enumerate(diagnosticos_id):
    print(f" -> Paciente {i+1}: Diagnóstico sugerido -> {mapeo_clases[idx.item()]}")

