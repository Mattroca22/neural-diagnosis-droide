import torch
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel
from PIL import Image, ImageOps
import numpy as np
from model_multimodal import RedMultimodalMedica

print("--- Paso 13: Módulo de Inferencia Médica en Tiempo Real ---")

# 1. Configuración de Hardware y Carga de Modelos
modelo_texto_base = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
modelo_texto_base.eval()

tokenizador = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Instanciar el cerebro multimodal y cargar los pesos entrenados del disco
modelo_ia = RedMultimodalMedica(num_clases=3)
modelo_ia.load_state_dict(torch.load("cerebro_medico_multimodal.pth"))
modelo_ia.eval()
print("✔ Modelos y pesos de la IA cargados correctamente desde el almacenamiento.")

# 2. SIMULACIÓN DE INGRESO DE UN NUEVO PACIENTE A URGENCIAS
print("\n--- CASO CLÍNICO ENTRANTE ---")

# Examen A: Datos de Sangre Manuales
# Valores ingresados: [Glóbulos Blancos, PCR, Neutrófilos]
# Nota: Para esta inferencia simple usamos valores crudos aproximados. En producción
# se usaría el promedio del StandardScaler del archivo original.
sangre_paciente = torch.tensor([[17200.0, 95.4, 84.0]], dtype=torch.float32)

# Examen B: Nota Clínica redactada por el médico de guardia
nota_medica_paciente = ["Paciente masculino presenta tos con expectoracion verdosa, picos febriles y disnea leve."]

# Examen C: Placa de Rayos X (Cargamos la imagen temporal ruidosa generada previamente)
ruta_imagen_paciente = "data_hospital/imagenes/paciente_1.jpg"

print(f" -> Analítica de Sangre: Blancos={sangre_paciente[0][0].item()}, PCR={sangre_paciente[0][1].item()}")
print(f" -> Nota de Admisión: '{nota_medica_paciente[0]}'")
print(f" -> Archivo de Imagen: {ruta_imagen_paciente}")

# 3. Preprocesamiento en Tiempo Real
transformador_imagen = T.Compose([
    T.Resize((224, 224)),
    T.Lambda(lambda img: img.convert("RGB")),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(ruta_imagen_paciente)
tensor_img = transformador_imagen(img).unsqueeze(0) # Agregar dimensión de lote [1, 3, 224, 224]

# Tokenizar el texto del nuevo paciente
tokens = tokenizador(nota_medica_paciente, padding=True, truncation=True, max_length=32, return_tensors="pt")

# 4. Procesar y extraer Embeddings de Texto
with torch.no_grad():
    salida_bert = modelo_texto_base(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
    embeddings_texto = salida_bert.last_hidden_state[:, 0, :]

# 5. Ejecutar la Inferencia Multimodal (Veredicto de la IA)
print("\n--- Procesando Diagnóstico Inteligente ---")
with torch.no_grad():
    logits_salida = modelo_ia(tensor_img, sangre_paciente, embeddings_texto)
    # Convertir las salidas brutas en probabilidades porcentuales usando Softmax
    probabilidades = torch.softmax(logits_salida, dim=1).flatten()

mapeo_clases = {0: "Neumonía Bacteriana", 1: "Derrame Pleural", 2: "Paciente Sano"}
clase_ganadora = torch.argmax(probabilidades).item()

print("\n📈 RESULTADOS DE PROBABILIDAD CLÍNICA:")
for id_clase, nombre in mapeo_clases.items():
    print(f" -> {nombre}: {probabilidades[id_clase].item() * 100:.2f}%")

print(f"\n🚨 VEREDICTO FINAL DE LA RED NEURONAL: {mapeo_clases[clase_ganadora].upper()}")
