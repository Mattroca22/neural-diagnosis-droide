import torch
import torchvision.transforms as T
from PIL import Image

print("--- Paso 4: Procesamiento de Imágenes de Rayos X ---")

# 1. Crear una imagen de prueba temporal en escala de grises
# Simulamos una radiografía de 300x300 píxeles para verificar el script
print("\n1. Creando una imagen médica de prueba en memoria...")
imagen_prueba = Image.new('L', (300, 300), color=128) # 'L' significa escala de grises
imagen_prueba.save("rayos_x_temporal.jpg")
print(" -> Imagen 'rayos_x_temporal.jpg' guardada con éxito en tu carpeta.")

# 2. Definir la tubería de transformación (Pipeline) para la IA
# Las imágenes médicas reales vienen en cualquier tamaño. La ResNet18 exige:
# - Que midan exactamente 224x224 píxeles.
# - Que tengan 3 canales (RGB), por lo que convertimos la escala de grises.
# - Que sus píxeles estén normalizados matemáticamente.
transformador = T.Compose([
    T.Resize((224, 224)),                    # Redimensionar la imagen
    T.Lambda(lambda img: img.convert("RGB")),# Forzar 3 canales para compatibilidad con ResNet
    T.ToTensor(),                            # Convertir a tensor (píxeles de 0.0 a 1.0)
    T.Normalize(                             # Normalización estándar de ImageNet
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Cargar la imagen del disco y aplicar las transformaciones
print("\n2. Cargando la imagen y aplicando el pipeline de Visión Artificial...")
img_cargada = Image.open("rayos_x_temporal.jpg")
tensor_imagen = transformador(img_cargada)

print("\n3. Tensor de imagen listo para inyectar al 'encoder_imagen':")
print(f" -> Tipo de dato: {tensor_imagen.type()}")
print(f" -> Forma del tensor individual [Canales, Alto, Ancho]: {tensor_imagen.shape}")

# 4. Simular la preparación de un lote (Batch)
# Agregamos una dimensión extra al inicio para representar al paciente
tensor_lote = tensor_imagen.unsqueeze(0)
print(f" -> Forma final del lote listo para la red [Pacientes, Canales, Alto, Ancho]: {tensor_lote.shape}")
