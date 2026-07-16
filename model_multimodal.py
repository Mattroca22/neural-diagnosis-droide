import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

class RedMultimodalMedica(nn.Module):
    def __init__(self, num_clases=3):
        super(RedMultimodalMedica, self).__init__()
        
        # 1. Encoder de Imagen: Cargamos una ResNet18 preentrenada
        # Modificamos la primera capa si las imágenes son en escala de grises (1 canal)
        self.encoder_imagen = models.resnet18(pretrained=True)
        self.encoder_imagen.fc = nn.Linear(self.encoder_imagen.fc.in_features, 128)
        
        # 2. Encoder de Sangre: Red densa simple para variables numéricas (ej. 3 valores bioquímicos)
        self.encoder_sangre = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # 3. Encoder de Texto: Espacio reservado para conectar ClinicalBERT (salida típica de 768 dimensiones)
        # Para el esqueleto, reduciremos esa dimensión a una capa lineal compacta
        self.proyeccion_texto = nn.Linear(768, 128)
        
        # 4. Capa de Fusión: Recibe 128 (imagen) + 64 (sangre) + 128 (texto) = 320 características
        self.capa_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 5. Capa de Salida: Clasificación final (Neumonía, Derrame Pleural, Sano)
        self.clasificador_final = nn.Linear(128, num_clases)

    def forward(self, x_img, x_sangre, x_text_embeddings):
        # Extraer características individuales
        feat_img = self.encoder_imagen(x_img)          # Salida: [batch_size, 128]
        feat_sangre = self.encoder_sangre(x_sangre)    # Salida: [batch_size, 64]
        feat_texto = self.proyeccion_texto(x_text_embeddings) # Salida: [batch_size, 128]
        
        # Concatenar todos los vectores en uno solo
        vector_multimodal = torch.cat((feat_img, feat_sangre, feat_texto), dim=1)
        
        # Procesar en la capa de fusión y clasificar
        x_fusionada = self.capa_fusion(vector_multimodal)
        veredicto = self.clasificador_final(x_fusionada)
        
        return veredicto

# Instanciar el modelo para verificar que compila correctamente
modelo = RedMultimodalMedica(num_clases=3)
print(modelo)
