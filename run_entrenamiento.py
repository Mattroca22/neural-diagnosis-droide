import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoModel
from model_multimodal import RedMultimodalMedica
from dataset_maestro import DatasetMedicoMultimodal
from sklearn.metrics import classification_report
from tqdm import tqdm


print("--- Paso 11: Bucle de Entrenamiento Científico con Datos Masivos ---")

# 1. Preparación y División de Datos (80% Entrenamiento, 20% Validación)
dataset_total = DatasetMedicoMultimodal()
tamano_train = int(0.8 * len(dataset_total))
tamano_val = len(dataset_total) - tamano_train

# random_split asegura que la división sea matemáticamente aleatoria
dataset_train, dataset_val = random_split(dataset_total, [tamano_train, tamano_val])

# Configuramos lotes de 16 pacientes para equilibrar el uso de memoria RAM/CPU
cargador_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
cargador_val = DataLoader(dataset_val, batch_size=16, shuffle=False)

print(f" -> Registros asignados para entrenamiento: {len(dataset_train)}")
print(f" -> Registros asignados para validación limpia: {len(dataset_val)}")

# 2. Inicialización de los Modelos
print("\nCargando extractores de características en memoria...")
modelo_texto_base = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
modelo_texto_base.eval()

modelo_ia = RedMultimodalMedica(num_clases=3)
criterio_error = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo_ia.parameters(), lr=0.0005) # Reducimos la tasa para un ajuste más fino

# 3. Bucle de Entrenamiento Real (3 Épocas es suficiente debido a la consistencia de los datos)
for epoca in range(3):
    modelo_ia.train()
    perdida_total = 0.0
    
    for batch in tqdm(cargador_train, desc=f"Época {epoca+1}/3", leave=False):

        imagenes = batch['imagen']
        sangre = batch['sangre']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        etiquetas_reales = batch['etiqueta']
        
        # Extraer embeddings de texto clínico congelados
        with torch.no_grad():
            salida_bert = modelo_texto_base(input_ids=input_ids, attention_mask=attention_mask)
            embeddings_texto = salida_bert.last_hidden_state[:, 0, :]
        
        optimizador.zero_grad()
        predicciones = modelo_ia(imagenes, sangre, embeddings_texto)
        perdida = criterio_error(predicciones, etiquetas_reales)
        perdida.backward()
        optimizador.step()
        
        perdida_total += perdida.item()
        
    print(f"Época [{epoca+1}/3] finalizada | Pérdida de Entrenamiento: {perdida_total/len(cargador_train):.4f}")

# 4. FASE CRÍTICA: EVALUACIÓN ANALÍTICA (Con los 200 pacientes aislados)
print("\n--- Iniciando Evaluación Médica Analítica ---")
modelo_ia.eval()

todas_las_predicciones = []
todas_las_etiquetas_reales = []

with torch.no_grad():
    for batch in cargador_val:
        imagenes = batch['imagen']
        sangre = batch['sangre']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        etiquetas_reales = batch['etiqueta']
        
        salida_bert = modelo_texto_base(input_ids=input_ids, attention_mask=attention_mask)
        embeddings_texto = salida_bert.last_hidden_state[:, 0, :]
        
        predicciones = modelo_ia(imagenes, sangre, embeddings_texto)
        clases_predichas = torch.argmax(predicciones, dim=1)
        
        # Guardar resultados para análisis estadístico
        todas_las_predicciones.extend(clases_predichas.cpu().numpy())
        todas_las_etiquetas_reales.extend(etiquetas_reales.cpu().numpy())

# 5. Desplegar reporte médico oficial de rendimiento de la IA
nombres_diagnosticos = ["Neumonía Bacteriana", "Derrame Pleural", "Paciente Sano"]
print("\nReporte de Rendimiento Estadístico del Modelo Multimodal:")
print(classification_report(todas_las_etiquetas_reales, todas_las_predicciones, target_names=nombres_diagnosticos))

# Guardar el cerebro entrenado en el disco duro
torch.save(modelo_ia.state_dict(), "cerebro_medico_multimodal.pth")
print("✔ Pesos óptimos guardados con éxito en 'cerebro_medico_multimodal.pth'")
