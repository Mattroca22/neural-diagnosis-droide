import torch
from transformers import AutoTokenizer

print("--- Paso 5: Procesamiento de Reportes Médicos Escritos ---")

# 1. Simulación de las notas clínicas reales tomadas por el médico en admisión
# Usamos un lote de 4 reportes que representan a los mismos 4 pacientes de los análisis de sangre
reportes_medicos = [
    "Paciente ingresa con fiebre alta, tos persistente y sospecha de neumonia bacteriana.",
    "El paciente se encuentra asintomatico, control de rutina sin hallazgos patologicos.",
    "Presenta disnea severa, dolor toracico agudo y signos compatibles con derrame pleural.",
    "Paciente refiere sentirse bien, signos vitales estables, pulmones limpios a la auscultacion."
]

print("\n1. Notas clínicas en texto plano recopiladas en el hospital:")
for i, reporte in enumerate(reportes_medicos):
    print(f" -> Paciente {i+1}: '{reporte}'")

# 2. Descargar el Tokenizador Médico
# Usaremos el tokenizador de BioBERT/ClinicalBERT. Este modelo ya sabe "leer" medicina
# porque fue entrenado con millones de historiales clínicos reales.
print("\n2. Descargando e inicializando el Tokenizador Médico desde Hugging Face...")
nombre_modelo = "emilyalsentzer/Bio_ClinicalBERT"
tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)

# 3. Transformar texto en números de control (Tokens e Input IDs)
# padding=True empareja el tamaño de los textos cortos agregando ceros al final
# truncation=True corta los textos si son ridículamente largos para no saturar la memoria
print("\n3. Convirtiendo las palabras de los reportes en tensores numéricos...")
tokens_salida = tokenizador(
    reportes_medicos, 
    padding=True, 
    truncation=True, 
    max_length=32, 
    return_tensors="pt" # "pt" significa que devuelva Tensores de PyTorch
)

# 4. Mostrar los resultados que recibirá nuestra capa de procesamiento de texto
print("\n4. Datos estructurados listos para ingresar al pipeline de lenguaje:")
print(f" -> Tensor de IDs de Palabras (input_ids):\n{tokens_salida['input_ids']}")
print(f" -> Dimensión del tensor obtenido [Pacientes, Longitud de Texto]: {tokens_salida['input_ids'].shape}")
print(f" -> Máscara de atención (attention_mask):\n{tokens_salida['attention_mask']}")
