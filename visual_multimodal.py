import streamlit as str
import torch
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import pandas as pd
from model_multimodal import RedMultimodalMedica

# Configuración inicial de la página web
str.set_page_config(page_title="Neural Diagnosis Droide", page_icon="🤖", layout="wide")

# Inicialización y caché de modelos pesados para que la web cargue rápido
@str.cache_resource
def cargar_modelos_base():
    tokenizador = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    modelo_texto_base = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    modelo_texto_base.eval()
    
    modelo_ia = RedMultimodalMedica(num_clases=3)
    # Corrección de seguridad para compatibilidad con las nuevas versiones de PyTorch
    modelo_ia.load_state_dict(torch.load("cerebro_medico_multimodal.pth", map_location=torch.device('cpu'), weights_only=False))
    modelo_ia.eval()
    return tokenizador, modelo_texto_base, modelo_ia

tokenizador, modelo_texto_base, modelo_ia = cargar_modelos_base()

# Pipeline de Visión Artificial para procesar las imágenes cargadas por el usuario
transformador_imagen = T.Compose([
    T.Resize((224, 224)),
    T.Lambda(lambda img: img.convert("RGB")),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- DISEÑO DE LA INTERFAZ GRÁFICA (FRONTEND) ----
str.title("🤖 Neural Diagnosis Droide - Panel de Control Multimodal")
str.write("Soporte de Inteligencia Artificial para Decisiones Clínicas en Urgencias.")
str.markdown("---")

# Dividir la pantalla web en dos columnas principales
col_izquierda, col_derecha = str.columns(2)

with col_izquierda:
    str.header("📋 Ingreso de Exámenes del Paciente")
    
    # Examen 1: Carga de Imagen de Radiografía
    archivo_imagen = str.file_uploader("1. Arrastre la Radiografía de Tórax (JPG/PNG)", type=["jpg", "png", "jpeg"])
    if archivo_imagen:
        str.image(archivo_imagen, caption="Radiografía cargada por el usuario", width=250)
        
    # Examen 2: Valores Numéricos de Laboratorio (Sangre)
    str.subheader("2. Analítica de Sangre (Hemograma)")
    sub_col1, sub_col2, sub_col3 = str.columns(3)
    with sub_col1:
        wbcs = str.number_input("Glóbulos Blancos (uL)", min_value=1000.0, max_value=50000.0, value=7500.0, step=100.0)
    with sub_col2:
        pcr = str.number_input("Proteína C Reactiva (mg/L)", min_value=0.0, max_value=300.0, value=3.2, step=0.5)
    with sub_col3:
        neutros = str.number_input("Neutrófilos (%)", min_value=10.0, max_value=100.0, value=60.0, step=0.5)
        
    # Examen 3: Reporte Clínico en Texto Plano
    str.subheader("3. Nota Clínica de Admisión")
    nota_clinica = str.text_area(
        "Redacte los síntomas observados en el paciente:",
        value="Paciente acude por valoracion general. Signos estables, asintomatico respiratorio actual."
    )

# Columna derecha encargada de procesar la inferencia matemática y graficar resultados
with col_derecha:
    str.header("📊 Veredicto Diagnóstico de la IA")
    str.write("Haga clic abajo para ejecutar la fusión de características en tiempo real.")
    
    # Botón disparador de la predicción
    if str.button("Ejecutar Diagnóstico Multimodal", type="primary"):
        if not archivo_imagen:
            str.error("❌ Error: Debe cargar una imagen de Radiografía para proceder.")
        else:
            with str.spinner("Procesando exámenes médicos de forma simultánea..."):
                # A. Preprocesamiento de la Imagen cargada
                img_pil = Image.open(archivo_imagen)
                tensor_img = transformador_imagen(img_pil).unsqueeze(0)
                
                # B. Preprocesamiento de la Sangre (Normalización básica de prueba)
                sangre_array = [[(wbcs-11000)/5000, (pcr-40)/35, (neutros-65)/10]]
                tensor_sangre = torch.tensor(sangre_array, dtype=torch.float32)
                
                # C. Preprocesamiento del Texto Clínico
                tokens = tokenizador([nota_clinica], padding=True, truncation=True, max_length=32, return_tensors="pt")
                with torch.no_grad():
                    salida_bert = modelo_texto_base(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
                    embeddings_texto = salida_bert.last_hidden_state[:, 0, :]
                    
                    # D. Inferencia Multimodal
                    logits_salida = modelo_ia(tensor_img, tensor_sangre, embeddings_texto)
                    probabilidades = torch.softmax(logits_salida, dim=1).flatten().numpy()
                
                # Mapear los resultados obtenidos en un DataFrame de Pandas para graficarlos
                mapeo_clases = ["Neumonía Bacteriana", "Derrame Pleural", "Paciente Sano"]
                df_grafico = pd.DataFrame({
                    "Diagnóstico": mapeo_clases,
                    "Probabilidad (%)": probabilidades * 100
                })
                
                # Desplegar la gráfica de barras nativa e interactiva de Streamlit
                str.success("✔ Procesamiento multimodal completado.")
                str.bar_chart(df_grafico.set_index("Diagnóstico"))
                
                # Veredicto de mayor probabilidad destacado en pantalla
                id_ganador = df_grafico["Probabilidad (%)"].idxmax()
                diagnostico_final = df_grafico.iloc[id_ganador]["Diagnóstico"]
                porcentaje_final = df_grafico.iloc[id_ganador]["Probabilidad (%)"]
                
                str.metric(label="Diagnóstico Sugerido con Mayor Certeza", value=diagnostico_final.upper(), delta=f"{porcentaje_final:.2f}% Probabilidad")
