# Neural Diagnosis Droide - Multimodal Medical AI 🤖🩻

Este repositorio contiene la arquitectura de una **Red Neuronal Multimodal** desarrollada en Python utilizando **PyTorch** y **Hugging Face Transformers**. El sistema actúa como una plataforma de Soporte de Decisiones Clínicas (CDSS) en salas de urgencias, fusionando tres tipos de exámenes médicos heterogéneos en tiempo real para emitir un veredicto diagnóstico probabilístico.

---

## 🎯 Alcance del Proyecto (Fase A Completada)
El modelo procesa simultáneamente tres fuentes de información de un mismo paciente:
1.  **Visión Artificial (Imágenes):** Radiografías de tórax digitales estandarizadas.
2.  **Datos Tabulares (Bioquímica):** Valores de un hemograma completo (Glóbulos Blancos, Proteína C Reactiva y Neutrófilos).
3.  **Procesamiento de Lenguaje Natural (Texto):** Notas clínicas de admisión redactadas por el personal médico de guardia.

El objetivo diagnóstico es clasificar la correlación de estos exámenes en tres estados clínicos: **Neumonía Bacteriana**, **Derrame Pleural** o **Paciente Sano**.

---

## 🛠️ Arquitectura Técnica y Pipeline de Fusión
El núcleo de la IA utiliza un enfoque de **Fusión de Características a Nivel Medio (Feature-Level Fusion)**:
*   **Encoder de Imagen:** Red Neuronal Convolucional (**ResNet18**) modificada para extraer vectores de características visuales de 128 dimensiones a partir de imágenes de $224 \times 224$ píxeles.
*   **Encoder de Texto:** Conexión con **Bio_ClinicalBERT** (Hugging Face), un transformador optimizado con millones de historiales clínicos reales, que genera embeddings semánticos densos de 768 dimensiones.
*   **Encoder de Laboratorio:** Red neuronal densa secuencial (MLP) encargada de procesar las métricas bioquímicas tras un pipeline de normalización estadística (`StandardScaler`).
*   **Capa de Fusión:** Un bloque conector que concatena los vectores individuales, aplica regularización (*Dropout*) para evitar el sobreajuste y calcula la distribución de probabilidad diagnóstica final mediante una capa de salida con pérdida de *Entropía Cruzada*.

---

## 📂 Estructura de Archivos del Repositorio
*   `model_multimodal.py`: Definición de la clase en PyTorch y la arquitectura de la red neuronal unificada.
*   `dataset_maestro.py`: Pipeline que indexa los datos, tokeniza el lenguaje natural y empaqueta los lotes mediante *DataLoaders* asíncronos que leen directamente desde el almacenamiento.
*   `generador_hospital.py`: Script de simulación avanzada que genera una base de datos local de 1,000 registros médicos sintéticos complejos con ruido visual gaussiano y solapamiento biológico real.
*   `run_entrenamiento.py`: Algoritmo de aprendizaje supervisado encargado de dividir los datos (80% train / 20% val), optimizar las neuronas mediante *Adam* y desplegar reportes analíticos de rendimiento estadístico.
*   `visual_multimodal.py`: **[NUEVO]** Panel de control e interfaz gráfica web interactiva desarrollada con **Streamlit** que conecta el backend de IA con componentes visuales listos para producción.

---

## 📊 Estado Actual del Desarrollo
1.  **Validación de Infraestructura:** El flujo de datos multimodal se encuentra validado y libre de fugas de datos (*data leakage*).
2.  **Generalización Científica:** Tras inyectar ruido estadístico en el dataset de 1,000 pacientes, el modelo superó la fase de memorización estática, aprendiendo a calcular márgenes de incertidumbre y distribuciones de probabilidad diagnóstica realistas.
3.  **Interfaz de Producción:** La aplicación web permite arrastrar archivos de radiografías reales, modificar parámetros numéricos de laboratorio en vivo, escribir notas clínicas personalizadas y visualizar los veredictos mediante gráficos de barras interactivos. Los pesos del modelo se guardan y cargan de manera persistente desde `cerebro_medico_multimodal.pth`.

---

## 🚀 Próximos Pasos en la Hoja de Ruta
*   **Fase B (Explicabilidad Médica - XAI):** Implementar mapas de calor **Grad-CAM** sobre el visor de radiografías para que la IA pinte de color rojo la región anatómica exacta donde detectó la anomalía visual.
*   **Fase C (Datos Internacionales Reales):** Iniciar el proceso de certificación ética ante PhysioNet para migrar el entrenamiento hacia el dataset real **MIMIC-IV** del MIT.
