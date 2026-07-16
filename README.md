# Multimodal Medical AI (MAI) - Prototype v1.0

Este repositorio contiene la arquitectura base de una **Red Neuronal Multimodal** desarrollada en Python utilizando **PyTorch** y **Hugging Face Transformers**. El sistema está diseñado para el Soporte de Decisiones Clínicas (CDSS), con la capacidad de fusionar tres tipos de exámenes médicos heterogéneos en paralelo para emitir un diagnóstico probabilístico unificado.

## 🎯 Alcance del Proyecto (MVP)
El modelo actual procesa simultáneamente:
1. **Visión Artificial (Imágenes):** Radiografías de tórax digitales.
2. **Datos Tabulares (Bioquímica):** Valores de un hemograma completo (Glóbulos Blancos, Proteína C Reactiva y Neutrófilos).
3. **Procesamiento de Lenguaje Natural (Texto):** Notas y reportes clínicos redactados en el área de admisión hospitalaria.

El objetivo diagnóstico de esta primera fase es clasificar con alta precisión entre tres estados clínicos mutuamente excluyentes: **Neumonía Bacteriana**, **Derrame Pleural** o **Paciente Sano**.

---

## 🛠️ Arquitectura Técnica del Sistema

El núcleo de la IA utiliza un enfoque de **Fusión de Características a Nivel Medio (Feature-Level Fusion)**:
*   **Encoder de Imagen:** Red Neuronal Convolucional (**ResNet18**) preentrenada, modificada para extraer vectores de características visuales de 128 dimensiones.
*   **Encoder de Texto:** Conexión directa con **Bio_ClinicalBERT** (un modelo transformador optimizado con millones de historiales clínicos reales), extrayendo embeddings semánticos densos de 768 dimensiones.
*   **Encoder de Laboratorio:** Red neuronal densa secuencial (MLP) encargada de procesar las métricas bioquímicas normalizadas estadísticamente.
*   **Capa de Fusión:** Un cerebro conector que concatena los vectores individuales, aplica regularización (*Dropout*) y calcula la distribución de probabilidad diagnóstica final mediante una capa de salida con pérdida de *Entropía Cruzada*.

---

## 📂 Estructura de Archivos del Repositorio

*   `model_multimodal.py`: Contiene la definición de la clase en PyTorch y la arquitectura de la red neuronal unificada.
*   `dataset_maestro.py`: Pipeline maestro encargado de indexar, tokenizar texto, redimensionar imágenes a $224 \times 224$ píxeles y empaquetar los lotes (*DataLoaders*).
*   `generador_hospital.py`: Script de simulación que genera una base de datos local de 1,000 registros médicos sintéticos con ruido estadístico y visual para el entrenamiento.
*   `run_entrenamiento.py`: Algoritmo de aprendizaje supervisado encargado de dividir los datos (80% train / 20% val), optimizar los pesos mediante *Adam* y desplegar el reporte analítico de rendimiento.
*   `inferencia_medica.py`: Módulo de producción que carga los pesos entrenados del disco (`.pth`) y procesa casos de pacientes individuales en tiempo real.

---

## 📊 Estado Actual del Desarrollo

Hemos completado la validación completa de la infraestructura de software:
1.  Se comprobó la integridad dimensional del flujo multimodal de datos de extremo a extremo.
2.  Se implementó un pipeline de normalización estadística (`StandardScaler`) para las analíticas sanguíneas y un pipeline de visión artificial para estandarizar las imágenes médicas.
3.  El modelo fue entrenado con un lote masivo simulado, demostrando una alta capacidad de convergencia matemática y guardando con éxito los pesos del modelo final (`cerebro_medico_multimodal.pth`).

*Nota: Actualmente el modelo muestra una precisión perfecta ($1.00$ F1-Score) debido a la naturaleza matemática controlada de los rangos del generador sintético. Esto valida el código de software, preparando la infraestructura para enfrentar la ambigüedad real en los siguientes pasos.*

---

## 🚀 Próximos Pasos en la Hoja de Ruta
*   **Fase A:** Desarrollo de una interfaz gráfica visual interactiva utilizando **Streamlit** para eliminar el uso de la consola.
*   **Fase B:** Implementación de técnicas de **Explicabilidad Médica (XAI)** como mapas de calor **Grad-CAM** sobre las imágenes para dar transparencia al diagnóstico.
*   **Fase C:** Migración y escalado del sistema hacia bases de datos internacionales protegidas y reales (**MIMIC-IV** del MIT).
