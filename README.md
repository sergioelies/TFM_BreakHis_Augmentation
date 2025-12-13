# Impacto del aumento de datos en la clasificación de imágenes histopatológicas de cáncer de mama

Este repositorio contiene el código fuente, la metodología y los resultados del **Trabajo Final de Máster (TFM)** del Máster en Ciencia de Datos de la Universitat Oberta de Catalunya.

## Objetivo

El objetivo principal de este estudio es cuantificar el impacto de técnicas de **aumento de datos avanzada** (deformaciones elásticas y ruido gaussiano) frente a técnicas geométricas clásicas, en la tarea de clasificación binaria (*Benigno* vs *Maligno*) de imágenes histopatológicas.

* **Dataset:** [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) (Magnificación 200x).
* **Arquitectura:** ResNet-50 (Feature Extraction).
* **Enfoque:** Comparativa de 3 escenarios de aumento (None, Basic, Advanced).

## Estructura del Repositorio

El proyecto se divide en tres módulos secuenciales:

### 1. Exploración y Metodología
*  **`1_Analisis_Exploratorio.ipynb`**
    * Análisis descriptivo del dataset y desbalance de clases.
    * Validación de la estrategia de particionado por paciente (evitando *data leakage*).
    * **Visualización de aumentos:** Generación de ejemplos visuales de las transformaciones aplicadas (incluyendo la deformación elástica simulando la plasticidad del tejido).

### 2. Entrenamiento (Pipeline ML)
*  **`2_Entrenamiento_y_Evaluación.py`**
    * Script maestro de entrenamiento reproducible.
    * Implementa el flujo completo: *Random Search* $\rightarrow$ *Validación Cruzada Interna* $\rightarrow$ *Test Final*.
    * Gestión de experimentos para los escenarios **None**, **Basic** y **Advanced**.

### 3. Resultados e Interpretabilidad
*  **`3_Analisis_y_Visualizacion.py`**
    * Generación de métricas finales y curvas **ROC-AUC** comparativas.
    * **Grad-CAM:** Generación de mapas de calor para validar la atención del modelo en regiones de interés biológico (núcleos vs. estroma).

---

## 📁 Datos y Resultados

* **`/results`**: Contiene las evidencias originales de la ejecución del TFM:
    * `tfm_test_metrics.csv`: Métricas detalladas con intervalos de confianza.
    * `train_log.txt`: Logs completos de la ejecución del entrenamiento.
    * Gráficas generadas (`.png`).
* **`/models`**: Pesos de los modelos entrenados (`.pth`).
* **`/data`**: *Nota: Debido al límite de tamaño de archivos de GitHub, los archivos de datos preprocesados (`.npz`) no se incluyen en el repositorio. El script de preprocesamiento original se encuentra documentado en el Notebook 1.*

## 📊 Resumen de Hallazgos

El estudio demuestra que la estrategia de aumento avanzada mejora significativamente la sensibilidad diagnóstica, reduciendo los falsos negativos críticos.

| Escenario | AUC (Test) | Sensibilidad | Falsos Negativos |
| :--- | :---: | :---: | :---: |
| **Advanced** | **0.9739** | **93.67%** | **11** |
| None | 0.9653 | 86.73% | 23 |
| Basic | 0.9541 | 91.88% | 14 |

> **Nota sobre Reproducibilidad:**
> Los archivos de logs y resultados numéricos ubicados en la carpeta `/results` corresponden a la ejecución original del proyecto. El código fuente presentado aquí ha sido refactorizado para mejorar la legibilidad y estructura, manteniendo intacta la lógica algorítmica y las semillas aleatorias (`SEED=42`) utilizadas.

## Instalación y Uso

1.  Clonar el repositorio.
2.  Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ejecutar los scripts en orden numérico.

---
**Autor:** Sergio Elies
**Licencia:** MIT
