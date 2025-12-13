# Impacto del aumento de datos en la clasificación de imágenes histopatológicas de cáncer de mama

Este repositorio contiene el código fuente, la metodología y los resultados del **Trabajo Final de Máster (TFM)** del Máster en Ciencia de Datos de la Universitat Oberta de Catalunya (UOC).

## Objetivo

El objetivo principal de este estudio es cuantificar el impacto de técnicas de **aumento de datos avanzada** (deformaciones elásticas y ruido gaussiano) frente a técnicas geométricas clásicas y no aumento, en la tarea de clasificación binaria (*Benigno* vs *Maligno*) de imágenes histopatológicas.

* **Dataset:** [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) (Magnificación 200x).
* **Arquitectura:** ResNet-50.
* **Enfoque:** Comparativa de 3 escenarios de aumento (None, Basic, Advanced).

## Estructura del Repositorio

El proyecto se divide en tres módulos secuenciales:

### 1. Exploración y Metodología
*  **`1_Analisis_Exploratorio.ipynb`**
    * Análisis descriptivo del dataset y desbalance de clases.
    * Validación de la estrategia de particionado por paciente.
    * **Visualización de aumentos:** Generación de ejemplos visuales de las transformaciones aplicadas.

### 2. Entrenamiento
*  **`2_Entrenamiento_y_Evaluacion.py`**
    * Script maestro de entrenamiento reproducible.
    * Implementa el flujo completo: *Random Search* $\rightarrow$ *Validación Cruzada Interna* $\rightarrow$ *Test Final*.
    * Gestión de experimentos para los escenarios **None**, **Basic** y **Advanced**.

### 3. Resultados e Interpretabilidad
*  **`3_Analisis_y_Visualizacion.py`**
    * Generación de métricas finales y curvas **ROC-AUC** comparativas.
    * **Grad-CAM:** Generación de mapas de calor para validar la atención del modelo en regiones de interés biológico.

---
## Datos, Modelos y Resultados

Esta estructura de directorios contiene tanto los datos de entrada como todas las evidencias generadas durante la ejecución del proyecto.

* **`/data`**: Contiene el dataset preprocesado listo para el entrenamiento.
    * `breakhis_npz/`: Carpeta con los archivos `.npz` (tensors de imágenes 200x normalizadas). Cada archivo agrupa las imágenes de un paciente específico para garantizar la independencia en el particionado.

* **`/models`**: Almacena los pesos del mejor modelo por escenario. (State Dictionaries de PyTorch).
* 
* **`/results`**: Contiene las evidencias forenses y visuales de la ejecución:
    * **Tablas de Métricas:**
        * `tfm_test_metrics.csv`: Resultados finales en el conjunto de Test (AUC, Umbral de decisión, Sensibilidad, Especificidad, VPP, VPN e Intervalos de Confianza).
        * `tfm_p1_random_search.csv`: Registro de experimentos de la fase de búsqueda de hiperparámetros.
        * `tfm_p2_internal_cv.csv`: Resultados de la validación cruzada interna.
    * **Logs y Predicciones:**
        * `train_log.txt`: Registro completo de la consola durante el proceso de entrenamiento.
        * `tfm_test_predictions.npy`: Archivo binario con las probabilidades crudas (`y_true`, `y_prob`) necesarias para regenerar las curvas ROC.

## Resumen de Hallazgos

El estudio demuestra que la estrategia de aumento avanzada mejora la sensibilidad diagnóstica, reduciendo los falsos negativos críticos.

| Escenario | AUC | Sensibilidad | Falsos Negativos |
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
