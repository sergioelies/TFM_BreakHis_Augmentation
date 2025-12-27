# Impacto del aumento de datos en la clasificación de imágenes histopatológicas de cáncer de mama

Este repositorio contiene el código, la metodología y los resultados del **Trabajo Final de Máster (TFM)** del *Máster en Ciencia de Datos* (UOC).

El estudio evalúa, con **particionado estricto por paciente**, cómo distintas estrategias de **aumento de datos** afectan al rendimiento de un modelo de *deep learning* en la clasificación binaria **benigno vs. maligno** sobre imágenes histopatológicas H&E.

---

## Objetivo

Cuantificar el impacto de un **aumento de datos avanzado** (deformaciones elásticas + ruido gaussiano) frente a un aumento **básico** (geométrico/fotométrico) y **sin aumento**, en la tarea de clasificación binaria:

- **Dataset:** BreakHis (magnificación **200×**).
- **Modelo:** ResNet-50 preentrenada (feature extractor) + cabezal denso entrenable.
- **Escenarios comparados:** **None**, **Basic**, **Advanced**.

> Nota: El objetivo del TFM no es “exprimir el mejor modelo posible”, sino **comparar escenarios** bajo un diseño controlado y reproducible.

---

## Resultados principales (conjunto de prueba)

En el *test* externo (13 pacientes, 263 tiles), los tres escenarios alcanzan AUC altas. A igualdad de discriminación global (IC solapados), el punto operativo (umbral elegido por Youden en desarrollo) produce perfiles de error distintos.

| Escenario   | AUC (test) | Sensibilidad | Especificidad | FN | FP |
|------------|------------|--------------|---------------|----|----|
| None       | 0.9639     | 0.6590       | 0.9667        | 59 | 3  |
| Basic      | 0.9609     | 0.8324       | 0.9444        | 29 | 5  |
| Advanced   | 0.9683     | 0.8728       | 0.9444        | 22 | 5  |

Los resultados completos (incluyendo IC al 95%) están en `results/tfm_test_metrics.csv`.

---

## Estructura del repositorio

El proyecto se organiza en tres etapas (ejecución recomendada en orden):

### 1) Exploración y preparación
- **`1_Analisis_Exploratorio.ipynb`**
  - Estadística descriptiva y desbalance.
  - Validación de particionado por paciente.
  - Visualización cualitativa de aumentos.

### 2) Entrenamiento y validación
- **`2_Entrenamiento_y_Validacion.py`**
  - Pipeline completo reproducible:
    - *Random Search* (15 configuraciones por escenario)
    - Validación cruzada interna por paciente (StratifiedGroupKFold*, k=3)
    - Entrenamiento final y evaluación en prueba externa 
  - Escenarios: **None**, **Basic**, **Advanced**
  - Exporta métricas y predicciones para análisis posterior.

### 3) Análisis y visualización (ROC + Grad-CAM)
- **`3_Analisis_y_Visualizacion.py`**
  - Curvas ROC-AUC comparativas.
  - Generación de visualizaciones y Grad-CAM.

---

## Datos, modelos y resultados

### `/data`
Contiene los datos preprocesados en formato **`.npz`**, con **un archivo por paciente**:

- Formato de nombre: `tumortype_subtype_patientID.npz`
- Cada archivo agrupa todos los *tiles* del paciente, lo que facilita el **particionado por grupos** (evita *data leakage*).

### `/models`
Pesos de los modelos finales por escenario (PyTorch `.pth`):
- `final_model_None.pth`
- `final_model_Basic.pth`
- `final_model_Advanced.pth`

### `/results`
Evidencias numéricas y artefactos generados:
- `tfm_p1_random_search.csv`: resultados de la búsqueda de hiperparámetros.
- `tfm_p2_internal_cv.csv`: resultados de validación cruzada interna.
- `tfm_test_metrics.csv`: métricas finales en prueba (incluye IC al 95%).
- `tfm_test_predictions.npy`: predicciones crudas (y_true, y_prob) para regenerar ROC.
- Figuras generadas (p. ej., ROC y paneles Grad-CAM).

---

## Reproducibilidad

- Semillas fijadas: **SEED = 42**
- Particionado por paciente:
  - Prueba externa (15% de pacientes) reservada desde el inicio.
  - Train/validación internos también por paciente.
- Los IC se estiman mediante **cluster bootstrap por paciente**.

> Los archivos dentro de `results/` reflejan la ejecución original del proyecto. El código puede haber sido refactorizado para mejorar legibilidad, manteniendo la lógica algorítmica y semillas.

---

## Instalación

1. Clona el repositorio.
2. Instala dependencias:

```bash
pip install -r requirements.txt
