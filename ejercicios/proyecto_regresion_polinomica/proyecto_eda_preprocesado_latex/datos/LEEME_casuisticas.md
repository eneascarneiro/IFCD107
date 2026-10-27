# CSV para prácticas de EDA

Los seis archivos se generan de forma determinista con `python scripts/generar_csv_casuisticas.py`. Comparten las columnas del ejemplo principal y contienen datos inventados para aprendizaje; no representan expedientes reales.

| Archivo | Casuística | Qué estudiar |
|---|---|---|
| `01_referencia_limpia.csv` | Referencia sin problemas intencionados | Tipos, distribuciones, asociaciones y flujo base. |
| `02_valores_ausentes.csv` | Faltantes en variables numéricas, categóricas y objetivo | Mapa/conteo de nulos, imputación y exclusión de objetivo ausente. |
| `03_valores_atipicos.csv` | Extremos intencionados, algunos potencialmente imposibles | Histogramas, IQR, validación de rangos y sensibilidad. |
| `04_categorias_raras.csv` | Categorías poco frecuentes, espacios y diferencias de mayúsculas | Normalización, cardinalidad y tratamiento de categorías nuevas. |
| `05_duplicados.csv` | Filas idénticas y mismo ID con objetivo distinto | Duplicados exactos frente a conflictos de clave. |
| `06_sesgo_y_heterocedasticidad.csv` | Predictor sesgado y dispersión de y creciente con x | Transformaciones, residuos y límites de correlación lineal. |

Todos son casos simulados; los anomalías son didácticas y requieren revisión contextual antes de decidir cualquier tratamiento.
