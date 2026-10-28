# EJERCICIO: preprocesamiento reutilizable para tablas numéricas y categóricas
# Solo pseudocódigo; contrastar con tabular_preprocessing.py.
#
# - Recibir nombres de columnas numéricas, categóricas y opción de escalado.
# - Para numéricas: imputar faltantes con mediana; si el modelo lo requiere,
#   escalar con StandardScaler.
# - Para categóricas: imputar con categoría más frecuente y aplicar one-hot
#   encoding que ignore niveles que aparezcan por primera vez en test/producción.
# - Combinar ambas ramas en ColumnTransformer con resto descartado o tratado
#   explícitamente.
# - Devolver transformador sin ajustarlo: Pipeline lo ajustará dentro de cada
#   partición de entrenamiento para evitar fuga de información.
# - Probar tablas solo numéricas, solo categóricas, mixtas y con valores ausentes.
