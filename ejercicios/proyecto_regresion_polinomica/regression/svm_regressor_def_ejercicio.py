# EJERCICIO: Support Vector Regression para datos tabulares
# Solo pseudocódigo; la solución funcional se encuentra en svm_regressor.py.
#
# CONSTRUCTOR DEL PIPELINE
# - Leer kernel permitido: linear, poly, rbf o sigmoid.
# - Validar C, epsilon, gamma y grado dentro de rangos razonables.
# - Recibir el preprocesador construido para las columnas del dataset.
# - Escalar columnas numéricas y codificar categorías antes de SVR.
# - Añadir SVR con las opciones elegidas al final del Pipeline.
#
# FLUJO DE ENTRENAMIENTO
# - Separar train/test antes de cualquier ajuste de transformaciones.
# - Ajustar Pipeline solo con train y predecir test.
# - Si búsqueda activa, probar combinaciones en CV interna de train y mantener
#   test aislado para evaluación final.
# - Calcular R², RMSE, MAE y sesgo medio; conservar hiperparámetros ganadores.
# - Revisar escalas, tiempo de entrenamiento y sensibilidad a C/epsilon/kernel.
