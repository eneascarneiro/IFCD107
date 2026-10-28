# EJERCICIO: AdaBoostRegressor para datos tabulares
# Solo pseudocódigo; completar siguiendo adaboost_regressor.py.
#
# CONSTRUCTOR
# - Recibir preprocesador y opciones: número de estimadores, profundidad del
#   árbol base, tasa de aprendizaje y función de pérdida permitida.
# - Crear árbol de decisión de regresión de profundidad limitada.
# - Crear AdaBoostRegressor que combine árboles débiles secuencialmente.
# - Mantener la versión de scikit-learn y el nombre del parámetro base estimator
#   compatibles con el entorno definido para el curso.
# - Devolver un Pipeline con preprocesador antes del estimador.
#
# ENTRENAMIENTO
# - Particionar train/test; ajustar solo con train.
# - Opcionalmente buscar hiperparámetros con CV anidada en train.
# - Evaluar holdout con R², RMSE y MAE; guardar parámetros y predicciones.
# - Explicar que los siguientes estimadores intentan corregir errores previos.
