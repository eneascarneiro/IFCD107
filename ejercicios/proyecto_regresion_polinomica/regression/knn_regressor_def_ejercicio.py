# EJERCICIO: regresión KNN con preprocesamiento y validación cruzada
# Solo pseudocódigo. La solución fuente de referencia está en knn_regressor.py.
#
# CONSTRUCTOR DE PIPELINE
# - Validar que n_neighbors sea entero positivo.
# - Crear primero StandardScaler, que aprende media y desviación solo del train.
# - Crear después KNeighborsRegressor con k vecinos, distancia y pesos elegidos.
# - Devolver ambos pasos encadenados para que CV vuelva a ajustar el escalador.
#
# INTEGRACIÓN CON K-FOLD / LEAVE-ONE-OUT
# - Comprobar que el dataset tenga target, al menos un predictor y filas válidas.
# - Rechazar predictores categóricos si esta variante está limitada a números.
# - Elegir KFold con semilla/barajado o LeaveOneOut según la ruta solicitada.
# - Comprobar que k no supere el número de filas disponibles en el menor train.
# - Obtener una predicción out-of-fold para cada observación.
# - Calcular residuos, R², RMSE y MAE sobre las predicciones OOF.
# - Ajustar un pipeline nuevo usando todas las filas para futuras inferencias.
# - Guardar modelo final y tabla de observados/predichos/residuos.
#
# RIESGO DIDÁCTICO
# - LOO realiza un ajuste por fila; estimar coste antes de aceptarlo para miles
#   de filas. K-Fold suele ser más económico para conjuntos grandes.
# - Nunca escalar toda la tabla antes de crear folds: eso filtra información.
