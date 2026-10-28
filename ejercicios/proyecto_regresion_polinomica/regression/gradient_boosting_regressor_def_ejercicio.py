# EJERCICIO: GradientBoostingRegressor para datos tabulares
# Solo pseudocódigo; completar siguiendo gradient_boosting_regressor.py.
#
# CONSTRUCTOR
# - Recibir preprocesador, número de árboles, profundidad, learning_rate,
#   loss y subsample.
# - Validar que loss sea una opción admitida y que subsample pertenezca a (0, 1].
# - Crear GradientBoostingRegressor con semilla reproducible.
# - Encadenar preprocesador y estimador en Pipeline.
#
# ENTRENAMIENTO Y ANÁLISIS
# - Separar holdout; ajustar todas las transformaciones solo con train.
# - Si hay búsqueda, usar RandomizedSearchCV sobre train y KFold interno.
# - Medir una sola vez el estimador seleccionado en test.
# - Calcular R², RMSE, MAE y residuo medio; persistir salidas y parámetros.
# - Comparar learning_rate, profundidad y cantidad de estimadores con cuidado:
#   árboles complejos pueden sobreajustar y los valores pequeños requieren más.
