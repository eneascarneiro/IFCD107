# EJERCICIO: catálogo seguro de distribuciones de hiperparámetros
# Solo pseudocódigo; consultar parameter_search.py para los nombres reales.
#
# - Crear un catálogo que mapee cada modelo permitido a una distribución finita.
# - Para polinomios, listar grados y opciones numéricas/booleanas compatibles.
# - Para Random Forest, listar n_estimators, max_depth y parámetros del pipeline.
# - Para SVR, listar kernel, C, epsilon y gamma dentro de intervalos válidos.
# - Para boosting, listar número de árboles, profundidad, learning_rate, loss y
#   los parámetros propios de cada familia.
# - Definir valores por defecto distintos de las listas de búsqueda cuando haga
#   falta; documentar qué valores espera scikit-learn en cada pipeline.
# - Rechazar nombre de modelo desconocido en vez de devolver parámetros vacíos.
# - Proporcionar una copia o iterador independiente para cada búsqueda.
# - No permitir que el usuario mande una clase Python, expresión o import dinámico.
