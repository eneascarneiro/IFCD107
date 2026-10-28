# EJERCICIO: regresión lineal simple con holdout
# Solo pseudocódigo; usar linear_regressor.py para contrastar la solución.
#
# CONSTRUCTOR
# - Validar test_size dentro del rango permitido y random_state entero no negativo.
# - Guardar opciones para repetir la partición.
#
# MÉTODO fit(X, y)
# 1. Convertir X/y a arrays numéricos y comprobar filas alineadas.
# 2. Para esta práctica, exigir exactamente un predictor x y mínimo de filas.
# 3. Rechazar valores no finitos.
# 4. Separar train/test usando test_size y semilla reproducible.
# 5. Ajustar una regresión lineal únicamente con train.
# 6. Predecir test y calcular R², raíz del error cuadrático medio y error absoluto.
# 7. Ajustar otro LinearRegression con todos los datos para el modelo de servicio.
# 8. Devolver modelo final, métricas de holdout y tamaños de train/test.
