# EJERCICIO: KNN calcula; un LLM local explica un resumen de resultados
# Solo pseudocódigo. Ollama es una herramienta auxiliar, no modifica el modelo.
#
# - Definir argumentos de consola para CSV, target, vecinos, escalador, modelo
#   local, URL y opción de omitir explicación.
# - Leer CSV, validar target y excluir ID de predictores.
# - Convertir predictores y objetivo a números y rechazar faltantes/infinito.
# - Dividir train/test con semilla fija.
# - Elegir StandardScaler o RobustScaler dentro de un Pipeline con KNN.
# - Ajustar pipeline en train, predecir test y calcular métricas con scikit-learn.
# - Construir un resumen pequeño: modelo, filas, columnas, parámetros y métricas.
# - Mostrar el resumen calculado antes de cualquier llamada de red.
# - Si se pidió omitir LLM, finalizar tras imprimir métricas.
# - En caso contrario, enviar a Ollama solo el resumen, nunca el CSV completo.
# - Solicitar explicación prudente en español sin permitir que invente resultados.
# - Gestionar servidor apagado, modelo inexistente, timeout y JSON mal formado.
# - Aclarar que texto generado no es una métrica ni sustituye evaluación técnica.
