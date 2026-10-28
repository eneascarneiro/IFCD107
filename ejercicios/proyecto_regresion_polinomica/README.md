# API Flask de regresión polinómica

Ejemplo ejecutable con rutas Flask en `app.py`, lógica compartida en `api_functions.py` y algoritmos separados en `regression/`. La guía completa y las instrucciones de Postman están en `documentacion/guia_proyecto.tex`.

El CSV incluido tiene 1.000 filas sintéticas para una demostración de gestión de expedientes de terrazas. No son registros, plazos ni resultados reales del Ayuntamiento. En el ajuste, `x` es un índice didáctico de complejidad y `y` son días de tramitación simulados.

## Arranque

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

En macOS/Linux, activa con `source .venv/bin/activate`, instala `requirements.txt` y ejecuta `python app.py`.

La API carga automáticamente los valores del archivo local `.env`. En esta copia, `.env` contiene una contraseña aleatoria para `admin` y un secreto aleatorio para firmar tokens. Para Postman, copia `API_ADMIN_PASSWORD` desde ese archivo a la variable de colección `password`.

`.env` está excluido de Git y no debe compartirse ni incluirse en el ZIP distribuible. Para configurar otro entorno, copia `.env.example` a `.env` y asigna credenciales propias. Mantén `API_TOKEN_SECRET` estable entre procesos y reinicios. Si no existe, el ejemplo genera un secreto temporal, por lo que los tokens se invalidan al reiniciar. En producción, utiliza contraseñas con hash y un gestor de secretos.

Importa `postman/Regresion_Polinomica.postman_collection.json` en Postman. Ejecuta `Obtener token` antes de las solicitudes de ajuste; la colección guarda el Bearer token automáticamente. Para la petición CSV, selecciona `ejemplos/datos_ejemplo.csv` en el campo `file` y usa la referencia `terrazas_madrid_demo`.

Para regenerar el conjunto determinista de 1.000 filas: `python scripts/generate_terrace_cases.py`.

La API mantiene endpoints en `app.py`, funciones auxiliares compartidas en `api_functions.py` y clases de algoritmo dentro de `regression/`. Los artefactos se crean bajo `runs/<reference>/` en `graficas/`, `csv_entrada/`, `logs_salida/`, `csv_salida/` y `modelos/`. El registro central `runs/_auditoria/api_audit.jsonl` conserva una entrada por petición con fecha UTC, método, ruta, estado, usuario, IP y huella SHA-256 truncada del token. Los logs nunca guardan el token Bearer completo. Si se usa un proxy inverso, `REMOTE_ADDR` reflejará el proxy salvo que se configure confianza explícita para proxies conocidos. El modelo se guarda como `.pkt`; carga solo archivos de confianza porque utiliza pickle.

Endpoints: `GET /api/v1/health` y `GET /api/v1/regresion/parametros` (públicos), `POST /api/v1/auth/token` (emite token), regresión polinómica JSON/CSV y búsqueda, KNN con K-Fold/Leave-One-Out, Random Forest desde CSV o desde el ejemplo, y SVR, AdaBoost y Gradient Boosting desde CSV. Todas las rutas de entrenamiento requieren Bearer. Los tokens expiran por defecto en una hora; configura `API_TOKEN_TTL_SECONDS` para cambiarlo.

`GET /api/v1/regresion/parametros` devuelve un catálogo donde cada parámetro se presenta como un elemento con `parameter` y `items`. Los endpoints de entrenamiento aceptan `search=true`, `n_iter` y `search_cv`; en ese modo `RandomizedSearchCV` toma sus candidatos de las mismas listas. La prueba final queda separada de la búsqueda. La respuesta incluye `parameters` con los valores ganadores, `requested_parameters` y las métricas de validación cruzada. En KNN la búsqueda se hace de forma anidada dentro de cada pliegue externo; al combinarla con Leave-One-Out se limita a 200 filas.

Ejemplo multipart para activar la búsqueda en SVR, AdaBoost, Gradient Boosting o KNN:

```text
search=true
n_iter=10
search_cv=5
```

En Random Forest se pueden enviar esos mismos campos en JSON o multipart. Con `search=false` se conserva el ajuste directo usando los parámetros indicados en la petición.


## Random Forest

La API dispone de `POST /api/v1/random-forest/terrazas`, que entrena con `ejemplos/datos_ejemplo_complejo.csv`, y `POST /api/v1/random-forest/csv`, que recibe un fichero mediante `multipart/form-data`. Ambas rutas requieren el token Bearer obtenido con `/api/v1/auth/token`.

Ejemplo para la ruta incluida:

```json
{
  "reference": "terrazas_rf_demo",
  "target": "y",
  "n_estimators": 200,
  "max_depth": null,
  "test_size": 0.2,
  "random_state": 42
}
```

`expediente_id` se conserva para identificar predicciones pero no se usa como predictor. Se procesan variables numéricas y categóricas; los artefactos se guardan por referencia como en la regresión polinómica. El conjunto de terrazas es sintético. La colección Postman incluye solicitudes para las dos rutas.

## SVR y boosting para regresión

Los modelos nuevos siguen el mismo contrato CSV y guardado de artefactos:

- `POST /api/v1/regresion/svm/csv` usa `SVR` dentro de un pipeline que imputa valores ausentes, codifica categorías y escala variables numéricas.
- `POST /api/v1/regresion/boosting/adaboost/csv` usa `AdaBoostRegressor` con árboles base.
- `POST /api/v1/regresion/boosting/gradient/csv` usa `GradientBoostingRegressor`.

Las tres rutas reciben `multipart/form-data` con `reference`, `target`, opciones del modelo y el fichero `file`. Requieren token Bearer y aceptan un CSV de hasta 4 MB. Las salidas mantienen la estructura del proyecto: CSV de entrada y predicciones, gráfico observado frente a predicho, modelo `.pkt`, log y métricas R², RMSE, MAE y sesgo. El preprocesamiento se ajusta dentro del pipeline solo con las filas de entrenamiento.

Cada estimador está en su módulo de `regression/`: `svm_regressor.py`, `adaboost_regressor.py` y `gradient_boosting_regressor.py`. `tabular_preprocessing.py` centraliza la imputación/codificación común y el escalado específico que necesita SVR. La colección Postman incluye una solicitud de ejemplo para cada endpoint nuevo usando el CSV sintético incluido.

### Ejecutar las seis prácticas SVR paso a paso

El script `regression/train_svm_casuisticas.py` es una guía comentada para principiantes y comparte `train_one_dataset` con el endpoint EDA. Para cada CSV primero genera el informe EDA y guarda los datos transformados; solo después ajusta el SVR y calcula las métricas en un conjunto de prueba. Ejecútalo desde la raíz:

```powershell
.venv\Scripts\python.exe regression\train_svm_casuisticas.py
```

Los PDFs, CSV preprocesados, predicciones, modelos y métricas individuales se guardan en `runs/svm_casuisticas/<casuistica>/`. `runs/svm_casuisticas/resumen_entrenamientos.csv` compara R², RMSE, MAE y sesgo entre los seis casos. Los datos y métricas son didácticos y no describen expedientes reales.

## EDA y preprocesado tabular con informe PDF

`POST /api/v1/eda/preprocesado/csv` acepta `multipart/form-data` con `reference` y el archivo `file`; requiere Bearer token y limita el CSV a 4 MB. El endpoint llama a `train_one_dataset` en `regression/train_svm_casuisticas.py`: primero elabora el informe EDA y el CSV transformado, y después entrena SVR sobre esos datos. La clase `regression/eda_preprocesado.py` convierte `tipo_tramite` y `distrito` a códigos numéricos reproducibles, convierte `requerimientos_documentacion` (o `requiere_documentacion`) a número o códigos si es texto, y sustituye `mes_presentacion` por las variables cíclicas seno/coseno. La respuesta incluye las métricas y rutas del informe PDF, CSV preprocesado, predicciones y modelo guardado bajo `runs/<reference>/`.

## n8n, Dify, Flowise y modelos locales con Ollama

El `compose.yaml` de la raíz levanta la API de regresión, n8n, Flowise y Ollama. También incluye el Compose oficial de Dify para mantener sus servicios y configuración alineados con el proyecto upstream. Se requiere Docker Compose v2.20 o posterior. Dify se incorpora desde `dify/docker/docker-compose.yaml`, por lo que antes del primer arranque hay que preparar su repositorio y su archivo de entorno:

```powershell
git clone https://github.com/langgenius/dify.git dify
if (!(Test-Path .env)) { Copy-Item .env.example .env }
if (!(Test-Path dify/docker/.env)) { Copy-Item dify/docker/.env.example dify/docker/.env }
```

Si ya existe `.env`, conserva sus valores y añade las nuevas variables que falten desde `.env.example`. Edita ambos archivos de entorno y sustituye las contraseñas y claves de ejemplo por valores aleatorios seguros. Mantén los mismos nombres de modelos (`llama3:8b` y `mistral:7b`) si quieres que los servicios usen los modelos descargados. La primera ejecución requiere conexión a Internet para descargar las imágenes de contenedor y varios GB de modelos:

En `dify/docker/.env`, limita también la publicación web de Dify al equipo local configurando `EXPOSE_NGINX_PORT=127.0.0.1:8080` y `EXPOSE_NGINX_SSL_PORT=127.0.0.1:8443`.

```powershell
docker compose up -d --build
```

Accesos locales:

- API de regresión: `http://localhost:5000`
- n8n: `http://localhost:5678`
- Flowise: `http://localhost:3001` (credenciales definidas en `.env`)
- Dify: `http://localhost:8080/install` la primera vez; después, `http://localhost:8080`
- Ollama: API local en `http://localhost:11434`

Ollama descarga por defecto `llama3:8b` y `mistral:7b`, que se guardan en el volumen local `./ollama`. En Flowise y n8n, configura una conexión Ollama con URL `http://ollama:11434`; en Dify, configura el proveedor Ollama con URL `http://host.docker.internal:11434` y el nombre exacto del modelo. Docker Desktop resuelve `host.docker.internal`; en instalaciones Linux puede ser necesario habilitar el acceso al host desde los contenedores.

La inferencia con estos modelos locales no factura llamadas a una API externa. El equipo sigue consumiendo almacenamiento, memoria, CPU/GPU y electricidad; el primer arranque también descarga imágenes y modelos. El Compose deja las interfaces ligadas a `127.0.0.1`, de modo que son accesibles desde el propio equipo. Para detener los servicios sin borrar los datos: `docker compose down`; no añadas `-v` si quieres conservar los volúmenes.

## Ejemplo de KNN con Ollama

`ejemplos/knn_con_ollama.py` ejecuta KNN de regresión con scikit-learn sobre `ejemplos/datos_knn_5000_outliers.csv`. Después, envía a Ollama solo un resumen de las métricas para obtener una explicación en lenguaje natural; Ollama no entrena KNN ni recibe las filas del CSV.

Con Ollama instalado y en ejecución, descarga un modelo local y lanza el ejemplo desde la raíz del proyecto:

```powershell
ollama pull llama3.2
python ejemplos/knn_con_ollama.py
```

Para probar la alternativa de escalado robusto ante outliers: `python ejemplos/knn_con_ollama.py --scaler robust`. Para cambiar el modelo, indica `--model nombre:tag` o establece `OLLAMA_MODEL`. La conexión usa la API local de Ollama en `http://localhost:11434`.

## NVIDIA Triton: regresores lineal, polinómico y Random Forest

El directorio independiente `triton/` entrena los modelos con las clases de `regression/`, guarda los artefactos y los publica mediante NVIDIA Triton Python backend. El entrenamiento ocurre antes de que Triton cargue sus modelos; las llamadas HTTP solo realizan inferencia.

Desde la raíz del proyecto:

```powershell
docker compose -f triton/compose.yaml up --build
```

Para hacer visible la GPU NVIDIA al contenedor Triton (con NVIDIA Container Toolkit disponible):

```powershell
docker compose -f triton/compose.yaml -f triton/compose.gpu.yaml up --build
```

La API V2 se publica en el puerto 8000; importa `triton/postman/NVIDIA_Triton_Regressors.postman_collection.json` para probar los tres modelos. Consulta `triton/README.md` para entradas, datos y explicación del flujo.

## Regresión polinómica servida con BentoML y Docker

El componente independiente `bentoml/` incluye el entrenamiento, el servicio de inferencia, la imagen Docker y la composición de contenedores. No modifica la API Flask existente. Consulta [`bentoml/README.md`](bentoml/README.md) para el flujo completo y la petición de Postman.

Desde la raíz del proyecto:

```powershell
docker compose -f bentoml/compose.yaml up --build
```

Compose entrena el pipeline de grado 2 con el CSV sintético `ejemplos/datos_ejemplo_xy.csv`, lo registra en el volumen BentoML y después sirve `POST http://localhost:3000/predict`. La ejecución verificada con 1.000 observaciones reservó 200 para evaluación: R² = 0,9287, RMSE = 6,9001 y MAE = 5,5080. Son métricas del conjunto de ejemplo, no de datos reales. La solicitud acepta JSON con `{"x":[40,50,60]}`; la respuesta añade las métricas, el grado y el número de filas usadas para el ajuste final.
