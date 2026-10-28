"""API web didáctica para ejecutar modelos de regresión.

Recorrido de una petición:
1. Flask recibe una ruta HTTP y comprueba el token Bearer.
2. Este archivo valida los campos propios de la petición y lee JSON o CSV.
3. api_functions.py prepara los datos, entrena y guarda métricas/artefactos.
4. regression/ contiene un módulo Python por algoritmo.

Las respuestas de entrenamiento devuelven JSON con un identificador de
ejecución y rutas relativas a los archivos guardados bajo runs/<reference>/.

Mapa de rutas:
- GET  /api/v1/health: estado del servicio; no necesita token.
- GET  /api/v1/regresion/parametros: listas de valores candidatos por modelo.
- POST /api/v1/auth/token: recibe username/password; devuelve Bearer.
- POST /api/v1/regresion/polinomica: puntos x/y en JSON.
- POST /api/v1/regresion/polinomica/csv: columnas x/y en CSV.
- POST /api/v1/regresion/polinomica/randomizedsearch: búsqueda desde JSON o CSV.
- POST /api/v1/random-forest/terrazas y /api/v1/random-forest/csv.
- POST /api/v1/regresion/knn/kfold y /api/v1/regresion/knn/leave-one-out.
- POST /api/v1/regresion/svm/csv: regresión con Support Vector Regression.
- POST /api/v1/regresion/boosting/adaboost/csv.
- POST /api/v1/regresion/boosting/gradient/csv.
- POST /api/v1/eda/preprocesado/csv: EDA, codificación, informe PDF y entrenamiento SVR.
- POST /api/v1/primitiva/analizar/csv: ejecuta el analizador local sobre un CSV histórico.
- POST /api/v1/primitiva/knn/csv y /api/v1/primitiva/cnn/csv: modelos didácticos sobre histórico CSV.

Todas las rutas POST de entrenamiento requieren el encabezado Authorization:
Bearer <token>. Los endpoints CSV reciben multipart/form-data con un campo
file; los polinómicos también aceptan puntos x/y como JSON.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, g, jsonify, request
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv

from api_functions import (
    parse_csv, parse_json_points, random_forest_options, run_model,
    run_random_forest, run_polynomial_search, run_knn_cross_validation,
    run_tabular_regressor, regression_model_options, validate_reference, parse_integer,
    parse_search_options, run_eda_preprocesado, run_primitiva_analysis, run_primitiva_model,
)
from regression.parameter_search import PARAMETER_CATALOG


# Carga variables locales antes de inicializar el diccionario de usuarios y la
# configuración de caducidad. Las variables ya definidas en el entorno prevalecen.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# Todas las salidas quedan ancladas al proyecto, no al directorio desde el que
# se haya lanzado Flask. Esto evita que un cambio de working directory disperse
# artefactos de una misma ejecución.
OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
# Diccionario didáctico usuario -> contraseña. Las contraseñas se leen del entorno
# y no se escriben en el repositorio. Añade aquí más pares para otros usuarios.
USERS = {
    "admin": os.environ.get("API_ADMIN_PASSWORD", ""),
}
# Los tokens Bearer caducan tras una hora por defecto.
TOKEN_MAX_AGE_SECONDS = max(1, int(os.environ.get("API_TOKEN_TTL_SECONDS", "3600")))


def create_app(output_root: Path = OUTPUT_ROOT, users: dict | None = None, token_secret: str | None = None) -> Flask:
    """Crea la aplicación Flask y registra todas las rutas.

    Los argumentos permiten indicar una carpeta de salida y credenciales
    distintas durante una ejecución local o en un entorno de despliegue.
    """
    app = Flask(__name__)
    output_root = Path(output_root)
    # API_TOKEN_SECRET debe ser estable y secreto en despliegues con varios
    # procesos. El secreto aleatorio permite arrancar el ejemplo localmente; en
    # ese caso los tokens dejan de ser válidos al reiniciar el proceso.
    token_secret = token_secret or os.environ.get("API_TOKEN_SECRET") or secrets.token_urlsafe(32)
    app.config["API_USERS"] = USERS if users is None else users
    app.config["API_TOKEN_MAX_AGE"] = TOKEN_MAX_AGE_SECONDS
    token_serializer = URLSafeTimedSerializer(token_secret, salt="regresion-polinomica-api-token")
    # Registro de auditoría central: incluye también login, health, errores 4xx
    # y rutas inexistentes, no solo las ejecuciones que crean un log por run.
    audit_dir = output_root / "_auditoria"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "api_audit.jsonl"
    audit_logger = logging.getLogger(
        "regression.api.audit." + hashlib.sha256(str(audit_path.resolve()).encode()).hexdigest()[:12]
    )
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    if not any(getattr(handler, "baseFilename", None) == str(audit_path.resolve())
               for handler in audit_logger.handlers):
        audit_handler = logging.FileHandler(audit_path, encoding="utf-8")
        audit_handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(audit_handler)
    # Flask rechaza cuerpos excesivos antes de parsearlos; el endpoint CSV aplica
    # además un máximo menor sobre el archivo concreto.
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

    def token_fingerprint(token: str) -> str:
        # Se registra solo una huella SHA-256, nunca el Bearer reutilizable.
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

    @app.before_request
    def capture_request_context():
        # Flask ejecuta esta función antes de cada ruta. Guardamos los datos de
        # auditoría en `g`, un espacio temporal disponible durante esta petición.
        authorization = request.headers.get("Authorization", "")
        scheme, separator, supplied_token = authorization.partition(" ")
        g.authenticated_user = None
        g.requested_username = None
        g.client_ip = request.remote_addr or "unknown"
        g.token_fingerprint = (
            token_fingerprint(supplied_token.strip())
            if separator and scheme.lower() == "bearer" and supplied_token.strip()
            else None
        )

    @app.after_request
    def write_audit_log(response):
        # JSONL mantiene una entrada estructurada y evita inyección de líneas en
        # los logs si usuario, ruta o cabeceras contienen caracteres especiales.
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "username": (getattr(g, "authenticated_user", None)
                         or getattr(g, "requested_username", None) or "anonymous"),
            "token_fingerprint": getattr(g, "token_fingerprint", None),
            "client_ip": getattr(g, "client_ip", None),
        }
        audit_logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error="El cuerpo supera el límite de 5 MB."), 413

    @app.post("/api/v1/auth/token")
    def issue_token():
        """Comprueba las credenciales y devuelve un token Bearer temporal."""
        # El endpoint de autenticación es público; solo emite un token cuando
        # el par usuario/contraseña coincide con el diccionario configurado.
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="El cuerpo debe ser un objeto JSON."), 400
        username = body.get("username")
        password = body.get("password")
        if isinstance(username, str):
            g.requested_username = username
        if not isinstance(username, str) or not isinstance(password, str):
            return jsonify(error="username y password son obligatorios."), 400
        expected_password = app.config["API_USERS"].get(username)
        # compare_digest reduce diferencias de tiempo al comparar contraseñas.
        valid = (
            isinstance(expected_password, str)
            and bool(expected_password)
            and hmac.compare_digest(expected_password.encode("utf-8"), password.encode("utf-8"))
        )
        if not valid:
            return jsonify(error="Credenciales no válidas."), 401
        token = token_serializer.dumps({"sub": username})
        # La llamada de login también queda asociada a usuario y huella del
        # token recién emitido, sin registrar el valor secreto del token.
        g.authenticated_user = username
        g.token_fingerprint = token_fingerprint(token)
        return jsonify(
            access_token=token,
            token_type="Bearer",
            expires_in=app.config["API_TOKEN_MAX_AGE"],
        ), 200

    def require_token(view):
        # Protege las rutas de entrenamiento sin duplicar la validación del token.
        @wraps(view)
        def wrapped(*args, **kwargs):
            # El decorador deja que cada endpoint se concentre en su modelo;
            # aquí comprobamos una sola vez que quien llama está autenticado.
            authorization = request.headers.get("Authorization", "")
            scheme, separator, token = authorization.partition(" ")
            if not separator or scheme.lower() != "bearer" or not token.strip():
                response = jsonify(error="Se requiere Authorization: Bearer <token>.")
                response.status_code = 401
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
            try:
                claims = token_serializer.loads(
                    token.strip(), max_age=app.config["API_TOKEN_MAX_AGE"]
                )
            except BadSignature:
                response = jsonify(error="Token inválido o caducado.")
                response.status_code = 401
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
            username = claims.get("sub") if isinstance(claims, dict) else None
            current_password = app.config["API_USERS"].get(username) if isinstance(username, str) else None
            if not isinstance(current_password, str) or not current_password:
                response = jsonify(error="Token inválido o caducado.")
                response.status_code = 401
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
            g.authenticated_user = username
            return view(*args, **kwargs)
        return wrapped

    @app.get("/api/v1/health")
    def health():
        """Ruta pública de comprobación: responde si Flask está disponible."""
        # Esta ruta pública sirve para comprobar que el servidor está activo.
        return jsonify(status="ok", service="regresion-polinomica-api", version="1.0.0")

    @app.get("/api/v1/regresion/parametros")
    def list_regression_parameters():
        """Publica las opciones discretas que usan las búsquedas de modelos."""
        return jsonify(
            models=PARAMETER_CATALOG.as_items(),
            search_defaults={"n_iter": 10, "cv": 5, "scoring": "neg_mean_squared_error"},
        )

    @app.post("/api/v1/regresion/polinomica")
    @require_token
    def fit_json():
        """POST JSON: ajusta un polinomio con una lista de puntos x/y.

        Respuesta 200: métricas y rutas de artefactos. Respuesta 400: campos o
        puntos inválidos. Requiere token Bearer.
        """
        # Este endpoint recibe puntos x/y en JSON; el ajuste se delega a run_model.
        # `silent=True` permite devolver un 400 uniforme para JSON mal formado,
        # en lugar de que el detalle interno del parser llegue al cliente.
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="El cuerpo debe ser un objeto JSON."), 400
        reference, error = validate_reference(body.get("reference"))
        if error:
            return jsonify(error=error), 400
        # El grado controla la complejidad y estabilidad del ajuste. Se limita
        # antes de construir el modelo para evitar modelos excesivamente flexibles.
        degree = body.get("degree", 2)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1 or degree > 10:
            return jsonify(error="degree debe ser un entero entre 1 y 10."), 400
        points = body.get("data")
        try:
            x, y = parse_json_points(points)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return run_model(app, output_root, reference, degree, x, y, "json")

    @app.post("/api/v1/regresion/polinomica/csv")
    @require_token
    def fit_csv():
        """POST multipart: ajusta un polinomio con columnas CSV x e y.

        El fichero se adjunta en el campo `file`; degree y reference son campos
        de texto. La respuesta 200 enlaza métricas y artefactos del ajuste.
        """
        # Esta variante recibe los mismos datos en un archivo CSV multipart.
        # Multipart separa los parámetros de control del fichero binario y es el
        # formato que Postman usa para adjuntar CSV sin codificarlo en JSON.
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        try:
            degree = int(request.form.get("degree", "2"))
        except (TypeError, ValueError):
            return jsonify(error="degree debe ser un entero entre 1 y 10."), 400
        if not 1 <= degree <= 10:
            return jsonify(error="degree debe ser un entero entre 1 y 10."), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Falta el archivo CSV en el campo multipart 'file'."), 400
        # Se lee una sola vez y se controla el tamaño antes de decodificarlo; luego
        # se conserva el original recibido como rastro reproducible de la entrada.
        raw = uploaded.read()
        if len(raw) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413
        try:
            text = raw.decode("utf-8-sig")
            x, y = parse_csv(text)
        except (UnicodeDecodeError, ValueError) as exc:
            return jsonify(error=f"CSV no válido: {exc}"), 400
        return run_model(app, output_root, reference, degree, x, y, "csv", raw_csv=raw)

    @app.post("/api/v1/regresion/polinomica/randomizedsearch")
    @require_token
    def fit_polynomial_randomized_search():
        """POST JSON o multipart: selecciona el grado con validación cruzada.

        JSON lleva una lista `data` de puntos x/y; multipart lleva `file`. Los
        campos n_iter, cv y random_state controlan la búsqueda reproducible.
        """
        raw_csv = None
        # El mismo endpoint admite dos formatos: JSON lleva una lista de puntos;
        # multipart/form-data lleva los parámetros en form y el archivo en file.
        if request.is_json:
            source = request.get_json(silent=True)
            if not isinstance(source, dict):
                return jsonify(error="El cuerpo debe ser un objeto JSON."), 400
            try:
                x, y = parse_json_points(source.get("data"))
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
        else:
            source = request.form
            uploaded = request.files.get("file")
            if uploaded is None or not uploaded.filename:
                return jsonify(error="Falta el archivo CSV en el campo multipart 'file'."), 400
            raw_csv = uploaded.read()
            if len(raw_csv) > 4 * 1024 * 1024:
                return jsonify(error="El CSV supera el límite de 4 MB."), 413
            try:
                x, y = parse_csv(raw_csv.decode("utf-8-sig"))
            except (UnicodeDecodeError, ValueError) as exc:
                return jsonify(error=f"CSV no válido: {exc}"), 400

        reference, error = validate_reference(source.get("reference"))
        if error:
            return jsonify(error=error), 400

        try:
            n_iter = parse_integer("n_iter", 20, 1, 80,source)
            cv = parse_integer("cv", 5, 2, 10,source)
            random_state = parse_integer("random_state", 42, 0, 2**32 - 1,source)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if cv >= int(len(x) * 0.8):
            return jsonify(error="cv debe ser menor que el número de observaciones de entrenamiento (80% de data)."), 400
        # Los pesos siguen el mismo orden que las filas de entrada. El ejecutor
        # recorta este vector junto con los datos en cada partición train/test.
        weights = source.get("w")
        if weights is not None:
            if isinstance(weights, str):
                try:
                    weights = json.loads(weights)
                except json.JSONDecodeError:
                    return jsonify(error="w debe ser una lista JSON de pesos numéricos."), 400
            try:
                weights = np.asarray(weights, dtype=float)
            except (TypeError, ValueError):
                return jsonify(error="w debe ser una lista numérica con un peso por observación."), 400
            if weights.ndim != 1 or len(weights) != len(x) or not np.isfinite(weights).all() or np.any(weights < 0):
                return jsonify(error="w debe contener pesos finitos no negativos, uno por observación."), 400
        options = {"n_iter": n_iter, "cv": cv, "random_state": random_state}
        return run_polynomial_search(output_root, reference, x, y, options, weights=weights, raw_csv=raw_csv)


    @app.post("/api/v1/random-forest/terrazas")
    @require_token
    def fit_random_forest_terraces():
        """POST JSON: ejecuta Random Forest sobre los datos sintéticos incluidos.

        El cliente envía target, reference y opciones del bosque; no adjunta
        datos. La respuesta incluye métricas y archivos creados para el run.
        """
        # Usa el CSV de ejemplo del proyecto; el cliente solo envía opciones y referencia.
        body = request.get_json(silent=True)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return jsonify(error="El cuerpo debe ser un objeto JSON."), 400
        reference, error = validate_reference(body.get("reference"))
        if error:
            return jsonify(error=error), 400
        try:
            options = random_forest_options(body)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        sample_path = Path(__file__).resolve().parent / "ejemplos" / "datos_ejemplo_complejo.csv"
        if not sample_path.is_file():
            return jsonify(error="No está disponible el CSV de ejemplo complejo."), 500
        return run_random_forest(app, output_root, reference, sample_path.read_bytes(), options,
                                 "datos_ejemplo_complejo.csv")

    @app.post("/api/v1/random-forest/csv")
    @require_token
    def fit_random_forest_csv():
        """POST multipart: ejecuta Random Forest sobre un CSV del cliente."""
        # Ruta multipart para entrenar con un CSV del cliente usando el mismo pipeline.
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        try:
            options = random_forest_options(request.form)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Falta el archivo CSV en el campo multipart 'file'."), 400
        raw = uploaded.read()
        if len(raw) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413
        return run_random_forest(app, output_root, reference, raw, options, uploaded.filename)

    def fit_knn_csv(method):
        """Valida el formulario común y ejecuta el tipo de validación KNN pedido."""
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Adjunta el CSV en el campo multipart 'file'."), 400
        raw_csv = uploaded.read()
        if len(raw_csv) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413

        try:
            options = {
                "target": request.form.get("target", "y"),
                "n_neighbors": parse_integer("n_neighbors", 5, 1, 50, request.form),
                "random_state": parse_integer("random_state", 42, 0, 2**32 - 1, request.form),
                "cv": parse_integer("cv", 5, 2, 10, request.form),
                **parse_search_options(request.form),
            }
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if not options["target"].strip():
            return jsonify(error="target debe ser el nombre de una columna."), 400
        return run_knn_cross_validation(output_root, reference, raw_csv, options, method)

    @app.post("/api/v1/regresion/knn/kfold")
    @require_token
    def fit_knn_kfold():
        """POST multipart: obtiene predicciones de validación cruzada K-Fold."""
        # Esta ruta agrupa las filas en K partes y repite el ajuste K veces.
        return fit_knn_csv("kfold")

    @app.post("/api/v1/regresion/knn/leave-one-out")
    @require_token
    def fit_knn_leave_one_out():
        """POST multipart: obtiene predicciones Leave-One-Out para KNN."""
        # Esta ruta deja una fila fuera en cada turno y la usa como validación.
        return fit_knn_csv("leave_one_out")

    def fit_tabular_regressor_csv(model_name):
        """Prepara la petición CSV compartida por SVR y los dos modelos boosting.

        Una sola función evita repetir validaciones idénticas. Las rutas
        concretas solo eligen model_name; las opciones y el archivo viajan en
        multipart/form-data y luego se delegan a api_functions.py.
        """
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        try:
            options = regression_model_options(request.form, model_name)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Adjunta el CSV en el campo multipart 'file'."), 400
        raw_csv = uploaded.read()
        if len(raw_csv) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413
        return run_tabular_regressor(app, output_root, reference, raw_csv,
                                     options, model_name)

    @app.post("/api/v1/regresion/svm/csv")
    @require_token
    def fit_svm_csv():
        """POST multipart: ajusta SVR; recibe CSV, target y opciones del modelo."""
        return fit_tabular_regressor_csv("svm")

    @app.post("/api/v1/regresion/boosting/adaboost/csv")
    @require_token
    def fit_adaboost_csv():
        """POST multipart: ajusta AdaBoostRegressor al CSV aportado."""
        return fit_tabular_regressor_csv("adaboost")

    @app.post("/api/v1/regresion/boosting/gradient/csv")
    @require_token
    def fit_gradient_boosting_csv():
        """POST multipart: ajusta GradientBoostingRegressor al CSV aportado."""
        return fit_tabular_regressor_csv("gradient_boosting")

    @app.post("/api/v1/eda/preprocesado/csv")
    @require_token
    def eda_preprocessing_csv():
        """POST multipart: genera informe EDA y después entrena SVR con datos transformados."""
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Adjunta el CSV en el campo multipart 'file'."), 400
        raw_csv = uploaded.read()
        if len(raw_csv) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413
        return run_eda_preprocesado(app, output_root, reference, raw_csv, uploaded.filename)

    @app.post("/api/v1/primitiva/analizar/csv")
    @require_token
    def analyze_primitiva_csv():
        """Ejecuta uno de los analizadores fijos de Primitiva sobre un CSV subido."""
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        version = request.form.get("version", "v02").strip().lower()
        if version not in {"v01", "v02"}:
            return jsonify(error="version debe ser 'v01' o 'v02'."), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Adjunta el CSV histórico en el campo multipart 'file'."), 400
        if Path(uploaded.filename).suffix.lower() != ".csv":
            return jsonify(error="El archivo debe tener extensión .csv."), 400
        raw_csv = uploaded.read()
        if len(raw_csv) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413
        return run_primitiva_analysis(output_root, reference, raw_csv, version)

    def fit_primitiva_model_csv(model_name):
        """Valida el formulario común y ejecuta el clasificador solicitado."""
        reference, error = validate_reference(request.form.get("reference"))
        if error:
            return jsonify(error=error), 400
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Adjunta el CSV histórico en el campo multipart 'file'."), 400
        if Path(uploaded.filename).suffix.lower() != ".csv":
            return jsonify(error="El archivo debe tener extensión .csv."), 400
        raw_csv = uploaded.read()
        if len(raw_csv) > 4 * 1024 * 1024:
            return jsonify(error="El CSV supera el límite de 4 MB."), 413
        try:
            options = {
                "window": parse_integer("window", 10, 3, 30, request.form),
                "random_state": parse_integer("random_state", 42, 0, 2**32 - 1, request.form),
            }
            if model_name == "knn":
                options["neighbors"] = parse_integer("neighbors", 5, 1, 25, request.form)
            else:
                options["epochs"] = parse_integer("epochs", 80, 10, 300, request.form)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return run_primitiva_model(output_root, reference, raw_csv, model_name, options)

    @app.post("/api/v1/primitiva/knn/csv")
    @require_token
    def analyze_primitiva_knn_csv():
        """Analiza el histórico con KNN multietiqueta y validación cronológica."""
        return fit_primitiva_model_csv("knn")

    @app.post("/api/v1/primitiva/cnn/csv")
    @require_token
    def analyze_primitiva_cnn_csv():
        """Analiza el histórico con una CNN 1D didáctica y validación cronológica."""
        return fit_primitiva_model_csv("cnn")

    return app
















app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
