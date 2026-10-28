"""KNN multietiqueta y una CNN 1D pequeña para el ejemplo de La Primitiva.

Los modelos aprenden a ordenar números usando sorteos pasados. Como los sorteos
son independientes, sus puntuaciones no representan ventajas reales de apuesta.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


NUMBER_COLUMNS = tuple(f"N{i}" for i in range(1, 7))
RANDOM_EXPECTED_HITS_AT_SIX = 36 / 49


def load_draw_matrix(raw_csv: bytes) -> np.ndarray:
    """Lee filas con seis números principales y devuelve una matriz 0/1 de 49 columnas."""
    try:
        text = raw_csv.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("El CSV debe estar codificado como UTF-8 o UTF-8 con BOM.") from None

    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        names = {part.strip().upper() for part in line.split(";")}
        if set(NUMBER_COLUMNS).issubset(names):
            header_index = index
            break
    if header_index is None:
        raise ValueError("No se encontró una cabecera CSV con las columnas N1, N2, ..., N6.")

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])), delimiter=";")
    records: list[tuple[datetime | None, tuple[int, ...], str | None]] = []
    seen_dates: set[str] = set()
    for row in reader:
        normalized = {(key or "").strip().upper(): value for key, value in row.items()}
        try:
            values = []
            for column in NUMBER_COLUMNS:
                raw = (normalized.get(column) or "").strip()
                numeric = float(raw.replace(",", "."))
                number = int(numeric)
                if numeric != number or not 1 <= number <= 49:
                    raise ValueError
                values.append(number)
            if len(set(values)) != 6:
                continue
        except (TypeError, ValueError):
            continue

        date_text = (normalized.get("FECHA") or "").strip()
        parsed_date = None
        if date_text:
            for date_format in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    parsed_date = datetime.strptime(date_text, date_format)
                    break
                except ValueError:
                    continue
        if date_text and date_text in seen_dates:
            continue
        if date_text:
            seen_dates.add(date_text)
        records.append((parsed_date, tuple(values), date_text or None))

    if len(records) < 1:
        raise ValueError("El CSV no contiene sorteos válidos con seis números distintos entre 1 y 49.")
    # Ordenamos por fecha únicamente si todas las filas tienen fechas válidas.
    if all(record[0] is not None for record in records):
        records.sort(key=lambda record: record[0])

    matrix = np.zeros((len(records), 49), dtype=np.float32)
    for row_index, (_, numbers, _) in enumerate(records):
        matrix[row_index, np.asarray(numbers, dtype=int) - 1] = 1.0
    return matrix


def make_sequences(draws: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Usa los últimos sorteos de cada ventana para predecir el sorteo siguiente."""
    if isinstance(window, bool) or not isinstance(window, int) or not 3 <= window <= 30:
        raise ValueError("window debe ser un entero entre 3 y 30.")
    if len(draws) < window + 6:
        raise ValueError(f"Se necesitan al menos {window + 6} sorteos válidos para window={window}.")
    features = np.stack([draws[end - window:end] for end in range(window, len(draws))])
    targets = draws[window:].copy()
    return features, targets


def _top_six(scores: np.ndarray) -> np.ndarray:
    """Devuelve los seis números con mayor puntuación, ordenados de menor a mayor."""
    return np.sort(np.argsort(-np.asarray(scores), kind="stable")[:6] + 1)


def _validation_metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    """Resume la validación temporal usando número de aciertos por sorteo."""
    hit_counts = []
    for predicted_scores, actual in zip(scores, truth):
        predicted = set(_top_six(predicted_scores).tolist())
        actual_numbers = set((np.flatnonzero(actual) + 1).tolist())
        hit_counts.append(len(predicted & actual_numbers))
    return {
        "validation_draws": int(len(hit_counts)),
        "mean_hits_at_6": float(np.mean(hit_counts)),
        "random_expected_hits_at_6": float(RANDOM_EXPECTED_HITS_AT_SIX),
        "exact_six_rate": float(np.mean(np.asarray(hit_counts) == 6)),
    }


def _knn_scores(model: KNeighborsClassifier, features: np.ndarray) -> np.ndarray:
    """Extrae la puntuación para la clase 1 de las 49 salidas del KNN."""
    probabilities = model.predict_proba(features)
    classes = model.classes_
    output = np.zeros((len(features), 49), dtype=np.float64)
    for number_index, (class_values, class_probabilities) in enumerate(zip(classes, probabilities)):
        positive_index = np.flatnonzero(np.asarray(class_values) == 1)
        if len(positive_index):
            output[:, number_index] = class_probabilities[:, positive_index[0]]
    return output


def _train_knn(
    train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray, neighbors: int
) -> np.ndarray:
    """Entrena el KNN multietiqueta y predice puntuaciones para cada número."""
    model = KNeighborsClassifier(
        n_neighbors=min(neighbors, len(train_x)),
        weights="distance",
        metric="hamming",
        n_jobs=1,
    )
    model.fit(train_x.reshape(len(train_x), -1), train_y.astype(np.int8))
    return _knn_scores(model, predict_x.reshape(len(predict_x), -1))


def _cnn_forward(features: np.ndarray, conv_weights: np.ndarray,
                 conv_bias: np.ndarray, dense_weights: np.ndarray,
                 dense_bias: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Calcula convolución temporal, ReLU, pooling medio y salida sigmoide."""
    batch_size, time_steps, _ = features.shape
    kernel_size, _, filter_count = conv_weights.shape
    output_steps = time_steps - kernel_size + 1
    convolution = np.zeros((batch_size, output_steps, filter_count), dtype=np.float64)
    for kernel_index in range(kernel_size):
        window = features[:, kernel_index:kernel_index + output_steps, :]
        convolution += np.einsum("bti,if->btf", window, conv_weights[kernel_index])
    convolution += conv_bias
    activation = np.maximum(convolution, 0.0)
    pooled = activation.mean(axis=1)
    logits = pooled @ dense_weights + dense_bias
    # El recorte evita overflow numérico en exp durante los primeros pasos.
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return scores, (convolution, activation, pooled, logits)


def _train_cnn(
    train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray,
    epochs: int, random_state: int,
) -> np.ndarray:
    """Entrena una CNN 1D pequeña con NumPy y optimización Adam."""
    rng = np.random.default_rng(random_state)
    kernel_size, filter_count, output_count = 3, 12, 49
    _, time_steps, input_count = train_x.shape
    conv_weights = rng.normal(0, np.sqrt(2 / (kernel_size * input_count)),
                              (kernel_size, input_count, filter_count))
    conv_bias = np.zeros(filter_count, dtype=np.float64)
    dense_weights = rng.normal(0, np.sqrt(2 / filter_count), (filter_count, output_count))
    dense_bias = np.zeros(output_count, dtype=np.float64)
    parameters = [conv_weights, conv_bias, dense_weights, dense_bias]
    first_moments = [np.zeros_like(value) for value in parameters]
    second_moments = [np.zeros_like(value) for value in parameters]
    beta_one, beta_two, learning_rate, epsilon = 0.9, 0.999, 0.01, 1e-8
    step = 0
    batch_size = min(16, len(train_x))

    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_x = train_x[indices].astype(np.float64)
            batch_y = train_y[indices].astype(np.float64)
            prediction, cache = _cnn_forward(
                batch_x, conv_weights, conv_bias, dense_weights, dense_bias
            )
            convolution, activation, pooled, _ = cache

            # Gradiente de entropía cruzada binaria respecto a los logits.
            grad_logits = (prediction - batch_y) / len(batch_x)
            grad_dense_weights = pooled.T @ grad_logits
            grad_dense_bias = grad_logits.sum(axis=0)
            grad_pooled = grad_logits @ dense_weights.T
            grad_activation = np.broadcast_to(
                grad_pooled[:, None, :] / activation.shape[1], activation.shape
            ).copy()
            grad_convolution = grad_activation * (convolution > 0)
            grad_conv_weights = np.zeros_like(conv_weights)
            for kernel_index in range(kernel_size):
                window = batch_x[:, kernel_index:kernel_index + activation.shape[1], :]
                grad_conv_weights[kernel_index] = np.einsum(
                    "bti,btf->if", window, grad_convolution
                )
            grad_conv_bias = grad_convolution.sum(axis=(0, 1))
            gradients = [grad_conv_weights, grad_conv_bias,
                         grad_dense_weights, grad_dense_bias]

            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first_moments[index] = beta_one * first_moments[index] + (1 - beta_one) * gradient
                second_moments[index] = beta_two * second_moments[index] + (1 - beta_two) * gradient**2
                corrected_first = first_moments[index] / (1 - beta_one**step)
                corrected_second = second_moments[index] / (1 - beta_two**step)
                parameter -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

    return _cnn_forward(
        predict_x.astype(np.float64), conv_weights, conv_bias, dense_weights, dense_bias
    )[0]


def analyze_draws(
    draws: np.ndarray, model_name: str, window: int = 10,
    neighbors: int = 5, epochs: int = 80, random_state: int = 42,
) -> dict:
    """Valida cronológicamente el modelo y calcula el ranking para el siguiente sorteo."""
    if model_name not in {"knn", "cnn"}:
        raise ValueError("model_name debe ser 'knn' o 'cnn'.")
    if isinstance(neighbors, bool) or not 1 <= int(neighbors) <= 25:
        raise ValueError("neighbors debe estar entre 1 y 25.")
    if isinstance(epochs, bool) or not 10 <= int(epochs) <= 300:
        raise ValueError("epochs debe estar entre 10 y 300.")

    features, targets = make_sequences(draws, int(window))
    if len(features) < 6:
        raise ValueError("Se necesitan más sorteos para reservar una validación temporal útil.")
    validation_count = max(1, int(np.ceil(len(features) * 0.2)))
    train_count = len(features) - validation_count
    train_x, validation_x = features[:train_count], features[train_count:]
    train_y, validation_y = targets[:train_count], targets[train_count:]

    if model_name == "knn":
        validation_scores = _train_knn(train_x, train_y, validation_x, int(neighbors))
        next_scores = _train_knn(features, targets, draws[-int(window):][None, :, :], int(neighbors))[0]
        parameters = {"neighbors": min(int(neighbors), train_count), "weights": "distance", "metric": "hamming"}
    else:
        validation_scores = _train_cnn(train_x, train_y, validation_x, int(epochs), int(random_state))
        next_scores = _train_cnn(features, targets, draws[-int(window):][None, :, :], int(epochs), int(random_state))[0]
        parameters = {"window": int(window), "kernel_size": 3, "filters": 12,
                      "epochs": int(epochs), "random_state": int(random_state)}

    ranking = sorted(
        ({"number": number, "score": float(next_scores[number - 1])} for number in range(1, 50)),
        key=lambda item: (-item["score"], item["number"]),
    )[:6]
    return {
        "algorithm": "KNeighborsClassifier" if model_name == "knn" else "CNN1D_Numpy",
        "historical_draws": int(len(draws)),
        "window": int(window),
        "suggested_numbers": [item["number"] for item in ranking],
        "ranking_scores": ranking,
        "parameters": parameters,
        "metrics": _validation_metrics(validation_scores, validation_y),
        "disclaimer": (
            "El sorteo es aleatorio e independiente. El ranking es didáctico; "
            "no aumenta la probabilidad teórica de acertar."
        ),
    }
