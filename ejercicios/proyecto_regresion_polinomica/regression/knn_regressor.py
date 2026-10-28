"""Herramientas sencillas para preparar un modelo KNN de regresión."""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn_pipeline(n_neighbors=5):
    """Crea el flujo de escalado y predicción usado en la validación cruzada.

    KNN compara distancias entre filas. StandardScaler pone cada predictor en
    una escala comparable para que las columnas grandes no dominen esas distancias.
    """
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors debe ser un entero positivo.")

    return Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=n_neighbors)),
    ])
