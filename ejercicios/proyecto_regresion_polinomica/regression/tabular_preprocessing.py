"""Preprocesamiento común para modelos tabulares de regresión."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_tabular_preprocessor(numeric_columns, categorical_columns, scale_numeric=False):
    """Construye transformaciones que se ajustan solo con el pliegue de train.

    Las columnas numéricas se imputan con la mediana y, cuando se solicita,
    se estandarizan. Las categóricas se imputan con la moda y se codifican
    mediante one-hot, ignorando categorías nuevas al predecir.
    """
    # ColumnTransformer aplica cada receta solo a las columnas que le indicamos.
    # La lista empieza vacía para que un CSV pueda tener solo números, solo
    # categorías o una mezcla de ambas.
    transformers = []
    if numeric_columns:
        # La mediana rellena huecos sin dejar que valores extremos dominen.
        # SVR usa distancias y se beneficia de que las escalas sean comparables.
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        # Primero rellenamos categorías vacías y luego creamos columnas 0/1.
        # handle_unknown evita fallar si test contiene una categoría nueva.
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("El CSV debe incluir al menos una característica utilizable.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
