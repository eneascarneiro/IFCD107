"""EDA y preprocesado reproducible del CSV sintético de terrazas.

Ejecutar desde cualquier directorio: python eda_preprocesado.py
Las salidas se guardan junto a este script, en resultados/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ejemplos" / "datos_ejemplo_complejo.csv"
OUT = Path(__file__).resolve().parent / "resultados"
TARGET = "y"
ID_COLUMN = "expediente_id"
RANDOM_STATE = 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "graficas"
    figures.mkdir(exist_ok=True)
    df = pd.read_csv(SOURCE)

    # Normalización mínima y validación de tipos; el identificador no se usa
    # como predictor. No se eliminan extremos automáticamente.
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    for col in df.columns.difference([ID_COLUMN]):
        df[col] = pd.to_numeric(df[col], errors="ignore")

    feature_columns = [c for c in df.columns if c not in (TARGET, ID_COLUMN)]
    X = df[feature_columns]
    y = pd.to_numeric(df[TARGET], errors="raise")
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    # Estadísticas descriptivas y control de calidad.
    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique_count": df.nunique(dropna=True),
    })
    profile.to_csv(OUT / "perfil_columnas.csv", encoding="utf-8-sig")
    df.describe(include="all").transpose().to_csv(OUT / "estadisticas_descriptivas.csv", encoding="utf-8-sig")
    df.to_csv(OUT / "datos_limpios.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    num_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlaciones entre variables numéricas")
    fig.tight_layout()
    fig.savefig(figures / "correlaciones.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(y, kde=True, ax=axes[0], color="#2878B5")
    axes[0].set(title="Distribución de y", xlabel="Días simulados")
    sns.scatterplot(data=df, x="x", y=TARGET, hue="tipo_tramite", alpha=.7, ax=axes[1])
    axes[1].set(title="Complejidad didáctica y días simulados")
    fig.tight_layout()
    fig.savefig(figures / "objetivo_y_relacion_x.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, max(1, len(numeric)), figsize=(4 * max(1, len(numeric)), 4))
    axes = [axes] if len(numeric) == 1 else list(axes)
    for ax, col in zip(axes, numeric):
        sns.boxplot(x=df[col], ax=ax, color="#70AD47")
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(figures / "diagramas_caja.png", dpi=160)
    plt.close(fig)

    # Partición antes de ajustar imputadores/codificadores para evitar fuga.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    transforms = []
    if numeric:
        transforms.append(("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), numeric))
    if categorical:
        transforms.append(("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    preprocessor = ColumnTransformer(transforms, remainder="drop", verbose_feature_names_out=False)
    train_encoded = preprocessor.fit_transform(X_train)
    test_encoded = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out().tolist()
    pd.DataFrame(train_encoded, columns=names, index=X_train.index).assign(**{TARGET: y_train}).sort_index().to_csv(
        OUT / "train_preprocesado.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(test_encoded, columns=names, index=X_test.index).assign(**{TARGET: y_test}).sort_index().to_csv(
        OUT / "test_preprocesado.csv", index=False, encoding="utf-8-sig"
    )
    import joblib
    joblib.dump({"preprocessor": preprocessor, "feature_columns": feature_columns,
                 "target": TARGET, "id_column": ID_COLUMN}, OUT / "preprocesador.joblib")

    numeric_desc = df[numeric + [TARGET]].describe().round(3).to_dict()
    summary = {
        "source": str(SOURCE.name), "synthetic_data": True,
        "rows_after_duplicate_removal": int(len(df)),
        "duplicate_rows_removed": int(pd.read_csv(SOURCE).duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "identifier": ID_COLUMN, "target": TARGET,
        "predictors": feature_columns, "numeric_predictors": numeric,
        "categorical_predictors": categorical,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "encoded_feature_count": int(len(names)), "random_state": RANDOM_STATE,
        "numeric_descriptives": numeric_desc,
    }
    (OUT / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# EDA y preprocesado: regresión de expedientes sintéticos\n
## Alcance\n
Se analizó `{SOURCE.relative_to(ROOT)}`. El README del proyecto indica que sus 1.000 registros son sintéticos: `x` representa un índice didáctico de complejidad y `y` días de tramitación simulados; no son datos ni resultados reales del Ayuntamiento.\n
## Calidad y variables\n
- Filas tras eliminar duplicados exactos: **{len(df)}** (duplicados encontrados: **{summary['duplicate_rows_removed']}**).\n+Celdas ausentes: **{summary['missing_cells']}**.\n+Identificador excluido de predictores: `{ID_COLUMN}`.\n+Objetivo: `{TARGET}`. Predictores: {', '.join(feature_columns)}.\n+Numéricos: {', '.join(numeric)}. Categóricos: {', '.join(categorical)}.\n
## Decisiones de preprocesado\n
Se reservó el 20% para test con `random_state={RANDOM_STATE}`. La imputación, el escalado numérico y la codificación one-hot se ajustan solo en train y luego se aplican a test. Numéricas: mediana y estandarización; categóricas: moda y one-hot con categorías desconocidas ignoradas. Los extremos se visualizan mediante cajas y no se eliminan automáticamente.\n
## Artefactos\n
- `perfil_columnas.csv`, `estadisticas_descriptivas.csv`, `datos_limpios.csv`\n+`train_preprocesado.csv`, `test_preprocesado.csv`, `preprocesador.joblib`\n+`graficas/correlaciones.png`, `graficas/objetivo_y_relacion_x.png`, `graficas/diagramas_caja.png`\n
Las figuras describen patrones del conjunto sintético y no implican causalidad.\n"""
    (OUT / "INFORME_EDA.md").write_text(report, encoding="utf-8")
    print(f"EDA y preprocesado completados: {OUT}")


if __name__ == "__main__":
    main()
