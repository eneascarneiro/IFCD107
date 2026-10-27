"""Análisis exploratorio y preparación de datos para el proyecto LaTeX."""
from pathlib import Path
import json

import joblib
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
DATA = ROOT / "datos" / "datos_ejemplo_complejo.csv"
OUT = ROOT / "resultados"
TARGET = "y"
ID = "expediente_id"
SEED = 42


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "graficas").mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    df = raw.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if TARGET not in df or ID not in df:
        raise ValueError(f"El CSV debe contener las columnas {TARGET!r} y {ID!r}.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    # Perfil de calidad y estadísticas sin ocultar datos originales.
    profile = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "ausentes": df.isna().sum(),
        "ausentes_pct": (df.isna().mean() * 100).round(2),
        "valores_unicos": df.nunique(dropna=True),
    })
    profile.to_csv(OUT / "perfil_columnas.csv", encoding="utf-8-sig")
    df.describe(include="all").T.to_csv(OUT / "estadisticas_descriptivas.csv", encoding="utf-8-sig")
    df.to_csv(OUT / "datos_limpios.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    numeric_all = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric_all.corr(), annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlaciones entre variables numéricas")
    fig.tight_layout()
    fig.savefig(OUT / "graficas" / "correlaciones.png", dpi=160)
    plt.close(fig)

    y = df[TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(y, kde=True, ax=axes[0], color="#2878B5")
    axes[0].set(title="Distribución del objetivo y", xlabel="Días simulados")
    sns.scatterplot(data=df, x="x", y=TARGET, hue="tipo_tramite", alpha=.7, ax=axes[1])
    axes[1].set_title("Índice didáctico x frente al objetivo y")
    fig.tight_layout()
    fig.savefig(OUT / "graficas" / "distribucion_y_relacion.png", dpi=160)
    plt.close(fig)

    numeric_features = [c for c in df.select_dtypes(include="number") if c != TARGET]
    if numeric_features:
        fig, axes = plt.subplots(len(numeric_features), 1, figsize=(9, 3 * len(numeric_features)))
        axes = [axes] if len(numeric_features) == 1 else list(axes)
        for ax, col in zip(axes, numeric_features):
            sns.boxplot(data=df, x=col, ax=ax, color="#70AD47")
            ax.set_title(f"Diagrama de caja: {col}")
        fig.tight_layout()
        fig.savefig(OUT / "graficas" / "diagramas_caja.png", dpi=160)
        plt.close(fig)

    # No usar el identificador como predictor. Dividir antes de ajustar recetas.
    feature_columns = [col for col in df.columns if col not in (TARGET, ID)]
    X = df[feature_columns]
    y = df[TARGET]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    transformers = []
    if numeric:
        transformers.append(("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=False)
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out()
    for filename, matrix, targets in (
        ("train_preprocesado.csv", train_matrix, y_train),
        ("test_preprocesado.csv", test_matrix, y_test),
    ):
        result = pd.DataFrame(matrix, columns=names)
        result[TARGET] = targets.to_numpy()
        result.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    joblib.dump({"preprocessor": preprocessor, "feature_columns": feature_columns,
                 "target": TARGET, "id_column": ID}, OUT / "preprocesador.joblib")

    summary = {
        "filas_entrada": int(len(raw)), "filas_salida": int(len(df)),
        "duplicados_exactos_eliminados": exact_duplicates,
        "celdas_ausentes": int(df.isna().sum().sum()),
        "id_excluido": ID, "objetivo": TARGET,
        "predictores": feature_columns, "numericos": numeric, "categoricos": categorical,
        "filas_train": int(len(X_train)), "filas_test": int(len(X_test)),
        "columnas_codificadas": int(len(names)), "semilla": SEED,
        "nota": "Datos sintéticos de uso didáctico; no representan expedientes reales.",
    }
    (OUT / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Análisis terminado. Salidas: {OUT}")


if __name__ == "__main__":
    main()
