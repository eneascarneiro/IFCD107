"""Análisis exploratorio y preparación de datos para el proyecto LaTeX."""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
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

CASE_NOTES = {
    "datos_ejemplo_complejo": "CSV sintético original de referencia",
    "01_referencia_limpia": "Referencia limpia: distribuciones y relaciones base",
    "02_valores_ausentes": "Nulos en columnas numéricas, categóricas y objetivo",
    "03_valores_atipicos": "Valores extremos intencionados para investigar",
    "04_categorias_raras": "Categorías poco frecuentes, espacios y mayúsculas",
    "05_duplicados": "Duplicados exactos y claves repetidas con conflicto",
    "06_sesgo_y_heterocedasticidad": "Sesgo del predictor y varianza creciente en y",
}


def generate_check_pdf():
    """Crea un informe PDF multipágina con comprobaciones de cada CSV."""
    csv_files = sorted(DATA.parent.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {DATA.parent}")
    output_pdf = OUT / "informe_comprobacion_datos.pdf"
    sns.set_theme(style="whitegrid")
    overview = []
    datasets = []
    for path in csv_files:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        duplicate_rows = int(frame.duplicated().sum())
        duplicate_ids = (int(frame[ID].duplicated().sum()) if ID in frame else 0)
        missing_cells = int(frame.isna().sum().sum())
        numeric_cols = frame.select_dtypes(include="number").columns.tolist()
        outliers = {}
        for col in numeric_cols:
            values = frame[col].dropna()
            if len(values):
                q1, q3 = values.quantile([.25, .75])
                iqr = q3 - q1
                outliers[col] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        info = {
            "path": path, "frame": frame, "duplicate_rows": duplicate_rows,
            "duplicate_ids": duplicate_ids, "missing_cells": missing_cells,
            "outliers": outliers,
        }
        datasets.append(info)
        overview.append([
            path.name, len(frame), len(frame.columns), missing_cells,
            duplicate_rows, duplicate_ids, sum(outliers.values()),
        ])

    with PdfPages(output_pdf) as pdf:
        # Página de síntesis para comparar de un vistazo las casuísticas.
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0, 1.02, "Informe de comprobación de los CSV", fontsize=20,
                fontweight="bold", transform=ax.transAxes, va="bottom")
        ax.text(0, .97,
                "Datos sintéticos para prácticas de EDA. Los candidatos IQR requieren revisión contextual.",
                fontsize=10, transform=ax.transAxes, va="top")
        table = ax.table(cellText=overview,
                         colLabels=["Archivo", "Filas", "Cols.", "Nulos", "Duplic.", "ID repet.", "Atípicos IQR"],
                         cellLoc="center", colLoc="center", loc="upper center",
                         bbox=[0, .44, 1, .46],
                         colWidths=[.40, .08, .08, .09, .09, .10, .16])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#24476b")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
        notes = [
            "Lectura de comprobaciones:",
            "• Nulos: total de celdas vacías del archivo.",
            "• Duplic.: filas completamente repetidas; ID repet.: identificadores repetidos, aunque el resto cambie.",
            "• Atípicos IQR: conteo acumulado entre las columnas numéricas según 1,5 × IQR; no significa error automáticamente.",
            "• La hoja siguiente de cada CSV muestra faltantes por variable, distribución de y y relación x-y cuando están disponibles.",
            "• No extraer conclusiones sobre casos reales: todas las filas de práctica son inventadas.",
        ]
        ax.text(0, .36, "\n".join(notes), fontsize=10, linespacing=1.7,
                transform=ax.transAxes, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Una página de diagnóstico por archivo, con tabla y gráficos.
        for item in datasets:
            frame = item["frame"]
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.suptitle(item["path"].name, fontsize=16, fontweight="bold", y=.98)
            note = CASE_NOTES.get(item["path"].stem, "Conjunto incluido en la carpeta datos")
            fig.text(.08, .93, note, fontsize=10)
            fig.text(.08, .89,
                     f"Filas: {len(frame)} | columnas: {len(frame.columns)} | celdas ausentes: {item['missing_cells']} | "
                     f"duplicados exactos: {item['duplicate_rows']} | IDs repetidos: {item['duplicate_ids']}",
                     fontsize=9)

            profile_rows = []
            for col in frame.columns:
                series = frame[col]
                missing_pct = 100 * series.isna().mean()
                profile_rows.append([col, str(series.dtype), int(series.isna().sum()),
                                     f"{missing_pct:.1f}%", int(series.nunique(dropna=True)),
                                     item["outliers"].get(col, "—")])
            ax_table = fig.add_axes([.07, .51, .86, .31])
            ax_table.axis("off")
            profile_table = ax_table.table(
                cellText=profile_rows,
                colLabels=["Variable", "Tipo", "Nulos", "% nulos", "Únicos", "Atípicos IQR"],
                loc="center", cellLoc="center", colLoc="center",
            )
            profile_table.auto_set_font_size(False)
            profile_table.set_fontsize(8)
            profile_table.scale(1, 1.25)
            for (row, _), cell in profile_table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#24476b")
                    cell.get_text().set_color("white")
                    cell.get_text().set_weight("bold")

            ax_y = fig.add_axes([.09, .09, .36, .33])
            if TARGET in frame:
                target = pd.to_numeric(frame[TARGET], errors="coerce").dropna()
                if len(target):
                    sns.histplot(target, kde=len(target) > 2, ax=ax_y, color="#2878B5")
                    ax_y.set_title("Distribución del objetivo y")
                else:
                    ax_y.text(.5, .5, "y no contiene valores numéricos", ha="center", va="center")
                    ax_y.set_axis_off()
            else:
                ax_y.text(.5, .5, "No existe columna objetivo y", ha="center", va="center")
                ax_y.set_axis_off()

            ax_xy = fig.add_axes([.56, .09, .36, .33])
            if "x" in frame and TARGET in frame:
                plot_data = frame[["x", TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(plot_data):
                    sns.scatterplot(data=plot_data, x="x", y=TARGET, ax=ax_xy,
                                    alpha=.65, s=22, color="#e07a35")
                    ax_xy.set_title("Relación entre x e y")
                else:
                    ax_xy.text(.5, .5, "No hay pares x-y completos", ha="center", va="center")
                    ax_xy.set_axis_off()
            else:
                ax_xy.text(.5, .5, "No están disponibles x e y", ha="center", va="center")
                ax_xy.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return output_pdf


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

    check_pdf = generate_check_pdf()

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
    print(f"Informe PDF con datos a comprobar: {check_pdf}")


if __name__ == "__main__":
    main()
