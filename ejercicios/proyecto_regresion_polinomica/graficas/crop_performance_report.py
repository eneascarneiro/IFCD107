import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from senfarming_ai_utils import display_func, display_func_val,calculate_r2_manual
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether
)

import json
import psycopg2

class CropPerformanceReport:
    """
    Analiza el rendimiento de un modelo de predicción SOC por categoría
    y genera un informe completo en PDF.

    X:
        DataFrame ORIGINAL.
        Contiene las etiquetas textuales:
            lc0_desc
            lc1_desc
            lu1_desc

    X_preprocessing:
        DataFrame utilizado por el modelo después del preprocessing.
        Puede contener las variables categóricas transformadas a números.

    usepreprocessing:
        0 -> utiliza X para el análisis numérico.
        1 -> utiliza X_preprocessing para los cálculos/modelo.

        IMPORTANTE:
        Las categorías SIEMPRE se obtienen de X original.

    y_true:
        Valores observados.

    y_pred:
        Predicciones OOF.

    output_path:
        Directorio donde se generan imágenes y PDF.

    model_name:
        Nombre del modelo.
    """

    def __init__(
        self,
        X,
        X_preprocessing,
        y_true,
        y_pred,
        output_path,
        model_name="Model",
        file_model="FileModel",
        usepreprocessing=0,
        crop_columns=None,
        min_samples=5,
        generate_shap=False,
        pipeline=None
    ):

        self.X = X.copy()

        self.X_preprocessing = (
            X_preprocessing.copy()
            if X_preprocessing is not None
            else None
        )

        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()

        self.output_path = output_path
        self.model_name = model_name
        self.file_model = file_model
        self.usepreprocessing = usepreprocessing
        self.min_samples = min_samples

        self.generate_shap = generate_shap
        self.pipeline = pipeline

        if crop_columns is None:
            crop_columns = [
                "lc0_desc",
                "lc1_desc",
                "lu1_desc"
            ]

        self.crop_columns = [
            c for c in crop_columns
            if c in self.X.columns
        ]

        os.makedirs(self.output_path, exist_ok=True)

        self.images = []
        self.tables = {}
        self.crop_results = {}

        self._validate()

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate(self):

        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.usepreprocessing == 1:

            if self.X_preprocessing is None:
                raise ValueError(
                    "X_preprocessing cannot be None when "
                    "usepreprocessing=1"
                )

            if len(self.X_preprocessing) != len(self.X):
                raise ValueError(
                    "X and X_preprocessing must have the same number "
                    "of rows"
                )

        if len(self.y_true) != len(self.X):
            raise ValueError(
                "X and y_true must have the same number of rows"
            )

        if len(self.y_pred) != len(self.X):
            raise ValueError(
                "X and y_pred must have the same number of rows"
            )

        if len(self.crop_columns) == 0:
            raise ValueError(
                "No crop categorical columns found"
            )

    # ============================================================
    # DATA USED FOR ANALYSIS
    # ============================================================

    def get_analysis_data(self):

        if self.usepreprocessing == 1:

            X_numeric = self.X_preprocessing.copy()

        else:

            X_numeric = self.X.copy()

        df = X_numeric.copy()

        # Always take categorical labels from original X
        for col in self.crop_columns:
            df[col] = self.X[col].values

        df["y_true"] = self.y_true
        df["y_pred"] = self.y_pred
        df["residual"] = (
            df["y_true"] - df["y_pred"]
        )

        return df

    # ============================================================
    # METRICS
    # ============================================================

    @staticmethod
    def calculate_metrics(y_true, y_pred):

        n = len(y_true)

        if n == 0:
            return {
                "n": 0,
                "r2": np.nan,
                "r2man": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "pearson": np.nan,
                "bias": np.nan
            }

        if n >= 2 and np.std(y_true) > 0:

            r2 = r2_score(
                y_true,
                y_pred
            )
            display_func_val(1,"crop calculate_metrics r2:",str(r2))
            r2man = calculate_r2_manual(
                y_true, 
                y_pred)
            display_func_val(1,"crop calculate_metrics r2:",str(r2))

        else:

            r2 = np.nan
            r2man = np.nan

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        # Pearson correlation
        if (
            n >= 2
            and np.std(y_true) > 0
            and np.std(y_pred) > 0
        ):
            pearson = np.corrcoef(
                y_true,
                y_pred
            )[0, 1]
        else:
            pearson = np.nan
            
        bias = np.mean(
            y_pred - y_true
        )

        return {
            "n": n,
            "r2": r2,
            "r2man": r2man,
            "rmse": rmse,
            "mae": mae,
            "pearson": pearson,
            "bias": bias
        }

    # ============================================================
    # ANALYSIS BY CATEGORY
    # ============================================================

    def calculate_category_metrics(self, column):

        df = self.get_analysis_data()

        rows = []

        for category, group in df.groupby(
            column,
            dropna=False
        ):

            if len(group) < self.min_samples:
                continue

            metrics = self.calculate_metrics(
                group["y_true"].values,
                group["y_pred"].values
            )

            metrics[column] = (
                "NA"
                if pd.isna(category)
                else str(category)
            )

            rows.append(metrics)

        if len(rows) == 0:

            return pd.DataFrame(
                columns=[
                    column,
                    "n",
                    "r2",
                    "r2man",
                    "rmse",
                    "mae",
                    "pearson",
                    "bias"
                ]
            )

        result = pd.DataFrame(rows)

        result = result[
            [
                column,
                "n",
                "r2",
                "r2man",
                "rmse",
                "mae",
                "pearson",
                "bias"
            ]
        ]

        result = result.sort_values(
            "r2",
            ascending=False
        )

        return result.reset_index(drop=True)

    # ============================================================
    # ALL CROP RESULTS
    # ============================================================

    def calculate_all_metrics(self):

        self.tables = {}

        for column in self.crop_columns:
            display_func_val(1,"calculate_all_metrics col:",column)
            self.tables[column] = (
                self.calculate_category_metrics(
                    column
                )
            )

        self.crop_results = self.tables

        return self.tables

    # ============================================================
    # GLOBAL MODEL METRICS
    # ============================================================

    def global_metrics(self):

        return self.calculate_metrics(
            self.y_true,
            self.y_pred
        )

    # ============================================================
    # SAVE TABLES
    # ============================================================

    def save_tables(self):

        for column, df in self.tables.items():

            filename = os.path.join(
                self.output_path,
                f"{self.model_name}_{column}_metrics.csv"
            )

            df.to_csv(
                filename,
                index=False
            )

    # ============================================================
    # RESIDUAL BOXPLOT
    # ============================================================

    def plot_residual_boxplot(self, column):

        df = self.get_analysis_data()

        categories = []

        for category, group in df.groupby(
            column,
            dropna=False
        ):

            if len(group) >= self.min_samples:

                categories.append(
                    (
                        str(category),
                        group["residual"].values
                    )
                )

        if len(categories) == 0:
            return None

        categories = sorted(
            categories,
            key=lambda x: len(x[1]),
            reverse=True
        )

        labels = [
            x[0] for x in categories
        ]

        residuals = [
            x[1] for x in categories
        ]

        fig_width = max(
            8,
            len(labels) * 0.7
        )

        plt.figure(
            figsize=(fig_width, 6)
        )

        plt.boxplot(
            residuals,
            labels=labels
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel(column)
        plt.ylabel(
            "Residual (Observed - Predicted)"
        )

        plt.title(
            f"{self.model_name} - "
            f"Residuals by {column}"
        )

        plt.xticks(
            rotation=60,
            ha="right"
        )

        plt.tight_layout()

        filename = os.path.join(
            self.output_path,
            f"{self.model_name}_{column}_residual_boxplot.png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        self.images.append(filename)

        return filename

    # ============================================================
    # SCATTER BY CATEGORY
    # ============================================================

    def plot_scatter_categories(self, column):

        df = self.get_analysis_data()

        categories = []

        for category, group in df.groupby(
            column,
            dropna=False
        ):

            if len(group) >= self.min_samples:

                categories.append(
                    (
                        str(category),
                        group.copy()
                    )
                )

        if len(categories) == 0:
            return []

        output_files = []

        for category, group in categories:

            plt.figure(
                figsize=(6, 6)
            )

            plt.scatter(
                group["y_true"],
                group["y_pred"],
                alpha=0.65
            )

            min_value = min(
                group["y_true"].min(),
                group["y_pred"].min()
            )

            max_value = max(
                group["y_true"].max(),
                group["y_pred"].max()
            )

            plt.plot(
                [min_value, max_value],
                [min_value, max_value],
                linestyle="--",
                label="1:1"
            )

            metrics = self.calculate_metrics(
                group["y_true"].values,
                group["y_pred"].values
            )

            plt.xlabel("Observed SOC")
            plt.ylabel("Predicted SOC")

            plt.title(
                f"{self.model_name}\n"
                f"{column}: {category}\n"
                f"N={metrics['n']}  "
                f"R²={metrics['r2']:.3f}  "
                f"RMSE={metrics['rmse']:.3f}"
            )

            plt.legend()

            plt.tight_layout()

            safe_category = (
                str(category)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            filename = os.path.join(
                self.output_path,
                f"{self.model_name}_"
                f"{column}_{safe_category}_scatter.png"
            )

            plt.savefig(
                filename,
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

            self.images.append(filename)
            output_files.append(filename)

        return output_files

    # ============================================================
    # RANKINGS
    # ============================================================

    def get_rankings(self, column):

        df = self.tables[column].copy()

        best = df.sort_values(
            "r2",
            ascending=False
        ).reset_index(drop=True)

        worst = df.sort_values(
            "r2",
            ascending=True
        ).reset_index(drop=True)

        return {
            "best": best,
            "worst": worst
        }

    # ============================================================
    # HEATMAP
    # ============================================================

    def create_r2_heatmap(
        self,
        model_crop_results,
        crop_column="lc0_desc"
    ):

        rows = []

        for model_name, results in model_crop_results.items():

            if crop_column not in results:
                continue

            df = results[crop_column].copy()

            for _, row in df.iterrows():

                rows.append(
                    {
                        "model": model_name,
                        "crop": row[crop_column],
                        "r2": row["r2"]
                    }
                )

        if len(rows) == 0:
            return None

        heatmap_df = pd.DataFrame(rows)

        pivot = heatmap_df.pivot_table(
            index="model",
            columns="crop",
            values="r2"
        )

        plt.figure(
            figsize=(
                max(8, pivot.shape[1] * 0.8),
                max(5, pivot.shape[0] * 0.7)
            )
        )

        plt.imshow(
            pivot.values,
            aspect="auto"
        )

        plt.colorbar(
            label="R²"
        )

        plt.xticks(
            range(len(pivot.columns)),
            pivot.columns,
            rotation=60,
            ha="right"
        )

        plt.yticks(
            range(len(pivot.index)),
            pivot.index
        )

        plt.xlabel("Crop / Land-use category")
        plt.ylabel("Model")

        plt.title(
            f"R² - Model × {crop_column}"
        )

        plt.tight_layout()

        filename = os.path.join(
            self.output_path,
            f"model_crop_r2_heatmap_{crop_column}.png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        self.images.append(filename)

        return filename

    # ============================================================
    # PDF UTILITIES
    # ============================================================

    def _format_value(self, value):

        if pd.isna(value):
            return ""

        if isinstance(value, (float, np.floating)):

            return f"{value:.4f}"

        return str(value)

    def dataframe_to_table(
        self,
        df,
        title=None,
        max_rows=None
    ):

        elements = []

        if title:

            elements.append(
                Paragraph(
                    title,
                    self.styles["Heading2"]
                )
            )

            elements.append(
                Spacer(1, 0.15 * cm)
            )

        if df is None or df.empty:

            elements.append(
                Paragraph(
                    "No data available.",
                    self.styles["Normal"]
                )
            )

            return elements

        display_df = df.copy()

        if max_rows is not None:

            display_df = display_df.head(
                max_rows
            )

        data = [
            [
                str(column)
                for column in display_df.columns
            ]
        ]

        for _, row in display_df.iterrows():

            data.append(
                [
                    self._format_value(value)
                    for value in row
                ]
            )

        page_width = landscape(
            A4
        )[0] - 2 * cm

        col_width = (
            page_width /
            len(data[0])
        )

        table = Table(
            data,
            repeatRows=1,
            colWidths=[
                col_width
            ] * len(data[0])
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#D9EAF7"
                        )
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.black
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.3,
                        colors.grey
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "RIGHT"
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor(
                                "#F5F5F5"
                            )
                        ]
                    )
                ]
            )
        )

        elements.append(table)

        return elements

    # ============================================================
    # IMAGE INTO PDF
    # ============================================================

    def image_for_pdf(
        self,
        filename,
        width=17 * cm
    ):

        if not os.path.exists(filename):
            return []

        elements = []

        elements.append(
            Spacer(1, 0.2 * cm)
        )

        img = Image(
            filename
        )

        ratio = (
            img.imageHeight /
            float(img.imageWidth)
        )

        height = width * ratio

        max_height = 22 * cm

        if height > max_height:

            height = max_height

            width = (
                height /
                ratio
            )

        img.drawWidth = width
        img.drawHeight = height

        elements.append(img)

        elements.append(
            Spacer(1, 0.3 * cm)
        )

        elements.append(
            Paragraph(
                os.path.basename(filename),
                self.styles["Caption"]
            )
        )

        return elements

    # ============================================================
    # SAVE RESULTS TO POSTGRESQL
    # ============================================================

    def save_results_to_db(self):
        # first load config from a json file,
        srvconf = json.load(open(os.environ["PYSRV_CONFIG_PATH"]))
        conn = psycopg2.connect(user= srvconf['PYSRV_DATABASE_USER'],
                                                password=srvconf['PYSRV_DATABASE_PASSWORD'],
                                                host= srvconf['PYSRV_DATABASE_HOST_POSTGRESQL'],
                                                port=srvconf['PYSRV_DATABASE_PORT'],
                                                database= srvconf['PYSRV_DATABASE_NAME'])

        if not self.crop_results:
            raise ValueError(
                "No crop results available. "
                "Call calculate_all_metrics() first."
            )

        cursor = conn.cursor()

        sql = """
            INSERT INTO crop_model_results (
                model_name,
                crop_field,
                crop_name,
                r2,
                r2man,
                rmse,
                mae,
                pearson,
                bias
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,%s
            )
            ON CONFLICT (
                model_name,
                crop_field,
                crop_name
            )
            DO UPDATE SET
                r2      = EXCLUDED.r2,
                r2man      = EXCLUDED.r2,
                rmse    = EXCLUDED.rmse,
                mae     = EXCLUDED.mae,
                pearson = EXCLUDED.pearson,
                bias    = EXCLUDED.bias
        """

        rows = []

        for crop_field, df in self.crop_results.items():

            for _, row in df.iterrows():

                crop_name = row[crop_field]

                # Convert NaN to None for PostgreSQL
                def db_value(value):
                    if pd.isna(value):
                        return None
                    return float(value)

                rows.append(
                    (
                        self.file_model,
                        crop_field,
                        str(crop_name),
                        db_value(row["r2"]),
                        db_value(row["r2man"]),
                        db_value(row["rmse"]),
                        db_value(row["mae"]),
                        db_value(row["pearson"]),
                        db_value(row["bias"])
                    )
                )

        if rows:
            cursor.executemany(
                sql,
                rows
            )

        conn.commit()

        cursor.close()

        return len(rows)

    # ============================================================
    # PDF
    # ============================================================

    def create_pdf(
        self,
        pdf_filename=None,
        model_crop_results=None,
        heatmap_column="lc0_desc"
    ):

        if pdf_filename is None:

            pdf_filename = os.path.join(
                self.output_path,
                f"{self.model_name}_"
                f"CropPerformanceReport.pdf"
            )

        doc = SimpleDocTemplate(
            pdf_filename,
            pagesize=landscape(A4),
            rightMargin=1.2 * cm,
            leftMargin=1.2 * cm,
            topMargin=1.2 * cm,
            bottomMargin=1.2 * cm
        )

        self.styles = getSampleStyleSheet()

        self.styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=self.styles["Title"],
                alignment=TA_CENTER,
                fontSize=20,
                spaceAfter=15
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="Caption",
                parent=self.styles["Normal"],
                alignment=TA_CENTER,
                fontSize=7
            )
        )

        story = []

        # ========================================================
        # COVER
        # ========================================================

        story.append(
            Paragraph(
                "Crop Performance Report",
                self.styles["ReportTitle"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Model:</b> "
                f"{self.model_name}",
                self.styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Preprocessing:</b> "
                f"{self.usepreprocessing}",
                self.styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Number of observations:</b> "
                f"{len(self.y_true)}",
                self.styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 0.5 * cm)
        )

        # ========================================================
        # GLOBAL METRICS
        # ========================================================

        gm = self.global_metrics()

        global_df = pd.DataFrame(
            [
                {
                    "Model": self.model_name,
                    "N": gm["n"],
                    "R2": gm["r2"],
                    "RMSE": gm["rmse"],
                    "MAE": gm["mae"],
                    "Bias": gm["bias"]
                }
            ]
        )

        story.extend(
            self.dataframe_to_table(
                global_df,
                "Global model performance"
            )
        )

        story.append(
            PageBreak()
        )

        # ========================================================
        # CATEGORY TABLES
        # ========================================================

        for column in self.crop_columns:

            story.append(
                Paragraph(
                    f"Performance by {column}",
                    self.styles["Heading1"]
                )
            )

            table = self.tables[column]

            story.extend(
                self.dataframe_to_table(
                    table,
                    f"Metrics by {column}"
                )
            )

            story.append(
                Spacer(1, 0.5 * cm)
            )

            rankings = self.get_rankings(
                column
            )

            story.extend(
                self.dataframe_to_table(
                    rankings["best"].head(10),
                    f"Best categories - {column}"
                )
            )

            story.extend(
                self.dataframe_to_table(
                    rankings["worst"].head(10),
                    f"Worst categories - {column}"
                )
            )

            story.append(
                PageBreak()
            )

        # ========================================================
        # ALL GENERATED IMAGES
        # ========================================================

        story.append(
            Paragraph(
                "Graphical analysis",
                self.styles["Heading1"]
            )
        )

        # Remove duplicate images while preserving order
        unique_images = list(
            dict.fromkeys(
                self.images
            )
        )

        for image_file in unique_images:

            story.extend(
                self.image_for_pdf(
                    image_file
                )
            )

            story.append(
                PageBreak()
            )

        # ========================================================
        # MODEL × CROP HEATMAP
        # ========================================================

        if model_crop_results is not None:

            heatmap = self.create_r2_heatmap(
                model_crop_results,
                crop_column=heatmap_column
            )

            if heatmap:

                story.append(
                    Paragraph(
                        "Model × Crop R² Heatmap",
                        self.styles["Heading1"]
                    )
                )

                story.extend(
                    self.image_for_pdf(
                        heatmap,
                        width=22 * cm
                    )
                )

                story.append(
                    PageBreak()
                )

        # ========================================================
        # BUILD PDF
        # ========================================================

        doc.build(
            story
        )

        return pdf_filename

    # ============================================================
    # COMPLETE REPORT
    # ============================================================

    def generate(
        self,
        model_crop_results=None,
        create_pdf=True,
        pdf_filename=None
    ):

        print(
            f"Generating CropPerformanceReport: "
            f"{self.model_name}"
        )

        # 1. Metrics
        self.calculate_all_metrics()

        # 2. Save CSV tables
        self.save_tables()

        # 3. Save PostgreSQL
        db_rows = 0

        db_rows = self.save_results_to_db()

        print(
            f"Saved {db_rows} crop performance "
            f"records to PostgreSQL"
        )
        # 4. Generate residual plots
        for column in self.crop_columns:

            self.plot_residual_boxplot(
                column
            )

        # 5. Generate scatter plots
        for column in self.crop_columns:

            self.plot_scatter_categories(
                column
            )

        # 6. Generate PDF
        pdf_file = None

        if create_pdf:

            pdf_file = self.create_pdf(
                pdf_filename=pdf_filename,
                model_crop_results=model_crop_results
            )

        return {
            "crop_results": self.crop_results,
            "images": self.images,
            "pdf": pdf_file
        }