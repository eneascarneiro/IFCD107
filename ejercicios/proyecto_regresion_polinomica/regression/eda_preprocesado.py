"""Codificación didáctica de columnas y generación de informe PDF de EDA."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class EdaPreprocesado:
    """Convierte campos seleccionados y genera un informe verificable en PDF.

    Las categorías nominales se sustituyen por códigos deterministas y sus
    diccionarios se devuelven para que el usuario pueda interpretar el CSV.
    El mes se representa con seno y coseno para respetar la naturaleza cíclica.
    """

    CATEGORY_COLUMNS = ("tipo_tramite", "distrito")
    DOCUMENT_COLUMNS = ("requerimientos_documentacion", "requiere_documentacion")
    MONTH_COLUMN = "mes_presentacion"

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        return series.map(
            lambda value: value.strip().casefold() if isinstance(value, str) else value
        )

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("El CSV debe contener al menos una fila de datos.")
        result = frame.copy()
        result.columns = [str(column).strip() for column in result.columns]
        if result.columns.duplicated().any():
            raise ValueError("El CSV no puede contener nombres de columna duplicados.")
        # required = set(self.CATEGORY_COLUMNS) | {self.MONTH_COLUMN}
        # if not any(column in result for column in self.DOCUMENT_COLUMNS):
        #     required.add(self.DOCUMENT_COLUMNS[0])
        # missing = sorted(required.difference(result.columns))
        # if missing:
        #     raise ValueError("Faltan columnas requeridas para este preprocesado: " + ", ".join(missing))
        mappings: dict[str, dict] = {}

        for column in self.CATEGORY_COLUMNS:
            if column not in result:
                continue
            values = self._normalize_text(result[column])
            categories = sorted(values.dropna().astype(str).unique().tolist())
            mapping = {category: index for index, category in enumerate(categories)}
            result[column] = values.map(
                lambda value: mapping.get(str(value)) if pd.notna(value) else np.nan
            ).astype("float64")
            mappings[column] = mapping

        for column in self.DOCUMENT_COLUMNS:
            if column not in result:
                continue
            raw = self._normalize_text(result[column])
            numeric = pd.to_numeric(raw, errors="coerce")
            nonempty = raw.notna() & raw.astype(str).str.strip().ne("")
            if (numeric[nonempty].notna()).all():
                result[column] = numeric
                mappings[column] = {"conversion": "numeric"}
            else:
                categories = sorted(raw.dropna().astype(str).unique().tolist())
                mapping = {category: index for index, category in enumerate(categories)}
                result[column] = raw.map(
                    lambda value: mapping.get(str(value)) if pd.notna(value) else np.nan
                ).astype("float64")
                mappings[column] = mapping

        if self.MONTH_COLUMN in result:
            raw_month = result[self.MONTH_COLUMN]
            month = pd.to_numeric(result[self.MONTH_COLUMN], errors="coerce")
            invalid_month = raw_month.notna() & raw_month.astype(str).str.strip().ne("") & month.isna()
            if invalid_month.any():
                raise ValueError("mes_presentacion debe contener números del 1 al 12 o estar vacío.")
            valid = month.isna() | month.between(1, 12)
            if not valid.all():
                raise ValueError("mes_presentacion debe estar entre 1 y 12 o vacío.")
            radians = 2 * np.pi * (month - 1) / 12
            result["mes_seno"] = np.sin(radians)
            result["mes_coseno"] = np.cos(radians)
            result = result.drop(columns=[self.MONTH_COLUMN])
            mappings[self.MONTH_COLUMN] = {
                "formula_seno": "sin(2*pi*(mes_presentacion-1)/12)",
                "formula_coseno": "cos(2*pi*(mes_presentacion-1)/12)",
            }

        for column in result.columns:
            if result[column].dtype == object:
                # Mantiene otras variables categóricas como texto, pero permite
                # normalizar una columna de conteos numéricos almacenada como str.
                numeric = pd.to_numeric(result[column], errors="coerce")
                nonempty = result[column].notna() & result[column].astype(str).str.strip().ne("")
                if nonempty.any() and numeric[nonempty].notna().all():
                    result[column] = numeric
        return result, mappings

    @staticmethod
    def _profile(frame: pd.DataFrame) -> list[list[str]]:
        rows = []
        for column in frame.columns:
            series = frame[column]
            rows.append([
                str(column), str(series.dtype), str(int(series.isna().sum())),
                f"{100 * series.isna().mean():.1f}%", str(int(series.nunique(dropna=True))),
            ])
        return rows

    def generate_report(self, source: pd.DataFrame, transformed: pd.DataFrame,
                        mappings: dict, pdf_path: Path, source_name: str = "entrada.csv",
                        correlation_matrix: pd.DataFrame | None = None,
                        correlation_min: float = 0.0, correlation_max: float = 1.0,
                        excluded_by_correlation: list[str] | None = None,
                        model_comparisons: dict[str, dict] | None = None) -> None:
        """Escribe un informe PDF con controles de calidad y distribuciones."""
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        # Matplotlib permite hacer un informe con histogramas. Si el entorno
        # todavía no tiene todas sus dependencias gráficas, creamos un PDF de
        # texto con las mismas comprobaciones básicas usando solo Python.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            self._generate_text_report(
                source, transformed, mappings, pdf_path, source_name, correlation_matrix,
                correlation_min, correlation_max, excluded_by_correlation, model_comparisons,
            )
            return

        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.suptitle("Informe EDA y preprocesado", fontsize=20, fontweight="bold", y=.97)
            fig.text(.08, .91, f"Archivo analizado: {source_name}", fontsize=11)
            fig.text(.08, .87,
                     f"Filas de entrada: {len(source)} | columnas de entrada: {len(source.columns)} | "
                     f"duplicados exactos: {int(source.duplicated().sum())}", fontsize=10)
            fig.text(.08, .83,
                     f"Filas de salida: {len(transformed)} | columnas de salida: {len(transformed.columns)} | "
                     f"celdas vacías tras transformación: {int(transformed.isna().sum().sum())}", fontsize=10)

            for title, data, box in (
                ("Perfil de entrada", source, [0.07, .48, .86, .27]),
                ("Perfil preprocesado", transformed, [0.07, .13, .86, .27]),
            ):
                ax = fig.add_axes(box)
                ax.axis("off")
                rows = self._profile(data)
                table = ax.table(cellText=rows[:16],
                                 colLabels=["Columna", "Tipo", "Nulos", "% nulos", "Únicos"],
                                 loc="center", cellLoc="center", colLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(7.5)
                table.scale(1, 1.12)
                for (row, _), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor("#24476b")
                        cell.get_text().set_color("white")
                        cell.get_text().set_weight("bold")
                if len(rows) > 16:
                    ax.text(.5, -.12, f"Mostradas 16 de {len(rows)} columnas",
                            ha="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(title, fontsize=11, pad=4)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if correlation_matrix is not None and not correlation_matrix.empty:
                # Dividimos la matriz en bloques para que el PDF incluya todas
                # las variables y las etiquetas sigan siendo legibles.
                block_size = 12
                for row_start in range(0, len(correlation_matrix.index), block_size):
                    for column_start in range(0, len(correlation_matrix.columns), block_size):
                        visible = correlation_matrix.iloc[
                            row_start:row_start + block_size,
                            column_start:column_start + block_size,
                        ]
                        fig, ax = plt.subplots(figsize=(11.7, 8.3))
                        image = ax.imshow(visible.to_numpy(dtype=float), cmap="coolwarm",
                                          vmin=-1, vmax=1, aspect="auto")
                        ax.set_xticks(np.arange(len(visible.columns)))
                        ax.set_yticks(np.arange(len(visible.index)))
                        ax.set_xticklabels([str(value) for value in visible.columns], rotation=55,
                                           ha="right", fontsize=8)
                        ax.set_yticklabels([str(value) for value in visible.index], fontsize=8)
                        for row in range(len(visible.index)):
                            for column in range(len(visible.columns)):
                                value = visible.iloc[row, column]
                                if pd.notna(value):
                                    ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                                            fontsize=7, color="black")
                        fig.colorbar(image, ax=ax, label="Correlación de Pearson")
                        fig.suptitle(
                            f"Matriz de correlación (filas {row_start + 1}-"
                            f"{row_start + len(visible.index)}, columnas "
                            f"{column_start + 1}-{column_start + len(visible.columns)})",
                            fontsize=13,
                        )
                        fig.text(.07, .025,
                                 f"Filtro respecto a y: |r| entre {correlation_min:.2f} y "
                                 f"{correlation_max:.2f}. Variables excluidas: "
                                 f"{', '.join(excluded_by_correlation or []) or 'ninguna'}. "
                                 "Las correlaciones no implican causalidad.", fontsize=8, wrap=True)
                        fig.tight_layout(rect=[0, .07, 1, .94])
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

            numeric_columns = transformed.select_dtypes(include="number").columns.tolist()
            numeric_columns = [column for column in numeric_columns if transformed[column].notna().any()]
            selected = numeric_columns[:6]
            fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
            axes = axes.ravel()
            for ax, column in zip(axes, selected):
                values = transformed[column].dropna()
                ax.hist(values, bins=min(24, max(5, int(np.sqrt(len(values))))),
                        color="#2878B5", edgecolor="white")
                ax.set_title(str(column), fontsize=9)
                ax.tick_params(labelsize=7)
            for ax in axes[len(selected):]:
                ax.axis("off")
            fig.suptitle("Distribuciones numéricas después del preprocesado", fontsize=15)
            fig.text(.08, .04,
                     "Transformaciones aplicadas: " + "; ".join(
                         f"{column}: {mapping}" for column, mapping in mappings.items()
                     )[:900], fontsize=7, wrap=True)
            fig.tight_layout(rect=[0, .09, 1, .94])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Cada búsqueda recibe una página independiente y usa el mismo
            # conjunto de prueba, lo que permite comparar ambas predicciones.
            for method_name, comparison in (model_comparisons or {}).items():
                actual = np.asarray(comparison["y_true"], dtype=float)
                predicted = np.asarray(comparison["y_predicted"], dtype=float)
                if not len(actual):
                    continue
                lower = float(min(actual.min(), predicted.min()))
                upper = float(max(actual.max(), predicted.max()))
                padding = max((upper - lower) * 0.05, 1e-9)
                fig, ax = plt.subplots(figsize=(9, 7))
                ax.scatter(actual, predicted, color="#2878B5", edgecolor="white",
                           alpha=.8, label="Predicciones")
                ax.plot([lower - padding, upper + padding],
                        [lower - padding, upper + padding], color="#d62728",
                        linestyle="--", label="Predicción ideal: predicho = real")
                ax.set_xlim(lower - padding, upper + padding)
                ax.set_ylim(lower - padding, upper + padding)
                ax.set_xlabel("Valor real")
                ax.set_ylabel("Valor predicho")
                ax.set_title(f"{method_name}: predicciones frente a valores reales")
                ax.grid(True, alpha=.25)
                ax.legend()
                ax.text(
                    .04, .96,
                    f"RMSE = {comparison['rmse']:.4g}\nR² = {comparison['r2']:.4g}\n"
                    f"Observaciones de prueba = {len(actual)}",
                    transform=ax.transAxes, va="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": .85},
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def _generate_text_report(self, source: pd.DataFrame, transformed: pd.DataFrame,
                              mappings: dict, pdf_path: Path, source_name: str,
                              correlation_matrix: pd.DataFrame | None = None,
                              correlation_min: float = 0.0, correlation_max: float = 1.0,
                              excluded_by_correlation: list[str] | None = None,
                              model_comparisons: dict[str, dict] | None = None) -> None:
        """Crea un PDF sencillo sin bibliotecas gráficas adicionales."""
        lines = [
            "INFORME EDA Y PREPROCESADO",
            f"Archivo: {source_name}",
            f"Filas de entrada: {len(source)} | columnas: {len(source.columns)}",
            f"Duplicados exactos de entrada: {int(source.duplicated().sum())}",
            f"Filas de salida: {len(transformed)} | columnas: {len(transformed.columns)}",
            f"Celdas vacias de salida: {int(transformed.isna().sum().sum())}",
            "",
            "PERFIL DE ENTRADA (columna | tipo | nulos | unicos)",
        ]
        for column in source.columns:
            series = source[column]
            lines.append(
                f"{column} | {series.dtype} | {int(series.isna().sum())} | "
                f"{int(series.nunique(dropna=True))}"
            )
        lines.extend(["", "TRANSFORMACIONES APLICADAS"])
        for column, mapping in mappings.items():
            lines.append(f"{column}: {mapping}")
        lines.extend([
            "", "MATRIZ DE CORRELACION DE PEARSON (solo columnas numericas)",
            f"Filtro respecto a y: correlacion absoluta entre {correlation_min:.2f} y {correlation_max:.2f}",
            "Variables excluidas por correlacion: " + (", ".join(excluded_by_correlation or []) or "ninguna"),
        ])
        if correlation_matrix is not None:
            lines.append(correlation_matrix.to_string())
        for method_name, comparison in (model_comparisons or {}).items():
            lines.extend([
                "", f"{method_name}: PREDICCIONES FRENTE A VALORES REALES",
                f"RMSE={comparison['rmse']:.6g}; R2={comparison['r2']:.6g}",
                "Real | Predicho",
            ])
            lines.extend(
                f"{actual:.6g} | {predicted:.6g}"
                for actual, predicted in zip(comparison["y_true"], comparison["y_predicted"])
            )
        lines.extend(["", "PERFIL NUMERICO PREPROCESADO (columna | min | mediana | max | nulos)"])
        for column in transformed.select_dtypes(include="number").columns:
            values = transformed[column].dropna()
            if values.empty:
                lines.append(f"{column} | sin valores numericos | nulos={int(transformed[column].isna().sum())}")
            else:
                lines.append(
                    f"{column} | {values.min():.4g} | {values.median():.4g} | "
                    f"{values.max():.4g} | nulos={int(transformed[column].isna().sum())}"
                )
        self._write_basic_pdf(pdf_path, lines)

    @staticmethod
    def _write_basic_pdf(pdf_path: Path, lines: list[str]) -> None:
        """Escribe texto paginado en PDF 1.4 usando solo la biblioteca estándar."""
        pages = [lines[index:index + 48] for index in range(0, len(lines), 48)] or [[]]
        objects: list[bytes] = []
        page_ids = [3 + 2 * index for index in range(len(pages))]
        content_ids = [page_id + 1 for page_id in page_ids]
        font_id = 3 + 2 * len(pages)
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))

        for page_number, page_lines in enumerate(pages, start=1):
            page_id = page_ids[page_number - 1]
            content_id = content_ids[page_number - 1]
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode("ascii")
            )
            commands = ["BT /F1 10 Tf"]
            for line_number, line in enumerate(page_lines):
                if line_number == 0 and page_number == 1:
                    commands.append("/F1 16 Tf")
                else:
                    commands.append("/F1 9 Tf")
                safe = str(line).encode("cp1252", errors="replace").decode("cp1252")
                safe = safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                y_position = 790 - line_number * 15
                commands.append(f"1 0 0 1 45 {y_position} Tm ({safe}) Tj")
            commands.append("ET")
            stream = "\n".join(commands).encode("cp1252", errors="replace")
            objects.append(b"<< /Length " + str(len(stream)).encode("ascii") +
                           b" >>\nstream\n" + stream + b"\nendstream")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

        # El índice xref necesita la posición exacta en bytes de cada objeto.
        document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_number, body in enumerate(objects, start=1):
            offsets.append(len(document))
            document.extend(f"{object_number} 0 obj\n".encode("ascii"))
            document.extend(body)
            document.extend(b"\nendobj\n")
        xref_position = len(document)
        document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        document.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        document.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_position}\n%%EOF\n".encode("ascii")
        )
        pdf_path.write_bytes(document)

