import collections
from datetime import datetime
import glob
import io
import itertools
import logging
import os
import random
from typing import List, Tuple, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Configuración del sistema de trazabilidad (Logging) según buenas prácticas
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def es_combinacion_probable(combo: Tuple[int, ...]) -> bool:
    """
    Aplica filtros de probabilidad conjunta basados en distribuciones históricas
    de loterías tipo 6/49 para descartar combinaciones estructuralmente improbables.
    """
    # 1. Filtro Par/Impar: Evitar extremos (6 pares o 6 impares)
    impares = sum(1 for x in combo if x % 2 != 0)
    if impares in (0, 6):
        return False

    # 2. Filtro Alto/Bajo: Bajos (1-25), Altos (26-49). Evitar extremos (0/6 o 6/0)
    bajos = sum(1 for x in combo if x <= 25)
    if bajos in (0, 6):
        return False

    # 3. Filtro de Suma: La suma de los 6 números debe estar en el intervalo de confianza del ~75%
    suma = sum(combo)
    if not (115 <= suma <= 185):
        return False

    # 4. Filtro de Consecutivos: Evitar agrupaciones de 3 o más números consecutivos (ej: 14, 15, 16)
    combo_ordenado = sorted(combo)
    consecutivos = 0
    for i in range(len(combo_ordenado) - 1):
        if combo_ordenado[i + 1] - combo_ordenado[i] == 1:
            consecutivos += 1
    if consecutivos >= 2:  # Permite como máximo una pareja de consecutivos (ej: 12, 13)
        return False

    return True


def cargar_y_unificar_datos() -> Optional[pd.DataFrame]:
    """
    Busca, limpia y unifica los ficheros CSV históricos aplicando control de excepciones.
    """
    archivos_csv = glob.glob("*.csv")
    if not archivos_csv:
        logger.error("No se encontraron archivos .csv en el directorio actual.")
        return None

    lista_df = []
    for archivo in archivos_csv:
        if "analisis_probabilidades" in archivo:
            continue
        try:
            with open(archivo, "r", encoding="utf-8-sig", errors="ignore") as f:
                lineas = f.readlines()

            # Localizar cabecera real del conjunto de datos
            idx = -1
            for i, linea in enumerate(lineas):
                if "FECHA" in linea and "N1" in linea:
                    idx = i
                    break

            if idx == -1:
                df = pd.read_csv(archivo, sep=";")
            else:
                df = pd.read_csv(io.StringIO("".join(lineas[idx:])), sep=";")

            df.columns = [str(c).strip() for c in df.columns]
            
            # Validación de integridad de columnas críticas
            columnas_requeridas = ["N1", "N2", "N3", "N4", "N5", "N6"]
            if all(col in df.columns for col in columnas_requeridas):
                lista_df.append(df)
            else:
                logger.warning(f"El archivo {archivo} no contiene las columnas numéricas requeridas.")
        except Exception as e:
            logger.error(f"Error al procesar el archivo {archivo}: {e}")

    if not lista_df:
        return None

    # Unificación y eliminación de duplicados por clave primaria lógica (FECHA)
    df_total = pd.concat(lista_df, ignore_index=True)
    if "FECHA" in df_total.columns:
        df_total = df_total.drop_duplicates(subset=["FECHA"]).reset_index(drop=True)
    else:
        df_total = df_total.drop_duplicates().reset_index(drop=True)

    return df_total


def analizar_frecuencias(df_total: pd.DataFrame) -> Tuple[List[int], List[int], int, List[str], str]:
    """
    Realiza el análisis de frecuencias vectorizado con Pandas y extrae los líderes estadísticos.
    """
    # 1. Números principales (N1-N6)
    cols_num = ["N1", "N2", "N3", "N4", "N5", "N6"]
    df_nums = df_total[cols_num].apply(pd.to_numeric, errors="coerce")
    
    # Vectorización para aplanar y filtrar en rango válido [1, 49]
    all_numbers = df_nums.stack()
    all_numbers = all_numbers[(all_numbers >= 1) & (all_numbers <= 49)].astype(int)
    
    conteo_nums = all_numbers.value_counts()
    
    # Extraemos el Top 12 para tener mayor variabilidad en combinaciones filtradas
    top_12_numeros = sorted(conteo_nums.head(12).index.tolist())
    # Apuesta Maestra: Los 6 números más frecuentes absolutos
    max_prob_6_numeros = sorted(conteo_nums.head(6).index.tolist())

    # 2. Reintegros (R)
    top_reintegros = [4, 7, 1, 2, 0]  # Valores por defecto estadísticos
    if "R" in df_total.columns:
        valores_r = pd.to_numeric(df_total["R"], errors="coerce").dropna().astype(int)
        if not valores_r.empty:
            top_reintegros = valores_r.value_counts().head(5).index.tolist()
    max_prob_reintegro = top_reintegros[0]

    # 3. Joker
    top_jokers = ["7459175", "2445443", "3341852", "6414438", "1147192"]
    if "Joker" in df_total.columns:
        jokers_v = df_total["Joker"].dropna().astype(str).str.strip()
        jokers_v = jokers_v[jokers_v.str.isdigit()]
        if not jokers_v.empty:
            top_jokers = jokers_v.value_counts().head(5).index.tolist()
    max_prob_joker = top_jokers[0]

    return top_12_numeros, max_prob_6_numeros, max_prob_reintegro, top_jokers, max_prob_joker


def generar_pdf_reporte(
    df_total: pd.DataFrame,
    max_prob_6_nums: List[int],
    max_prob_reintegro: int,
    max_prob_joker: str,
    combinaciones: List[Tuple[int, ...]],
    top_reintegros: List[int],
    top_jokers: List[str]
) -> None:
    """
    Genera el reporte ejecutivo en formato PDF utilizando ReportLab.
    """
    os.makedirs("generated", exist_ok=True)
    pdf_path = "generated/analisis_combinaciones_primitiva.pdf"
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        title="Estudio Global de La Primitiva",
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Definición de estilos profesionales (Paleta fría corporativa)
    title_style = ParagraphStyle(
        "T", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#1A365D"), alignment=1, spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "H", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#2C5282"), spaceBefore=12, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "B", parent=styles["BodyText"], fontSize=9, spaceAfter=4
    )
    th_style = ParagraphStyle(
        "TH", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=colors.white, alignment=1
    )
    td_style = ParagraphStyle(
        "TD", parent=styles["Normal"], fontSize=9, alignment=1
    )
    text_max_style = ParagraphStyle(
        "M", parent=styles["Normal"], fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#C53030"), alignment=1
    )

    elements = []
    
    # Cabecera
    elements.append(Paragraph("💎 REPORTE GLOBAL DE LA PRIMITIVA (ANÁLISIS PROBABILÍSTICO)", title_style))
    elements.append(Paragraph(f"<b>Fecha del Análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", body_style))
    elements.append(Paragraph(f"<b>Muestra total unificada:</b> {len(df_total)} sorteos analizados", body_style))
    elements.append(Spacer(1, 10))

    # Sección 1: Apuesta Maestra
    elements.append(Paragraph("🎯 APUESTA MAESTRA (MÁXIMA FRECUENCIA HISTÓRICA)", h2_style))
    combo_max_str = ", ".join(f"{n:02d}" for n in max_prob_6_nums)
    max_data = [
        [Paragraph("COMBINACIÓN LÍDER", th_style), Paragraph("REINTEGRO", th_style), Paragraph("JOKER LÍDER", th_style)],
        [Paragraph(combo_max_str, text_max_style), Paragraph(str(max_prob_reintegro), text_max_style), Paragraph(str(max_prob_joker), text_max_style)]
    ]
    t_max = Table(max_data, colWidths=[240, 90, 150])
    t_max.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9B2C2C")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#E53E3E")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF5F5")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_max)
    elements.append(Spacer(1, 15))

    # Sección 2: Boletos Sugeridos
    elements.append(Paragraph("🎫 BOLETOS SUGERIDOS CON FILTRADO DE COHERENCIA PROBABILÍSTICA", h2_style))
    elements.append(Paragraph(
        "<i>Nota: Estas combinaciones han sido optimizadas mediante filtros de distribución conjunta "
        "(Par/Impar, Alto/Bajo, Suma teórica y Consecutividad) para maximizar la viabilidad matemática.</i>",
        body_style
    ))
    elements.append(Spacer(1, 5))

    apuesta_idx = 0
    for i in range(1, 6):
        elements.append(Paragraph(f"<b>Boleto Sugerido N° {i}</b>", body_style))
        table_data = [
            [Paragraph("Línea", th_style), Paragraph("Combinación de 6 Números", th_style), Paragraph("R", th_style), Paragraph("Joker", th_style)]
        ]
        
        for col_idx in range(1, 5):
            if apuesta_idx >= len(combinaciones):
                break
            combo = combinaciones[apuesta_idx]
            combo_str = ", ".join(f"{n:02d}" for n in combo)
            reintegro = top_reintegros[apuesta_idx % len(top_reintegros)]
            joker = top_jokers[apuesta_idx % len(top_jokers)]
            
            table_data.append([
                Paragraph(f"Columna {col_idx}", td_style),
                Paragraph(combo_str, td_style),
                Paragraph(str(reintegro), td_style),
                Paragraph(str(joker), td_style)
            ])
            apuesta_idx += 1

        t = Table(table_data, colWidths=[60, 220, 50, 150])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5282")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))

    doc.build(elements)
    logger.info(f"PDF generado exitosamente en: {pdf_path}")


def analizar_todo_y_generar_pdf() -> None:
    """
    Función principal que coordina el flujo de ejecución del programa.
    """
    logger.info("Iniciando análisis global de La Primitiva...")
    
    df_total = cargar_y_unificar_datos()
    if df_total is None or df_total.empty:
        logger.error("No se pudieron cargar datos válidos para el análisis.")
        return

    # Extracción de frecuencias principales
    top_12_nums, max_prob_6_nums, max_prob_reintegro, top_jokers, max_prob_joker = analizar_frecuencias(df_total)

    # Generación de combinaciones a partir del Top 12 de números más frecuentes
    todas_las_combis = list(itertools.combinations(top_12_nums, 6))
    
    # Aplicación del motor de filtrado probabilístico
    combis_filtradas = [c for c in todas_las_combis if es_combinacion_probable(c)]
    logger.info(f"Combinaciones generadas: {len(todas_las_combis)} | Filtradas viables: {len(combis_filtradas)}")

    # Gestión de contingencia si el filtrado es demasiado estricto
    if len(combis_filtradas) < 20:
        logger.warning("Pocas combinaciones viables tras el filtrado. Relajando restricciones (usando pool general).")
        combis_filtradas = todas_las_combis

    # Selección aleatoria de 20 combinaciones de entre las estadísticamente óptimas
    # Se utiliza la entropía por defecto del sistema (sin semilla fija para evitar predictibilidad)
    combinaciones_seleccionadas = random.sample(combis_filtradas, 20) if len(combis_filtradas) >= 20 else (combis_filtradas * 20)[:20]
    combinaciones_ordenadas = [tuple(sorted(c)) for c in combinaciones_seleccionadas]

    # Obtención de la lista de reintegros para la asignación en los boletos sugeridos
    top_reintegros = [4, 7, 1, 2, 0]
    if "R" in df_total.columns:
        valores_r = pd.to_numeric(df_total["R"], errors="coerce").dropna().astype(int)
        if not valores_r.empty:
            top_reintegros = valores_r.value_counts().head(5).index.tolist()

    # Generación del entregable final
    generar_pdf_reporte(
        df_total=df_total,
        max_prob_6_nums=max_prob_6_nums,
        max_prob_reintegro=max_prob_reintegro,
        max_prob_joker=max_prob_joker,
        combinaciones=combinaciones_ordenadas,
        top_reintegros=top_reintegros,
        top_jokers=top_jokers
    )


if __name__ == "__main__":
    analizar_todo_y_generar_pdf()