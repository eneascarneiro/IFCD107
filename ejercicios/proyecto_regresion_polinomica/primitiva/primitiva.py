import collections
from datetime import datetime
import glob
import io
import itertools
import os
import random
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def analizar_todo_y_generar_pdf():
    archivos_csv = glob.glob("*.csv")
    if not archivos_csv:
        print("❌ Error: No hay archivos .csv en esta carpeta.")
        return

    lista_df = []
    for archivo in archivos_csv:
        if "analisis_probabilidades" in archivo:
            continue
        try:
            with open(archivo, "r", encoding="utf-8-sig", errors="ignore") as f:
                lineas = f.readlines()
            
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
            if all(col in df.columns for col in ["N1", "N2", "N3", "N4", "N5", "N6"]):
                lista_df.append(df)
        except Exception as e:
            print(f"⚠️ Error al leer {archivo}: {e}")

    if not lista_df:
        print("❌ No se pudieron cargar datos válidos.")
        return

    df_total = pd.concat(lista_df, ignore_index=True).drop_duplicates(subset=["FECHA"]).reset_index(drop=True)
    
    # 1. Frecuencias de Números
    todos_los_numeros = []
    for col in ["N1", "N2", "N3", "N4", "N5", "N6"]:
        valores = pd.to_numeric(df_total[col], errors="coerce").dropna().astype(int)
        todos_los_numeros.extend([n for n in valores if 1 <= n <= 49])
    
    conteo_nums = collections.Counter(todos_los_numeros)
    top_10_numeros = sorted([num for num, _ in conteo_nums.most_common(10)])
    max_prob_6_numeros = sorted([num for num, _ in conteo_nums.most_common(6)])

    # 2. Frecuencias de Reintegros (R)
    top_reintegros = [4, 7, 1, 2, 0]
    if "R" in df_total.columns:
        valores_r = pd.to_numeric(df_total["R"], errors="coerce").dropna().astype(int)
        if not valores_r.empty:
            top_reintegros = [num for num, _ in collections.Counter(valores_r).most_common(5)]
    max_prob_reintegro = top_reintegros[0]

    # 3. Frecuencias de Joker
    top_jokers = ["7459175", "2445443", "3341852", "6414438", "1147192"]
    if "Joker" in df_total.columns:
        jokers_v = [str(j).strip() for j in df_total["Joker"].dropna() if str(j).strip().isdigit()]
        if jokers_v:
            top_jokers = [joker for joker, _ in collections.Counter(jokers_v).most_common(5)]
    max_prob_joker = top_jokers[0]

    # Combinaciones secundarias
    todas_las_combis = list(itertools.combinations(top_10_numeros, 6))
    random.seed(datetime.now().timestamp())
    combinaciones_seleccionadas = random.sample(todas_las_combis, 20) if len(todas_las_combis) >= 20 else (todas_las_combis * 20)[:20]

    # Generar PDF
    os.makedirs("generated", exist_ok=True)
    pdf_path = "generated/analisis_combinaciones_primitiva.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, title="Estudio Global de La Primitiva")
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#1A365D"), alignment=1)
    h2_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#2C5282"), spaceBefore=10)
    body_style = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9)
    th_style = ParagraphStyle("TH", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=colors.white, alignment=1)
    td_style = ParagraphStyle("TD", parent=styles["Normal"], fontSize=9, alignment=1)
    text_max_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#C53030"), alignment=1)

    elements = []
    elements.append(Paragraph("💎 REPORTE GLOBAL DE LA PRIMITIVA", title_style))
    elements.append(Paragraph(f"<b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", body_style))
    elements.append(Paragraph(f"<b>Muestra total unificada:</b> {len(df_total)} sorteos analizados", body_style))
    elements.append(Spacer(1, 10))

    # Apuesta Maestra
    elements.append(Paragraph("🎯 APUESTA MAESTRA (MÁXIMA PROBABILIDAD)", h2_style))
    combo_max_str = ", ".join(f"{n:02d}" for n in max_prob_6_numeros)
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
    ]))
    elements.append(t_max)
    elements.append(Spacer(1, 15))

    # Boletos Secundarios Sugeridos
    elements.append(Paragraph("🎫 Boletos Sugeridos Secundarios (Distribución Top 10)", h2_style))
    apuesta_idx = 0
    for i in range(1, 6):
        elements.append(Paragraph(f"<b>Boleto Sugerido N° {i}</b>", body_style))
        table_data = [[Paragraph("Línea", th_style), Paragraph("Combinación de 6 Números", th_style), Paragraph("R", th_style), Paragraph("Joker", th_style)]]
        
        for col_idx in range(1, 5):
            combo = combinaciones_seleccionadas[apuesta_idx]
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
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))

    doc.build(elements)
    print("🎉 PDF Generado con éxito en: generated/analisis_combinaciones_primitiva.pdf")

if __name__ == "__main__":
    analizar_todo_y_generar_pdf()




