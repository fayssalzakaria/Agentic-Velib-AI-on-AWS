"""
report_generator.py — Génère un rapport PDF avec analyse Groq
Sans Aurora, directement depuis Streamlit
"""
import io
import os
from datetime import datetime

import pandas as pd
import pytz
from llm_client import call_llm_text

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PARIS_TZ = pytz.timezone("Europe/Paris")


def _call_groq(prompt: str) -> str:
    try:
        answer, _ = call_llm_text(
            prompt,
            max_tokens=1000,
            temperature=0.7,
            timeout=30,
        )
        return answer
    except Exception as e:
        return f"Analyse IA indisponible : {e}"


def generate_pdf_report(df: pd.DataFrame) -> bytes:
    """
    Génère un rapport PDF complet avec analyse Groq.
    Retourne les bytes du PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )

    now = datetime.now(PARIS_TZ)
    snap_str = now.strftime("%Y-%m-%d %H:%M")

    # Stats
    total_stations = len(df)
    total_bikes = int(df["numbikesavailable"].sum())
    total_docks = int(df["numdocksavailable"].sum())
    avg_ratio = float(df["bike_ratio"].mean())
    empty = int(df["is_empty"].sum())
    full = int(df["is_full"].sum())
    mechanical = int(df["mechanical"].sum())
    ebike = int(df["ebike"].sum())
    top3_full = ", ".join(df.nlargest(3, "numbikesavailable")["name"].tolist())
    top3_empty = ", ".join(df.nsmallest(3, "numbikesavailable")["name"].tolist())

    # Appel Groq
    prompt = f"""
Tu es un expert en mobilite urbaine a Paris specialise dans le reseau Velib.
Genere une analyse professionnelle en francais du reseau Velib.

Donnees ({snap_str}) :
- Stations actives : {total_stations}
- Velos disponibles : {total_bikes}
- Bornes disponibles : {total_docks}
- Taux de remplissage moyen : {avg_ratio:.1%}
- Stations vides : {empty}
- Stations pleines : {full}
- Velos mecaniques : {mechanical}
- Velos electriques : {ebike}
- Top 3 mieux fournies : {top3_full}
- Top 3 plus vides : {top3_empty}
- Heure : {now.hour}h, Jour : {now.strftime('%A')}

Fournis :
1. Synthese globale du reseau (2-3 phrases)
2. Points d'attention (stations critiques)
3. Recommandations pour les usagers
4. Tendances observees selon l'heure et le jour

Sois concis et professionnel.
"""
    ai_analysis = _call_groq(prompt)

    # Génération PDF
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=18, spaceAfter=6,
    )
    style_subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#888780"),
        spaceAfter=20,
    )
    style_heading = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#185FA5"),
        spaceAfter=8, spaceBefore=16,
    )
    style_body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=16, spaceAfter=8,
    )

    elements = []

    # Titre
    elements.append(Paragraph("Rapport Velib — Analyse IA", style_title))
    elements.append(Paragraph(f"Genere le {snap_str}", style_subtitle))

    # Analyse Groq
    elements.append(Paragraph("Analyse du reseau", style_heading))
    for ligne in ai_analysis.split("\n"):
        if ligne.strip():
            elements.append(Paragraph(ligne.strip(), style_body))
    elements.append(Spacer(1, 16))

    # Stats globales
    elements.append(Paragraph("Statistiques globales", style_heading))
    stats = [
        ["Indicateur", "Valeur"],
        ["Stations actives", str(total_stations)],
        ["Velos disponibles", str(total_bikes)],
        ["Bornes disponibles", str(total_docks)],
        ["Taux de remplissage", f"{avg_ratio:.1%}"],
        ["Stations vides", str(empty)],
        ["Stations pleines", str(full)],
        ["Velos mecaniques", str(mechanical)],
        ["Velos electriques", str(ebike)],
    ]
    t = Table(stats, colWidths=[9*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#185FA5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E6F1FB")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B4B2A9")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))

    # Top 10 mieux fournies
    elements.append(Paragraph("Top 10 stations les mieux fournies", style_heading))
    top10 = df.nlargest(10, "numbikesavailable")[["name", "numbikesavailable", "ebike"]]
    data_top = [["Station", "Velos dispo", "dont electriques"]] + [
        [row["name"], str(row["numbikesavailable"]), str(row["ebike"])]
        for _, row in top10.iterrows()
    ]
    t2 = Table(data_top, colWidths=[9*cm, 4*cm, 4*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D9E75")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E1F5EE")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B4B2A9")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 16))

    # Top 10 plus vides
    elements.append(Paragraph("Top 10 stations les plus vides", style_heading))
    empty10 = df.nsmallest(10, "numbikesavailable")[["name", "numbikesavailable", "numdocksavailable"]]
    data_empty = [["Station", "Velos dispo", "Bornes dispo"]] + [
        [row["name"], str(row["numbikesavailable"]), str(row["numdocksavailable"])]
        for _, row in empty10.iterrows()
    ]
    t3 = Table(data_empty, colWidths=[9*cm, 4*cm, 4*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#A32D2D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FCEBEB")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B4B2A9")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t3)

    doc.build(elements)
    return buf.getvalue()