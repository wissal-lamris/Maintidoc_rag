"""Génère le corpus de test de MaintiDoc : 10 documents de la station de pompage fictive « SP-Oulad Saleh ».

Tout est inventé (station, équipements, valeurs, incidents). Les documents contiennent du texte,
des tableaux et des schémas, pour tester l'extraction multimodale, CLIP et les descriptions d'images.

Lancement (facultatif : les PDF sont déjà dans data/pdfs) :
    pip install reportlab matplotlib
    python -m scripts.generer_documents
"""
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "pdfs"
ST = getSampleStyleSheet()
H1, H2, P, CAP = ST["Title"], ST["Heading2"], ST["BodyText"], ST["Italic"]
MENTION = "Document de test - station SP-Oulad Saleh, équipements et valeurs entièrement fictifs."


# ======================= Outils de mise en page =======================
def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def tab(caption, rows, widths):
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return [Paragraph(caption, CAP), t, Spacer(1, 10)]


def fig(buf, caption, width=15):
    img = Image(buf)
    img.drawWidth, img.drawHeight = width * cm, width * cm * img.imageHeight / img.imageWidth
    return [img, Paragraph(caption, CAP), Spacer(1, 10)]


def txt(*paragraphs):
    return [Paragraph(p, P) for p in paragraphs]


def build(name, title, story):
    OUT.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT / name), pagesize=A4, title=title,
                      leftMargin=2 * cm, rightMargin=2 * cm).build(
        [Paragraph(title, H1), Paragraph(MENTION, CAP), Spacer(1, 6), *story])
    print("Créé :", name)


def bloc(ax, x, y, w, h, text, color, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="black"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def fleche(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2))


# ======================= Schémas =======================
def synoptique():
    f, ax = plt.subplots(figsize=(8, 4.2))
    bloc(ax, 0.2, 1.6, 1.6, 1.0, "Bâche d'aspiration\n800 m³", "#c6dbef")
    for i, y in enumerate([3.2, 1.9, 0.6]):
        bloc(ax, 2.6, y, 1.5, 0.8, f"Groupe P{i + 1}\nHM-160", "#9ecae1" if i < 2 else "#deebf7")
        fleche(ax, (1.8, 2.1), (2.6, y + 0.4))
        bloc(ax, 4.6, y + 0.15, 0.9, 0.5, f"Clapet C{i + 1}", "#fdd0a2")
        fleche(ax, (4.1, y + 0.4), (4.6, y + 0.4))
        fleche(ax, (5.5, y + 0.4), (6.1, 2.1))
    bloc(ax, 6.1, 1.6, 1.3, 1.0, "Collecteur\nDN300\nVanne V10", "#fdae6b")
    fleche(ax, (7.4, 2.1), (8.0, 2.1))
    bloc(ax, 8.0, 1.4, 1.6, 1.4, "Réservoir\nde tête\n2 500 m³", "#c7e9c0")
    ax.text(2.6, 0.2, "P1 et P2 en service, P3 en secours", fontsize=8, style="italic")
    ax.set(xlim=(0, 9.8), ylim=(0, 4.3), title="Station SP-Oulad Saleh - synoptique hydraulique")
    ax.axis("off")
    return png(f)


def unifilaire():
    f, ax = plt.subplots(figsize=(8, 4.5))
    bloc(ax, 3.4, 4.0, 2.2, 0.6, "Arrivée réseau 22 kV", "#fee0d2")
    fleche(ax, (4.5, 4.0), (4.5, 3.5))
    bloc(ax, 3.4, 2.9, 2.2, 0.6, "Transformateur TR1\n22 kV / 400 V - 250 kVA", "#fcbba1")
    fleche(ax, (4.5, 2.9), (4.5, 2.45))
    ax.plot([0.8, 8.2], [2.4, 2.4], lw=4, color="black")
    ax.text(8.3, 2.35, "Jeu de barres TGBT", fontsize=8)
    for i, x in enumerate([1.2, 3.4, 5.6]):
        fleche(ax, (x + 0.6, 2.4), (x + 0.6, 1.75))
        bloc(ax, x, 1.15, 1.2, 0.6, f"Variateur\nVF-45 n°{i + 1}", "#deebf7")
        fleche(ax, (x + 0.6, 1.15), (x + 0.6, 0.65))
        bloc(ax, x, 0.05, 1.2, 0.6, f"Moteur M{i + 1}\n37 kW", "#9ecae1")
    fleche(ax, (7.6, 2.4), (7.6, 1.75))
    bloc(ax, 7.0, 1.15, 1.3, 0.6, "Auxiliaires\n(éclairage, API)", "#e5f5e0")
    ax.set(xlim=(0.5, 9.8), ylim=(-0.1, 4.8), title="Schéma électrique unifilaire simplifié")
    ax.axis("off")
    return png(f)


def courbe_pompe():
    q, h, eta = [0, 40, 80, 120, 160, 200, 240], [62, 61, 59, 55, 49, 41, 31], [0, 38, 60, 74, 79, 75, 63]
    f, a1 = plt.subplots(figsize=(7, 4))
    a1.plot(q, h, "o-", color="#08519c", label="Hauteur manométrique (m)")
    a1.set(xlabel="Débit (m³/h)", ylabel="Hauteur manométrique (m)",
           title="Pompe HM-160 - courbe caractéristique à 2950 tr/min")
    a1.axvline(160, ls="--", color="grey")
    a1.text(165, 36, "Point nominal\n160 m³/h - 49 m", fontsize=8)
    a2 = a1.twinx()
    a2.plot(q, eta, "s-", color="#e6550d", label="Rendement (%)")
    a2.set_ylabel("Rendement (%)")
    f.legend(loc="lower left", bbox_to_anchor=(0.12, 0.12), fontsize=8)
    return png(f)


def coupe_pompe():
    f, ax = plt.subplots(figsize=(7, 4))
    ax.add_patch(Circle((3, 2), 1.5, fill=False, lw=2))
    ax.add_patch(Circle((3, 2), 0.8, color="#9ecae1"))
    for a in range(0, 360, 45):
        ax.plot([3, 3 + 0.8 * math.cos(math.radians(a))], [2, 2 + 0.8 * math.sin(math.radians(a))],
                color="#08519c", lw=2)
    ax.add_patch(Rectangle((3, 1.85), 4.5, 0.3, color="#636363"))
    ax.add_patch(Rectangle((4.6, 1.6), 0.4, 0.8, color="#fdae6b"))
    ax.add_patch(Rectangle((5.6, 1.55), 0.5, 0.9, color="#74c476"))
    ax.add_patch(Rectangle((6.6, 1.55), 0.5, 0.9, color="#74c476"))
    ax.add_patch(Rectangle((2.6, 3.4), 0.8, 1.0, fill=False, lw=2))
    ax.add_patch(Rectangle((0.2, 1.7), 1.4, 0.6, fill=False, lw=2))
    for xy, xt, t in [((3, 2), (1.2, 0.2), "Roue (impulseur)"), ((3, 3.5), (1.0, 3.9), "Volute"),
                      ((4.8, 2.4), (4.0, 3.1), "Garniture mécanique"), ((5.85, 2.45), (5.9, 3.9), "Roulement côté pompe"),
                      ((6.85, 1.55), (6.6, 0.4), "Roulement côté moteur"), ((7.3, 2.0), (8.2, 2.6), "Arbre"),
                      ((3.0, 4.4), (3.6, 4.6), "Refoulement DN150"), ((0.9, 1.7), (0.2, 0.9), "Aspiration DN200")]:
        ax.annotate(t, xy=xy, xytext=xt, fontsize=9, arrowprops={"arrowstyle": "->"})
    ax.set(xlim=(0, 9.5), ylim=(0, 5))
    ax.axis("off")
    return png(f)


def raccordement_variateur():
    f, ax = plt.subplots(figsize=(8, 3.8))
    bloc(ax, 0.2, 1.4, 1.6, 1.0, "Disjoncteur Q1\n400 V", "#fee0d2")
    bloc(ax, 2.6, 1.4, 1.6, 1.0, "Self de ligne\nL1", "#fdd0a2")
    bloc(ax, 5.0, 1.0, 2.0, 1.8, "Variateur VF-45\nL1 L2 L3 -> U V W\nBornes 1-12", "#deebf7")
    bloc(ax, 7.8, 1.4, 1.5, 1.0, "Moteur M\n37 kW", "#9ecae1")
    for a, b in [((1.8, 1.9), (2.6, 1.9)), ((4.2, 1.9), (5.0, 1.9)), ((7.0, 1.9), (7.8, 1.9))]:
        fleche(ax, a, b)
    bloc(ax, 5.0, 3.2, 2.0, 0.6, "Capteur de pression\n4-20 mA (borne 5)", "#e5f5e0")
    fleche(ax, (6.0, 3.2), (6.0, 2.8))
    bloc(ax, 5.0, 0.0, 2.0, 0.6, "Automate (marche/arrêt,\ndéfaut : bornes 8-9)", "#e5f5e0")
    fleche(ax, (6.0, 0.6), (6.0, 1.0))
    ax.set(xlim=(0, 9.5), ylim=(-0.1, 4.0), title="Raccordement du variateur VF-45")
    ax.axis("off")
    return png(f)


def calendrier():
    f, ax = plt.subplots(figsize=(8, 3.6))
    taches = [("Graissage roulements", [1, 4, 7, 10]), ("Contrôle alignement", [3, 9]),
              ("Analyse vibratoire", list(range(1, 13))), ("Nettoyage bâche", [6]),
              ("Essai groupe de secours P3", [2, 5, 8, 11]), ("Révision variateurs", [12])]
    for i, (t, mois) in enumerate(taches):
        for m in mois:
            ax.add_patch(Rectangle((m - 0.4, i - 0.3), 0.8, 0.6, color="#3182bd"))
    ax.set(yticks=range(len(taches)), yticklabels=[t for t, _ in taches], xticks=range(1, 13),
           xticklabels=["janv", "févr", "mars", "avr", "mai", "juin", "juil", "août", "sept", "oct", "nov", "déc"],
           xlim=(0.4, 12.6), ylim=(-0.6, len(taches) - 0.4), title="Calendrier de maintenance préventive 2026")
    ax.grid(axis="x", alpha=0.3)
    return png(f)


def armoire():
    f, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.add_patch(Rectangle((0.5, 0.3), 5.5, 4.4, fill=False, lw=2))
    bloc(ax, 0.9, 3.6, 4.7, 0.7, "Sectionneur général QS0 (cadenassable)", "#fcbba1")
    for i in range(3):
        bloc(ax, 0.9 + i * 1.6, 2.2, 1.4, 1.0, f"Départ P{i + 1}\nQ{i + 1}\n(cadenassable)", "#fdd0a2", 7)
        bloc(ax, 0.9 + i * 1.6, 0.8, 1.4, 1.0, f"Variateur\nVF-45 n°{i + 1}", "#deebf7", 7)
    ax.text(6.2, 3.9, "1. Couper ici", fontsize=9, color="red")
    ax.text(6.2, 2.6, "2. Cadenasser\nle départ", fontsize=9, color="red")
    ax.text(6.2, 1.2, "3. Attendre 5 min\n(condensateurs)", fontsize=9, color="red")
    ax.set(xlim=(0.3, 8.3), ylim=(0.1, 4.9), title="Armoire de puissance - points de consignation")
    ax.axis("off")
    return png(f)


def eclate_garniture():
    f, ax = plt.subplots(figsize=(8, 2.8))
    pieces = [("Arbre", "#969696", 1.0), ("Bague\nd'arrêt", "#fdd0a2", 0.8), ("Ressort", "#c7e9c0", 1.0),
              ("Grain\ntournant\n(carbone)", "#9ecae1", 1.0), ("Grain fixe\n(SiC)", "#6baed6", 1.0),
              ("Joint\ntorique", "#fcbba1", 0.6), ("Fouloir", "#bdbdbd", 1.0)]
    x = 0.2
    for i, (name, color, w) in enumerate(pieces):
        bloc(ax, x, 0.6, w, 1.2, name, color, 7)
        ax.text(x + w / 2, 0.25, f"{i + 1}", ha="center", fontsize=9, weight="bold")
        x += w + 0.35
    ax.set(xlim=(0, x), ylim=(0, 2.1), title="Vue éclatée de la garniture mécanique HM160-GM04 (ordre de montage)")
    ax.axis("off")
    return png(f)


def logigramme():
    f, ax = plt.subplots(figsize=(7, 5))
    for x, y, t, c in [(2.5, 5.6, "Vibration > 7,1 mm/s ?", "#fcbba1"),
                       (0.2, 4.2, "Mesurer roulements\n(côté pompe et moteur)", "#deebf7"),
                       (4.8, 4.2, "Fonctionnement normal :\nsuivi mensuel", "#c7e9c0"),
                       (0.2, 2.8, "Pic à 1x la vitesse ?", "#fcbba1"),
                       (4.8, 2.8, "Balourd de la roue :\nnettoyer / équilibrer", "#fdd0a2"),
                       (0.2, 1.4, "Pic à 2x la vitesse ?", "#fcbba1"),
                       (4.8, 1.4, "Désalignement :\nréaligner l'accouplement", "#fdd0a2"),
                       (0.2, 0.0, "Défaut de roulement :\nremplacer le roulement", "#fdd0a2")]:
        bloc(ax, x, y, 3.0, 0.6, t, c)
    for a, b, lab, pos in [((3.2, 5.6), (1.7, 4.8), "oui", (2.0, 5.2)), ((4.8, 5.6), (6.3, 4.8), "non", (5.8, 5.2)),
                           ((1.7, 4.2), (1.7, 3.4), "", (0, 0)), ((3.2, 3.1), (4.8, 3.1), "oui", (3.8, 3.2)),
                           ((1.7, 2.8), (1.7, 2.0), "non", (1.8, 2.35)), ((3.2, 1.7), (4.8, 1.7), "oui", (3.8, 1.8)),
                           ((1.7, 1.4), (1.7, 0.6), "non", (1.8, 0.95))]:
        fleche(ax, a, b)
        if lab:
            ax.text(*pos, lab, fontsize=8)
    ax.set(xlim=(0, 8), ylim=(-0.2, 6.4))
    ax.axis("off")
    return png(f)


def debits_octobre():
    jours = list(range(1, 32))
    debits = [7400 + 600 * math.sin(j / 3) + (900 if j in (12, 13) else 0) - (2200 if j == 21 else 0) for j in jours]
    f, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(jours, debits, color="#6baed6")
    ax.axhline(7500, ls="--", color="grey")
    ax.text(22, 7650, "Objectif : 7 500 m³/jour", fontsize=8)
    ax.annotate("Arrêt de P2 (cavitation)", xy=(21, debits[20]), xytext=(23, 4500),
                fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Jour d'octobre 2026", ylabel="Volume pompé (m³)", title="Volumes pompés par jour - octobre 2026")
    return png(f)


def pression_aspiration():
    heures = [6 + q / 12 for q in range(0, 73)]              # de 6 h à 12 h, toutes les 5 minutes

    def pression(h):
        if h < 8.25:                                          # avant 8 h 15 : normal
            return 0.35
        if h < 8.75:                                          # 8 h 15 - 8 h 45 : chute (crépine colmatée)
            return 0.35 - (h - 8.25) * 1.2
        if h < 9.17:                                          # jusqu'à l'arrêt de 9 h 10 : cavitation
            return -0.25
        return 0.30                                           # pompe arrêtée : pression statique

    f, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(heures, [pression(h) for h in heures], color="#08519c")
    ax.axhline(-0.10, ls="--", color="red")
    ax.text(10.0, -0.08, "Seuil de cavitation (-0,10 bar)", fontsize=8, color="red")
    ax.annotate("Alarme A05 (8 h 15)", xy=(8.25, 0.35), xytext=(6.3, 0.1), fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.annotate("Arrêt de P2 (9 h 10)", xy=(9.17, -0.25), xytext=(9.6, -0.2), fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Heure (21 octobre 2026)", ylabel="Pression d'aspiration P2 (bar)",
           title="Pression d'aspiration du groupe P2 pendant l'incident")
    return png(f)


def spectre():
    fr = [i / 2 for i in range(0, 401)]
    amp = [0.2 + 3.8 * math.exp(-((x - 49.2) ** 2) / 0.6) + 1.1 * math.exp(-((x - 98.4) ** 2) / 0.6)
           + 0.9 * math.exp(-((x - 147.6) ** 2) / 0.8) for x in fr]
    f, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(fr, amp, color="#08519c")
    for x, lab in [(49.2, "1x (49,2 Hz)"), (98.4, "2x"), (147.6, "3x")]:
        ax.annotate(lab, xy=(x, max(amp[int(x * 2) - 2:int(x * 2) + 3])), xytext=(x + 8, 3.5 if lab.startswith("1x") else 1.8),
                    fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Fréquence (Hz)", ylabel="Vitesse vibratoire (mm/s)",
           title="Spectre vibratoire du groupe P1 - roulement côté pompe (septembre 2026)")
    return png(f)


# ======================= Les 10 documents =======================
def doc01():
    build("01_dossier_technique_station_SP-Oulad_Saleh.pdf", "Dossier technique - Station de pompage SP-Oulad Saleh", [
        Paragraph("1. Présentation de la station", H2),
        *txt("La station de pompage SP-Oulad Saleh, mise en service en 2019, alimente en eau potable les quartiers "
             "sud de la commune. Elle aspire l'eau traitée dans une bâche de 800 m³ et la refoule vers un réservoir "
             "de tête de 2 500 m³ situé 42 m plus haut. Sa capacité nominale est de 320 m³/h avec deux groupes en "
             "service ; le troisième groupe reste en secours et les groupes sont permutés chaque lundi.",
             "La station fonctionne 24 h/24 sans personnel permanent. Un automate surveille les niveaux, les "
             "pressions et les défauts, et envoie les alarmes au centre de conduite."),
        *fig(synoptique(), "Figure 1 : Synoptique hydraulique de la station SP-Oulad Saleh"),
        Paragraph("2. Liste des équipements", H2),
        *tab("Tableau 1 : Équipements principaux de la station", [
            ["Repère", "Équipement", "Modèle", "Caractéristique principale"],
            ["P1, P2, P3", "Pompe centrifuge", "HM-160", "160 m³/h à 49 m"], ["M1, M2, M3", "Moteur asynchrone", "MA-37", "37 kW, 2950 tr/min"],
            ["VF1 à VF3", "Variateur de fréquence", "VF-45", "45 kVA, 400 V"], ["C1 à C3", "Clapet anti-retour", "CL-150", "DN150, PN16"],
            ["V10", "Vanne de sectionnement", "VP-300", "DN300, motorisée"], ["TR1", "Transformateur", "TS-250", "22 kV / 400 V, 250 kVA"],
            ["API", "Automate programmable", "AP-20", "32 entrées / 24 sorties"]], [2.5, 4.5, 2.5, 5.5]),
        Paragraph("3. Alimentation électrique", H2),
        *txt("La station est alimentée en moyenne tension 22 kV. Le transformateur TR1 abaisse la tension à 400 V "
             "et alimente le tableau général basse tension (TGBT), qui distribue l'énergie aux trois variateurs."),
        *fig(unifilaire(), "Figure 2 : Schéma électrique unifilaire simplifié de la station")])


def doc02():
    build("02_notice_constructeur_pompe_HM-160.pdf", "Notice constructeur - Pompe centrifuge HM-160", [
        Paragraph("1. Description", H2),
        *txt("La HM-160 est une pompe centrifuge monocellulaire à aspiration axiale. L'arbre en acier inoxydable "
             "porte une roue fermée en bronze ; l'étanchéité est assurée par une garniture mécanique simple et "
             "l'arbre est guidé par deux roulements à billes."),
        *tab("Tableau 2 : Caractéristiques nominales de la pompe HM-160", [
            ["Paramètre", "Valeur"], ["Débit nominal", "160 m³/h"], ["Hauteur manométrique", "49 m"],
            ["Vitesse de rotation", "2950 tr/min"], ["Puissance absorbée", "27 kW"], ["Rendement au point nominal", "79 %"],
            ["NPSH requis", "4,2 m"], ["Température maximale de l'eau", "40 °C"], ["Masse", "185 kg"]], [8, 7]),
        Paragraph("2. Plage de fonctionnement", H2),
        *txt("La pompe doit fonctionner entre 110 et 200 m³/h. En dessous de 80 m³/h, elle chauffe et vibre : ne "
             "jamais la faire tourner longtemps vanne de refoulement fermée."),
        *fig(courbe_pompe(), "Figure 3 : Courbe caractéristique de la pompe HM-160 (hauteur et rendement)"),
        *fig(coupe_pompe(), "Figure 4 : Coupe simplifiée de la pompe HM-160 et de ses composants"),
        *tab("Tableau 3 : Matériaux et pièces de rechange", [
            ["Pièce", "Matériau", "Référence"], ["Corps (volute)", "Fonte EN-GJL-250", "HM160-CP01"],
            ["Roue", "Bronze CuSn10", "HM160-RO02"], ["Arbre", "Inox 1.4021", "HM160-AR03"],
            ["Garniture mécanique", "Carbure de silicium / carbone", "HM160-GM04"],
            ["Roulement côté pompe", "6309-2Z", "HM160-RL05"], ["Roulement côté moteur", "6311-2Z", "HM160-RL06"]], [5, 5.5, 4.5])])


def doc03():
    build("03_notice_variateur_VF-45.pdf", "Notice d'utilisation - Variateur de fréquence VF-45", [
        Paragraph("1. Rôle du variateur", H2),
        *txt("Chaque moteur est piloté par un variateur VF-45 qui ajuste sa vitesse pour maintenir une pression de "
             "refoulement de consigne de 5,2 bar. La vitesse varie entre 2 100 et 2 950 tr/min selon la demande."),
        *fig(raccordement_variateur(), "Figure 5 : Raccordement du variateur VF-45 (puissance et commande)"),
        *tab("Tableau 4 : Paramètres réglés sur les variateurs de la station", [
            ["Paramètre", "Code", "Valeur réglée"], ["Fréquence minimale", "F01", "35 Hz"], ["Fréquence maximale", "F02", "50 Hz"],
            ["Temps d'accélération", "F05", "15 s"], ["Temps de décélération", "F06", "20 s"],
            ["Consigne de pression", "P10", "5,2 bar"], ["Courant nominal moteur", "M03", "68 A"]], [6, 3, 6]),
        *tab("Tableau 5 : Codes de défaut du variateur VF-45", [
            ["Code", "Défaut", "Action"], ["F-OC", "Surintensité en sortie", "Vérifier le moteur et le câble"],
            ["F-OV", "Surtension du bus continu", "Augmenter le temps de décélération"],
            ["F-OH", "Surchauffe du variateur (> 75 °C)", "Nettoyer le filtre de ventilation"],
            ["F-PH", "Perte d'une phase d'alimentation", "Contrôler les fusibles et le disjoncteur Q1"]], [2.5, 6, 6.5])])


def doc04():
    build("04_plan_maintenance_preventive_2026.pdf", "Plan de maintenance préventive 2026", [
        Paragraph("1. Gamme de maintenance des groupes", H2),
        *tab("Tableau 6 : Périodicités de maintenance préventive des groupes HM-160", [
            ["Opération", "Périodicité", "Intervenant"], ["Contrôle visuel des fuites et du bruit", "Quotidienne", "Opérateur"],
            ["Relevé des vibrations des roulements", "Mensuelle", "Technicien"],
            ["Graissage des roulements du moteur", "Toutes les 2 000 heures", "Technicien"],
            ["Contrôle de l'alignement de l'accouplement", "Semestrielle", "Technicien"],
            ["Remplacement de la garniture mécanique", "Toutes les 16 000 heures", "Technicien"],
            ["Révision complète du groupe", "Toutes les 40 000 heures", "Atelier"]], [8, 4, 3]),
        Paragraph("2. Graissage", H2),
        *tab("Tableau 7 : Quantités de graisse par roulement", [
            ["Roulement", "Graisse", "Quantité"], ["Moteur côté accouplement (6311)", "Lithium NLGI 2", "15 g"],
            ["Moteur côté ventilateur (6309)", "Lithium NLGI 2", "10 g"]], [6.5, 4.5, 4]),
        *txt("Ne pas dépasser les quantités indiquées : un excès de graisse fait chauffer le roulement."),
        Paragraph("3. Calendrier annuel", H2),
        *fig(calendrier(), "Figure 6 : Calendrier de maintenance préventive 2026")])


def doc05():
    build("05_procedure_consignation_electrique.pdf", "Procédure de consignation électrique d'un groupe", [
        Paragraph("1. Objet", H2),
        *txt("Cette procédure doit être appliquée avant toute intervention mécanique ou électrique sur un groupe "
             "de pompage. Elle est réalisée par une personne habilitée électriquement (habilitation B1V minimum)."),
        Paragraph("2. Étapes", H2),
        *txt("1) Prévenir le centre de conduite et passer le groupe en mode « maintenance » sur l'automate. "
             "2) Arrêter le groupe depuis l'automate. 3) Ouvrir le départ Q du groupe dans l'armoire. "
             "4) Cadenasser le départ et poser l'étiquette « Ne pas manœuvrer ». 5) Attendre 5 minutes la décharge "
             "des condensateurs du variateur. 6) Vérifier l'absence de tension avec un VAT. "
             "7) Fermer les vannes d'aspiration et de refoulement du groupe."),
        *fig(armoire(), "Figure 7 : Armoire de puissance et points de consignation", 12),
        *tab("Tableau 8 : Équipements de protection individuelle obligatoires", [
            ["Équipement", "Norme", "Quand"], ["Gants isolants classe 0", "EN 60903", "Vérification d'absence de tension"],
            ["Écran facial anti-arc", "EN 166", "Manœuvre dans l'armoire"], ["Chaussures de sécurité", "EN ISO 20345", "Toujours"],
            ["Cadenas personnel", "-", "Consignation"]], [5, 3.5, 6.5])])


def doc06():
    build("06_procedure_remplacement_garniture_mecanique.pdf", "Procédure de remplacement de la garniture mécanique", [
        Paragraph("1. Quand remplacer la garniture ?", H2),
        *txt("La garniture mécanique HM160-GM04 est remplacée toutes les 16 000 heures, ou plus tôt si une fuite "
             "supérieure à 10 gouttes par minute apparaît le long de l'arbre."),
        Paragraph("2. Étapes", H2),
        *txt("1) Consigner le groupe (procédure de consignation électrique). 2) Vidanger le corps de pompe. "
             "3) Déposer le moteur et l'accouplement. 4) Démonter la roue puis l'ancienne garniture. 5) Nettoyer "
             "les portées. 6) Monter la garniture neuve dans l'ordre de la vue éclatée, sans toucher les faces de "
             "frottement. 7) Remonter, réaligner l'accouplement (écart maximal 0,05 mm) et remettre en service "
             "progressivement."),
        *fig(eclate_garniture(), "Figure 8 : Vue éclatée de la garniture mécanique et ordre de montage"),
        *tab("Tableau 9 : Couples de serrage", [
            ["Assemblage", "Couple"], ["Écrou de roue", "90 N·m"], ["Vis du corps de pompe (M16)", "120 N·m"],
            ["Vis de la lanterne (M12)", "55 N·m"], ["Vis de l'accouplement (M10)", "40 N·m"]], [9, 6]),
        *tab("Tableau 10 : Outillage nécessaire", [
            ["Outil", "Usage"], ["Clé dynamométrique 20-200 N·m", "Serrages au couple"], ["Extracteur de roue", "Dépose de la roue"],
            ["Comparateur à cadran", "Alignement de l'accouplement"], ["Graisse silicone", "Montage des joints toriques"]], [7, 8])])


def doc07():
    build("07_guide_alarmes_automate.pdf", "Guide des alarmes de l'automate de la station", [
        Paragraph("1. Codes d'alarme", H2),
        *tab("Tableau 11 : Codes d'alarme affichés par l'automate AP-20", [
            ["Code", "Signification", "Conduite à tenir"], ["A01", "Niveau très bas dans la bâche (< 0,8 m)", "Arrêt automatique, attendre le remplissage"],
            ["A02", "Pression de refoulement trop haute (> 6 bar)", "Vérifier la vanne V10 et les clapets"],
            ["A03", "Défaut thermique du moteur", "Laisser refroidir, contrôler la ventilation"],
            ["A04", "Vibrations excessives (> 7,1 mm/s)", "Appliquer le logigramme de la figure 9"],
            ["A05", "Pression d'aspiration trop basse (< -0,10 bar)", "Risque de cavitation : réduire le débit"],
            ["A06", "Défaut variateur", "Lire le code sur le variateur (notice VF-45)"],
            ["A07", "Débit nul avec pompe en marche", "Vérifier l'amorçage et la vanne d'aspiration"],
            ["A08", "Perte de communication avec le centre de conduite", "Vérifier le modem et l'antenne"]], [1.6, 7, 6.4]),
        Paragraph("2. Seuils de vibration", H2),
        *tab("Tableau 12 : Seuils de vibration des groupes (vitesse efficace)", [
            ["Zone", "Vitesse vibratoire", "Décision"], ["Bon", "jusqu'à 2,8 mm/s", "Fonctionnement normal"],
            ["Acceptable", "de 2,8 à 7,1 mm/s", "Surveillance renforcée"], ["Dangereux", "au-delà de 7,1 mm/s", "Arrêt et diagnostic"]], [4, 5, 6]),
        *fig(logigramme(), "Figure 9 : Logigramme de diagnostic d'une vibration excessive", 13)])


def doc08():
    build("08_rapport_exploitation_octobre_2026.pdf", "Rapport d'exploitation - octobre 2026", [
        Paragraph("1. Synthèse du mois", H2),
        *txt("En octobre 2026, la station a pompé 228 600 m³, soit 7 374 m³ par jour en moyenne, légèrement sous "
             "l'objectif de 7 500 m³/jour. L'écart s'explique par l'arrêt du groupe P2 le 21 octobre (cavitation) "
             "et par sa remise en service le 22 octobre au matin."),
        *fig(debits_octobre(), "Figure 10 : Volumes pompés par jour en octobre 2026"),
        *tab("Tableau 13 : Heures de fonctionnement des groupes", [
            ["Groupe", "Heures du mois", "Compteur total", "Prochaine garniture à"],
            ["P1", "498 h", "15 620 h", "16 000 h"], ["P2", "452 h", "12 940 h", "16 000 h"], ["P3", "236 h", "9 310 h", "16 000 h"]], [3, 3.5, 4, 4.5]),
        *tab("Tableau 14 : Consommation électrique", [
            ["Indicateur", "Valeur"], ["Énergie consommée", "41 800 kWh"], ["Énergie spécifique", "0,183 kWh/m³"],
            ["Puissance maximale appelée", "84 kW"], ["Facteur de puissance moyen", "0,93"]], [8, 7]),
        *txt("Point d'attention : le groupe P1 atteindra 16 000 heures en novembre ; prévoir le remplacement de sa "
             "garniture mécanique.")])


def doc09():
    build("09_rapport_incident_cavitation_P2.pdf", "Rapport d'incident - Cavitation du groupe P2 (21 octobre 2026)", [
        Paragraph("1. Description de l'incident", H2),
        *txt("Le 21 octobre 2026 à 8 h 15, l'automate a déclenché l'alarme A05 (pression d'aspiration trop basse) "
             "sur le groupe P2, suivie d'un bruit de « cailloux » dans la pompe. Le groupe a été arrêté à 9 h 10. "
             "La cause était un encrassement de la crépine d'aspiration par des algues, qui a fait chuter la "
             "pression d'aspiration sous le seuil de cavitation."),
        *fig(pression_aspiration(), "Figure 11 : Pression d'aspiration du groupe P2 pendant l'incident"),
        *tab("Tableau 15 : Chronologie de l'incident", [
            ["Heure", "Événement"], ["8 h 15", "Alarme A05 sur P2"], ["8 h 40", "Arrivée du technicien d'astreinte"],
            ["9 h 10", "Arrêt et consignation de P2 ; démarrage de P3"], ["14 h 30", "Crépine nettoyée, roue inspectée"],
            ["22 oct. 7 h 00", "Remise en service de P2"]], [4, 11]),
        *tab("Tableau 16 : Mesures avant et après intervention", [
            ["Mesure", "Pendant l'incident", "Après nettoyage"], ["Pression d'aspiration", "-0,25 bar", "+0,30 bar"],
            ["Vibrations roulement côté pompe", "8,4 mm/s", "2,1 mm/s"], ["Débit du groupe", "95 m³/h", "158 m³/h"]], [6, 4.5, 4.5]),
        *txt("Actions décidées : nettoyage mensuel des crépines au lieu de trimestriel, et ajout d'un capteur de "
             "colmatage sur chaque crépine d'aspiration.")])


def doc10():
    build("10_analyse_vibratoire_2026.pdf", "Analyse vibratoire des groupes - campagne de septembre 2026", [
        Paragraph("1. Méthode", H2),
        *txt("Les vibrations sont mesurées en vitesse efficace (mm/s) sur les roulements côté pompe et côté moteur "
             "de chaque groupe, avec un collecteur portable. Le spectre permet d'identifier la cause : un pic à 1x "
             "la vitesse de rotation indique un balourd, un pic à 2x un désalignement."),
        *tab("Tableau 17 : Mesures de vibration de septembre 2026", [
            ["Groupe", "Roulement côté pompe", "Roulement côté moteur", "État"], ["P1", "4,6 mm/s", "3,1 mm/s", "Acceptable"],
            ["P2", "2,2 mm/s", "1,9 mm/s", "Bon"], ["P3", "1,8 mm/s", "1,6 mm/s", "Bon"]], [2.5, 4.5, 4.5, 3.5]),
        *fig(spectre(), "Figure 12 : Spectre vibratoire du groupe P1, roulement côté pompe"),
        Paragraph("2. Conclusion", H2),
        *txt("Le spectre du groupe P1 montre un pic dominant à 1x la vitesse de rotation (49,2 Hz) : la cause "
             "probable est un balourd de la roue, sans doute dû à un dépôt. Recommandation : nettoyer et équilibrer "
             "la roue de P1 lors du remplacement de sa garniture mécanique en novembre.")])


if __name__ == "__main__":
    for build_doc in (doc01, doc02, doc03, doc04, doc05, doc06, doc07, doc08, doc09, doc10):
        build_doc()
