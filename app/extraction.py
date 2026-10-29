"""Extraire le texte, les tableaux et les images d'un PDF avec LlamaIndex.

Chaque morceau devient un "node" LlamaIndex avec des métadonnées :
    type        : "texte", "tableau" ou "image"
    page        : numéro de page dans le PDF
    source      : nom du fichier PDF
    image_path  : chemin du fichier image (pour les images seulement)
"""
from pathlib import Path

import pymupdf
import pymupdf4llm
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, IMAGE_DIR, ROOT

MIN_IMAGE_SIDE = 60  # on ignore les très petites images (logos, icônes)


def extract_text(pdf_path: Path) -> list[TextNode]:
    """Texte : LlamaIndex lit le PDF page par page, puis le découpe en passages."""
    pages = pymupdf4llm.LlamaMarkdownReader().load_data(str(pdf_path))
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = []
    for page_doc in pages:
        page_num = page_doc.metadata["page"]
        for chunk in splitter.split_text(page_doc.text):
            if len(chunk.strip()) < 30:
                continue
            nodes.append(TextNode(text=chunk, metadata={
                "type": "texte", "page": page_num, "source": pdf_path.name}))
    return nodes


def _caption_near(page: pymupdf.Page, bbox: pymupdf.Rect, prefix: str) -> str:
    """Cherche la légende ("Figure 1: ..." ou "Table 2: ...") juste au-dessus ou en dessous."""
    best, best_dist = "", 1e9
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = " ".join(text.split())
        if not text.lower().startswith(prefix):
            continue
        dist = min(abs(y0 - bbox.y1), abs(bbox.y0 - y1))
        if dist < best_dist and dist < 120:
            best, best_dist = text, dist
    return best


def extract_tables(pdf_path: Path) -> list[TextNode]:
    """Tableaux : détectés sur chaque page, puis convertis en texte (format Markdown)."""
    nodes = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for table in page.find_tables().tables:
                caption = _caption_near(page, pymupdf.Rect(table.bbox), "table")
                text = (caption + "\n\n" if caption else "") + table.to_markdown()
                nodes.append(TextNode(text=text, metadata={
                    "type": "tableau", "page": page.number + 1, "source": pdf_path.name,
                    "caption": caption}))
    return nodes


def extract_images(pdf_path: Path) -> list[TextNode]:
    """Images : enregistrées dans data/images/<nom du pdf>/, avec leur légende comme texte provisoire.
    (À l'étape suivante, Qwen-VL remplacera ce texte par une vraie description de l'image.)"""
    out_dir = IMAGE_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_nodes = []
            for i, info in enumerate(page.get_image_info(xrefs=True), start=1):
                if not info["xref"]:
                    continue
                pix = pymupdf.Pixmap(doc, info["xref"])
                if pix.width < MIN_IMAGE_SIDE or pix.height < MIN_IMAGE_SIDE:
                    continue
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                image_name = f"page{page.number + 1}_image{i}.png"
                image_path = out_dir / image_name
                pix.save(image_path)
                caption = _caption_near(page, pymupdf.Rect(info["bbox"]), "figure")
                page_nodes.append((info["bbox"][0], TextNode(
                    text=caption or f"Image de la page {page.number + 1}", metadata={
                        "type": "image", "page": page.number + 1, "source": pdf_path.name,
                        "caption": caption, "image_path": image_path.relative_to(ROOT).as_posix(),
                        "image_name": image_name, "position": ""})))
            _set_positions(page_nodes)
            nodes += [node for _, node in page_nodes]
    return nodes


def _set_positions(page_nodes) -> None:
    """Plusieurs images avec la même légende (ex. "(left) ... (right) ...") : on indique à chacune
    sa position, de gauche à droite, pour que Qwen-VL sache quelle partie de la figure il décrit."""
    groups = {}
    for x0, node in page_nodes:
        if node.metadata["caption"]:
            groups.setdefault(node.metadata["caption"], []).append((x0, node))
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda item: item[0])
        names = ["left", "right"] if len(group) == 2 else [f"part {i} of {len(group)}"
                                                            for i in range(1, len(group) + 1)]
        for name, (_, node) in zip(names, group):
            node.metadata["position"] = name


def extract_pdf(pdf_path: str | Path) -> list[TextNode]:
    """Extrait tout le PDF : texte + tableaux + images."""
    pdf_path = Path(pdf_path)
    return extract_text(pdf_path) + extract_tables(pdf_path) + extract_images(pdf_path)
