"""Générer une description textuelle de chaque image avec Qwen2.5-VL.

CLIP sait retrouver une image, mais il ne sait pas écrire. Qwen2.5-VL (modèle "vision" d'Ollama)
regarde l'image et la décrit en texte. Cette description est ensuite indexée avec BGE-M3,
comme n'importe quel passage du PDF.

Les descriptions sont gardées dans data/images/<nom du pdf>/descriptions.json :
sur un PC sans GPU, chacune prend 1 à 3 minutes, on ne les recalcule donc pas à chaque indexation.
"""
import io
import json
from pathlib import Path

import ollama
from PIL import Image

from app.config import OLLAMA_HOST, ROOT, VISION_MODEL

MAX_SIDE = 768  # image réduite avant l'envoi : plus rapide sur CPU, assez net pour un schéma

PROMPT = (
    "You are describing a figure taken from a technical document. "
    "Describe precisely what the figure shows: the blocks or elements, the labels written in it, "
    "how they are connected (arrows, order, inputs and outputs) and what the figure explains. "
    "Follow the arrows from the inputs to the output to give the steps in the right order. "
    "The caption may describe several sub-figures (for example left and right): this image shows "
    "only ONE of them, so say which one and describe only what is visible in this image. "
    "Write 5 to 10 sentences, in English, with no introduction."
)


def _image_bytes(path: Path) -> bytes:
    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def describe_image(image_path: str, caption: str = "", force: bool = False, position: str = "") -> str:
    """Retourne la description de l'image (depuis le fichier de cache si elle existe déjà).
    force=True : refait la description même si elle existe."""
    full_path = ROOT / image_path
    cache_file = full_path.parent / "descriptions.json"
    cache = json.loads(cache_file.read_text("utf-8")) if cache_file.exists() else {}
    if full_path.name in cache and not force:
        return cache[full_path.name]

    prompt = PROMPT + (f"\nThe caption printed under the figure is: \"{caption}\"" if caption else "")
    if position:
        prompt += (f"\nThis image is only the {position.upper()} part of the figure. Use what the caption "
                   f"says about the {position} part to identify it, and describe only that part.")
    response = ollama.Client(host=OLLAMA_HOST).chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [_image_bytes(full_path)]}],
        options={"temperature": 0},  # température 0 : même image -> même description
        keep_alive=0,                # libère la RAM dès que la description est écrite
    )
    description = response["message"]["content"].strip()

    cache[full_path.name] = description
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), "utf-8")
    return description


def cached_description(image_path: str) -> str:
    """Description déjà calculée (sans appeler le modèle). Chaîne vide si elle n'existe pas."""
    cache_file = (ROOT / image_path).parent / "descriptions.json"
    if not cache_file.exists():
        return ""
    return json.loads(cache_file.read_text("utf-8")).get(Path(image_path).name, "")


def add_descriptions(nodes) -> None:
    """Remplace le texte provisoire des images (leur légende) par légende + description."""
    images = [n for n in nodes if n.metadata["type"] == "image"]
    for i, node in enumerate(images, start=1):
        print(f"  Description de l'image {i}/{len(images)} : {node.metadata['image_name']} ...")
        description = describe_image(node.metadata["image_path"], node.metadata["caption"],
                                     position=node.metadata.get("position", ""))
        title = node.metadata["caption"] or f"Image de la page {node.metadata['page']}"
        node.text = f"{title}\n\nDescription de l'image : {description}"
