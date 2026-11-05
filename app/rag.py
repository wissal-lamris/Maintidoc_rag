"""Le RAG (Retrieval-Augmented Generation) avec Qwen via Ollama.

    question ──> 1. récupérer : BGE-M3 (texte, tableaux, descriptions d'images) + CLIP (image)
             ──> 2. construire le prompt : consigne + extraits numérotés [1], [2]... + question
             ──> 3. générer : qwen2.5:3b, température 0
             ──> 4. rendre : réponse + sources + images citées
"""
import re

import ollama

from app.clip_images import search_images_clip
from app.config import LLM_MODEL, LLM_HAS_THINKING, OLLAMA_HOST
from app.describe import cached_description
from app.indexing import search

TOP_K = 4        # nombre d'extraits ramenés par BGE-M3
NUM_CTX = 8192   # taille de la "mémoire de lecture" de Qwen, en tokens (extraits + question + réponse)
IMAGE_MIN_SCORE = 0.6  # secours : score BGE-M3 minimal pour afficher une image non citée

SYSTEM_PROMPT = """You are MaintiDoc, an assistant that answers questions about a technical document.
Rules:
- Use ONLY the numbered excerpts given by the user. Do not use your own knowledge.
- Explain in your own words. Do not copy formulas or tables character by character.
- If the excerpts do not contain the answer, say that the document does not give this information.
- Answer in the same language as the question: a question in French gets an answer in French,
  a question in English gets an answer in English.
- Be clear and concise."""


def retrieve(question: str, top_k: int = TOP_K, source: str | None = None) -> list[dict]:
    """1. Récupérer les extraits utiles pour la question."""
    sources = []
    for r in search(question, top_k=top_k, source=source):
        m = r.node.metadata
        sources.append({"type": m["type"], "page": m["page"], "score": r.score, "via": "BGE-M3",
                        "source": m["source"],
                        "text": r.node.get_content(), "image_path": m.get("image_path")})

    # CLIP : l'image qui ressemble le plus à la question, si elle n'est pas déjà dans les extraits
    known = {s["image_path"] for s in sources if s["image_path"]}
    for score, node in search_images_clip(question, top_k=1, source=source):
        m = node.metadata
        if m["image_path"] in known:
            continue
        title = m["caption"] or f"Image de la page {m['page']}"
        description = cached_description(m["image_path"])
        sources.append({"type": "image", "page": m["page"], "score": score, "via": "CLIP",
                        "source": m["source"],
                        "text": f"{title}\n\nDescription de l'image : {description}" if description else title,
                        "image_path": m["image_path"]})

    for i, s in enumerate(sources, start=1):
        s["num"] = i
    return sources


def build_prompt(question: str, sources: list[dict]) -> str:
    """2. Mettre les extraits numérotés avant la question."""
    excerpts = "\n\n".join(f"[{s['num']}] ({s['type']}, page {s['page']})\n{s['text']}" for s in sources)
    # Un petit modèle cite plus facilement ses sources dans une dernière ligne qu'au milieu des phrases.
    return (f"Excerpts from the document:\n\n{excerpts}\n\n"
            f"Question: {question}\n\n"
            "At the end of your answer, add one last line: \"Sources : \" followed by the numbers "
            "of the excerpts you used, for example \"Sources : [1], [3]\".")


def answer(question: str, source: str | None = None) -> dict:
    """Pose la question au RAG. Retourne la réponse, les sources et les images citées."""
    sources = retrieve(question, source=source)
    response = ollama.Client(host=OLLAMA_HOST).chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(question, sources)}],
        options={"temperature": 0, "num_ctx": NUM_CTX},  # 3. générer, température 0
        **({"think": False} if LLM_HAS_THINKING else {}),     # Qwen 3.5 : pas de « réflexion » cachée
    )
    text = response["message"]["content"].strip()

    # 4. Les images à afficher : celles que Qwen a citées dans sa réponse ([2], [5]...)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    images = []
    for s in sources:
        if s["num"] in cited and s["image_path"] and s["image_path"] not in images:
            images.append(s["image_path"])
    # Secours si Qwen n'a rien cité : les images trouvées par BGE-M3 avec un bon score
    if not cited:
        images = [s["image_path"] for s in sources
                  if s["image_path"] and s["via"] == "BGE-M3" and s["score"] >= IMAGE_MIN_SCORE]
    return {"question": question, "answer": text, "sources": sources, "images": images}
