"""Rendre le RAG agentique avec LangChain.

L'agent (Qwen2.5 via Ollama, température 0) choisit lui-même ses outils, lit ce qu'ils renvoient,
peut relancer une recherche ou lire une page entière, puis rédige la réponse.

    question (+ photo) ──> agent Qwen ──> choisit un outil :
                              ▲             chercher_texte / chercher_tableaux / chercher_images
                              │             chercher_par_photo / lire_page / inventaire_document
                              └── lit les extraits ◄──┘        (plusieurs étapes, au maximum MAX_STEPS)
                              │
                              └──> réponse + sources + images + trace du raisonnement

Les 4 points de la consigne :
  - sélection d'outils      : 6 outils, l'agent choisit selon la question
  - interrogation de l'index : Qdrant (BGE-M3 et CLIP) avec filtres par document, type et page
  - gestion du contexte     : historique limité, extraits sans doublons, budget de texte, nombre d'étapes limité
  - génération              : réponse dans la langue de la question, sources citées, images d'origine
"""
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from app.clip_images import search_by_image, search_images_clip
from app.config import LLM_MODEL, LLM_HAS_THINKING, OLLAMA_HOST
from app.describe import cached_description
from app.indexing import inventory, page_chunks, search

NUM_CTX = 8192
HISTORY_TURNS = 3        # échanges précédents gardés en mémoire
CONTEXT_BUDGET = 7000    # nombre maximal de caractères d'extraits envoyés à Qwen pour une question
EXCERPT_MAX = 1200       # longueur maximale d'un extrait
MAX_STEPS = 6            # nombre maximal d'appels d'outils avant de forcer la réponse
IMAGE_MIN_SCORE = 0.6    # secours : score BGE-M3 minimal pour afficher une image non citée
PHOTO_MARGIN = 0.05      # photo : figures gardées si leur ressemblance est à moins de 5 points de la meilleure

SYSTEM_PROMPT = """You are MaintiDoc, an assistant that answers questions about technical documents.
Your tools:
- chercher_texte: passages of the document (explanations, definitions, methods, numbers in the text).
- chercher_tableaux: tables of the document (scores, results, comparisons).
- chercher_images: figures and diagrams found from a text description (architecture, schemas).
- chercher_par_photo: figures that look like the photo attached by the user. Use it only if a photo is attached.
- lire_page: the whole content of one page, when an excerpt is cut or you need the context around it.
- inventaire_document: the list of the tables and figures of the document, for questions like
  "which tables / figures does the document contain?".
Rules:
- Always use at least one tool before answering. Use several tools if the question needs it.
- Use ONLY what the tools return. Do not use your own knowledge.
- If the tools do not give the answer, say that the document does not give this information.
- Explain in your own words. Do not copy formulas or tables character by character.
- Answer in the same language as the question: a question in French gets an answer in French,
  a question in English gets an answer in English.
- At the end of your answer, add one last line: "Sources : " followed by the numbers of the
  excerpts you used, for example "Sources : [1], [3]"."""

# État de la question en cours (remis à zéro à chaque question)
_found: list[dict] = []       # extraits numérotés [1], [2]...
_trace: list[dict] = []       # étapes de l'agent (outil, argument, résultat)
_state = {"source": None, "photo": None, "chars": 0}


# ---------- Gestion du contexte ----------
def _add(sources: list[dict]) -> tuple[str, int]:
    """Numérote les nouveaux extraits (sans doublons, dans la limite du budget) et les renvoie en texte."""
    known = {(s["source"], s["page"], s["text"]) for s in _found}
    lines, added = [], 0
    for s in sources:
        if (s["source"], s["page"], s["text"]) in known:
            continue
        text = s["text"] if len(s["text"]) <= EXCERPT_MAX else s["text"][:EXCERPT_MAX] + " [...]"
        if _state["chars"] + len(text) > CONTEXT_BUDGET:
            lines.append("Context budget reached: answer now with the excerpts you already have.")
            break
        _state["chars"] += len(text)
        s["num"] = len(_found) + 1
        _found.append(s)
        added += 1
        lines.append(f"[{s['num']}] ({s['type']}, page {s['page']}, {s['source']})\n{text}")
    return ("\n\n".join(lines) or "No new excerpt found."), added


def _step(outil: str, argument: str, resultat: str) -> None:
    _trace.append({"outil": outil, "argument": argument, "resultat": resultat})


def _bge(requete: str, top_k: int, type_filter: str) -> list[dict]:
    return [{"type": r.node.metadata["type"], "page": r.node.metadata["page"], "score": r.score,
             "via": "BGE-M3", "source": r.node.metadata["source"], "text": r.node.get_content(),
             "image_path": r.node.metadata.get("image_path")}
            for r in search(requete, top_k=top_k, type_filter=type_filter, source=_state["source"])]


def _clip_source(score: float, node, via: str) -> dict:
    m = node.metadata
    title = m["caption"] or f"Image de la page {m['page']}"
    description = cached_description(m["image_path"])
    return {"type": "image", "page": m["page"], "score": score, "via": via, "source": m["source"],
            "text": f"{title}\n\nDescription de l'image : {description}" if description else title,
            "image_path": m["image_path"]}


# ---------- Les 6 outils ----------
@tool
def chercher_texte(requete: str) -> str:
    """Search the text passages of the document. Use it for explanations, definitions and methods."""
    text, n = _add(_bge(requete, 3, "texte"))
    _step("chercher_texte", requete, f"{n} passage(s)")
    return text


@tool
def chercher_tableaux(requete: str) -> str:
    """Search the tables of the document. Use it for numbers, scores, results and comparisons."""
    text, n = _add(_bge(requete, 2, "tableau"))
    _step("chercher_tableaux", requete, f"{n} tableau(x)")
    return text


@tool
def chercher_images(requete: str) -> str:
    """Search the figures and diagrams of the document from a text description."""
    sources = _bge(requete, 2, "image")  # par leur description (Qwen2.5-VL)
    known = {s["image_path"] for s in sources}
    for score, node in search_images_clip(requete, top_k=1, source=_state["source"]):  # par CLIP
        if node.metadata["image_path"] not in known:
            sources.append(_clip_source(score, node, "CLIP"))
    text, n = _add(sources)
    _step("chercher_images", requete, f"{n} figure(s)")
    return text


@tool
def chercher_par_photo() -> str:
    """Find the figures of the document that look like the photo attached by the user."""
    if not _state["photo"]:
        _step("chercher_par_photo", "-", "aucune photo jointe")
        return "No photo is attached to the question."
    results = search_by_image(_state["photo"], top_k=3, source=_state["source"])
    if not results:
        _step("chercher_par_photo", "photo jointe", "aucune figure")
        return "No figure found."
    best = results[0][0]
    # On ne garde que les figures presque aussi ressemblantes que la meilleure
    results = [(score, node) for score, node in results if score >= best - PHOTO_MARGIN]
    sources = []
    for score, node in results:
        s = _clip_source(score, node, "CLIP-photo")
        s["text"] = f"Similarity with the photo: {score:.0%}\n{s['text']}"
        sources.append(s)
    text, n = _add(sources)
    _step("chercher_par_photo", "photo jointe",
          f"{n} figure(s) ; meilleure ressemblance {best:.0%} ({results[0][1].metadata['image_name']})")
    return "The first figure is the most similar to the photo.\n\n" + text


@tool
def lire_page(page: int) -> str:
    """Read the whole content of one page of the document (text, tables and figure descriptions)."""
    chunks = page_chunks(page, _state["source"])
    sources = [{"type": c["type"], "page": c["page"], "score": 1.0, "via": "page", "source": c["source"],
                "text": c["text"], "image_path": c.get("image_path")} for c in chunks]
    text, n = _add(sources)
    _step("lire_page", str(page), f"{n} morceau(x)" if chunks else "page introuvable")
    return text if chunks else f"Page {page} not found."


@tool
def inventaire_document() -> str:
    """List the tables and figures of the document, with their page and caption."""
    inv = inventory(_state["source"])
    lines = [f"Documents: {', '.join(inv['documents'])} ({inv['pages']} pages, {inv['passages']} text passages)",
             f"Tables ({len(inv['tableaux'])}):"]
    lines += [f"- page {p}: {c}" for _, p, c in inv["tableaux"]]
    lines += [f"Figures ({len(inv['figures'])}):"]
    lines += [f"- page {p}: {c}" for _, p, c, _ in inv["figures"]]
    _step("inventaire_document", "-", f"{len(inv['tableaux'])} tableau(x), {len(inv['figures'])} figure(s)")
    # L'inventaire devient aussi une source citable
    text, _ = _add([{"type": "inventaire", "page": 0, "score": 1.0, "via": "inventaire",
                     "source": ", ".join(inv["documents"]), "text": "\n".join(lines), "image_path": None}])
    return text


TOOLS = [chercher_texte, chercher_tableaux, chercher_images, chercher_par_photo, lire_page,
         inventaire_document]
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        extra = {"reasoning": False} if LLM_HAS_THINKING else {}  # pas de « réflexion » cachée (lent sur CPU)
        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0, num_ctx=NUM_CTX,
                         keep_alive="30m", **extra)
        _agent = create_agent(llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    return _agent


def ask(question: str, history: list[dict] | None = None, source: str | None = None,
        photo: str | None = None) -> dict:
    """Pose la question à l'agent.
    history : échanges précédents [{"role": "user"/"assistant", "content": "..."}]
    source  : nom du PDF dans lequel chercher (None = tous les documents)
    photo   : chemin d'une photo jointe à la question (recherche image -> image)"""
    _found.clear()
    _trace.clear()
    _state.update(source=source, photo=photo, chars=0)

    messages = []
    for turn in (history or [])[-2 * HISTORY_TURNS:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    content = question + ("\n\n[The user attached a photo to this question.]" if photo else "")
    messages.append(HumanMessage(content=content))

    try:
        # recursion_limit compte les passages modèle + outils : 2 par étape, plus la réponse finale
        result = get_agent().invoke({"messages": messages},
                                    config={"recursion_limit": 2 * MAX_STEPS + 1})
        new_messages = result["messages"][len(messages):]
        text = new_messages[-1].content.strip()
    except Exception as e:  # trop d'étapes ou erreur du modèle : on passe au RAG simple
        _step("secours", type(e).__name__, "RAG simple")
        text = ""

    if not _trace or not text:
        # Secours : aucun outil utilisé ou pas de réponse -> RAG simple
        from app.rag import answer
        result = answer(question, source=source)
        result["tools"] = ["RAG simple (secours)"]
        result["trace"] = list(_trace) + [{"outil": "RAG simple", "argument": question,
                                           "resultat": f"{len(result['sources'])} extrait(s)"}]
        return result

    # Images à afficher : celles que l'agent a citées, sinon (secours) les images bien classées
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    images = []
    for s in _found:
        if s["num"] in cited and s["image_path"] and s["image_path"] not in images:
            images.append(s["image_path"])
    if not cited:
        images = [s["image_path"] for s in _found if s["image_path"]
                  and (s["via"] == "CLIP-photo" or (s["via"] == "BGE-M3" and s["score"] >= IMAGE_MIN_SCORE))]
    return {"question": question, "answer": text, "sources": list(_found), "images": images,
            "tools": [t["outil"] for t in _trace], "trace": list(_trace)}
