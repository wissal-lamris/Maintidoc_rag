"""Évaluation de MaintiDoc sur un jeu de questions dont on connaît la réponse.

Deux niveaux :
  1. Recherche (sans LLM, rapide) : la bonne page est-elle retrouvée ? à quel rang ?
  2. Agent (avec Qwen, lent sur CPU) : la réponse contient-elle la bonne information,
     dans la bonne langue, avec le bon outil et des sources citées ?

Lancement (depuis le dossier du projet) :
    python -m evaluation.evaluate --recherche-seule     # niveau 1 seulement (~1 min)
    python -m evaluation.evaluate                       # niveaux 1 et 2 (~2 à 3 min par question)
    python -m evaluation.evaluate --questions Q1,Q8     # seulement certaines questions

Résultats : evaluation/resultats.json (détail) et evaluation/rapport.md (tableaux lisibles sur GitHub).
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from app.config import LLM_MODEL

HERE = Path(__file__).resolve().parent
TOP_K = 4

FR_WORDS = {"le", "la", "les", "des", "est", "une", "du", "et", "en", "pour", "dans", "qui", "sont"}
EN_WORDS = {"the", "is", "of", "and", "to", "in", "are", "for", "with", "that", "uses", "by"}


def langue(texte: str) -> str:
    """Devine la langue d'une réponse en comptant quelques mots très courants."""
    mots = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", texte.lower())
    fr = sum(m in FR_WORDS for m in mots)
    en = sum(m in EN_WORDS for m in mots)
    return "fr" if fr >= en else "en"


def mots_cles_ok(texte: str, groupes: list[list[str]]) -> bool:
    """Chaque groupe doit être présent ; dans un groupe, une seule variante suffit."""
    texte = texte.lower()
    return all(any(v.lower() in texte for v in groupe) for groupe in groupes)


def _bon(source: str, page: int, q: dict) -> bool:
    """Le morceau vient-il d'un des documents attendus, à une des pages attendues ?"""
    return source in q["documents"] and page in q["pages"]


def eval_recherche(q: dict) -> dict:
    """Niveau 1 : parmi les 4 morceaux les plus proches (tous documents), trouve-t-on la bonne page ?"""
    from app.indexing import search
    resultats = search(q["question"], top_k=TOP_K)
    trouves = [(r.node.metadata["source"], r.node.metadata["page"]) for r in resultats]
    pages = [f"{s[:2]}-p{p}" for s, p in trouves]  # ex. "07-p1" = document 07, page 1
    types = [r.node.metadata["type"] for r in resultats]
    rang = next((i for i, (s, p) in enumerate(trouves, start=1) if _bon(s, p, q)), None)
    return {"pages_trouvees": pages, "types_trouves": types, "rang_bonne_page": rang,
            "page_trouvee": rang is not None, "type_trouve": q["type"] in types,
            "rr": 1 / rang if rang else 0.0}


def eval_agent(q: dict) -> dict:
    """Niveau 2 : on pose la question à l'agent et on vérifie sa réponse."""
    from app.agent import ask
    start = time.time()
    r = ask(q["question"])  # l'agent cherche dans tous les documents
    duree = time.time() - start
    pages_sources = sorted({f"{s['source'][:2]}-p{s['page']}" for s in r["sources"]})
    return {
        "reponse": r["answer"], "outils": r["tools"], "pages_sources": pages_sources,
        "images": r["images"], "duree_s": round(duree, 1),
        "reponse_juste": mots_cles_ok(r["answer"], q["mots_cles"]),
        "bonne_langue": langue(r["answer"]) == q["langue"],
        "bon_outil": (not q["outils"]) or any(o in r["tools"] for o in q["outils"]),
        "source_citee": "[" in r["answer"],
        "page_dans_sources": (not q["pages"]) or any(_bon(s["source"], s["page"], q) for s in r["sources"]),
    }


def pct(valeurs) -> str:
    valeurs = list(valeurs)
    return f"{100 * sum(valeurs) / len(valeurs):.0f} %" if valeurs else "-"


def oui(b) -> str:
    return "✅" if b else "❌"


def rapport(questions, res, avec_agent: bool) -> str:
    lignes = [f"# Évaluation de MaintiDoc\n",
              f"Corpus : station de pompage SP-Oulad Saleh (10 documents) · {len(questions)} questions · "
              f"{datetime.now():%d/%m/%Y %H:%M}\n",
              "## 1. Recherche (BGE-M3 + Qdrant, sans LLM)\n"]
    rech = [res[q["id"]]["recherche"] for q in questions if not q.get("hors_document")]
    lignes += ["| Indicateur | Valeur |", "|---|---|",
               f"| Bonne page dans les {TOP_K} premiers résultats (Hit@{TOP_K}) | {pct(r['page_trouvee'] for r in rech)} |",
               f"| Bonne page en 1re position (Hit@1) | {pct(r['rang_bonne_page'] == 1 for r in rech)} |",
               f"| Rang moyen inverse (MRR) | {sum(r['rr'] for r in rech) / len(rech):.2f} |" if rech else "",
               f"| Bon type de source trouvé (texte / tableau / image) | {pct(r['type_trouve'] for r in rech)} |\n"]
    if avec_agent:
        ag = [res[q["id"]]["agent"] for q in questions]
        lignes += [f"## 2. Agent (LangChain + {LLM_MODEL}, température 0)\n",
                   "| Indicateur | Valeur |", "|---|---|",
                   f"| Réponse juste (mots-clés attendus) | {pct(a['reponse_juste'] for a in ag)} |",
                   f"| Réponse dans la langue de la question | {pct(a['bonne_langue'] for a in ag)} |",
                   f"| Outil attendu choisi par l'agent | {pct(a['bon_outil'] for a in ag)} |",
                   f"| Sources citées | {pct(a['source_citee'] for a in ag)} |",
                   f"| Bonne page parmi les sources | {pct(a['page_dans_sources'] for a in ag)} |",
                   f"| Temps moyen par question (CPU) | {sum(a['duree_s'] for a in ag) / len(ag):.0f} s |\n"]
    lignes += ["## 3. Détail par question\n"]
    if avec_agent:
        lignes += ["| # | Question | Page | Juste | Langue | Outil(s) choisi(s) | Temps |", "|---|---|---|---|---|---|---|"]
        for q in questions:
            r, a = res[q["id"]], res[q["id"]]["agent"]
            page = "hors doc." if q.get("hors_document") else (r["recherche"]["rang_bonne_page"] or "❌")
            lignes.append(f"| {q['id']} | {q['question']} | {page} | {oui(a['reponse_juste'])} | "
                          f"{oui(a['bonne_langue'])} | {', '.join(a['outils'])} | {a['duree_s']:.0f} s |")
    else:
        lignes += ["| # | Question | Attendu | Trouvé (document-page) | Rang |", "|---|---|---|---|---|"]
        for q in questions:
            if q.get("hors_document"):
                continue
            r = res[q["id"]]["recherche"]
            attendu = ", ".join(f"{d[:2]}-p{p}" for d in q["documents"] for p in q["pages"])
            lignes.append(f"| {q['id']} | {q['question']} | {attendu} | {', '.join(r['pages_trouvees'])} | "
                          f"{r['rang_bonne_page'] or '❌'} |")
    lignes.append("\n*Rang* : position de la première source qui vient de la bonne page (1 = la meilleure).")
    return "\n".join(l for l in lignes if l is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recherche-seule", action="store_true", help="sans LLM (rapide)")
    parser.add_argument("--questions", help="ex. Q1,Q8")
    args = parser.parse_args()

    data = json.loads((HERE / "questions.json").read_text("utf-8"))
    questions = data["questions"]
    if args.questions:
        ids = {x.strip() for x in args.questions.split(",")}
        questions = [q for q in questions if q["id"] in ids]
    avec_agent = not args.recherche_seule

    res = {}
    for i, q in enumerate(questions, start=1):
        print(f"\n[{i}/{len(questions)}] {q['id']} : {q['question']}")
        res[q["id"]] = {"question": q["question"]}
        if not q.get("hors_document"):
            res[q["id"]]["recherche"] = eval_recherche(q)
            r = res[q["id"]]["recherche"]
            print(f"  Recherche : pages {r['pages_trouvees']} -> bonne page au rang {r['rang_bonne_page']}")
        if avec_agent:
            res[q["id"]]["agent"] = a = eval_agent(q)
            print(f"  Agent ({a['duree_s']:.0f} s) : juste={a['reponse_juste']} langue={a['bonne_langue']} "
                  f"outils={a['outils']}")

    (HERE / "resultats.json").write_text(json.dumps(
        {"date": datetime.now().isoformat(timespec="seconds"),
         "avec_agent": avec_agent, "resultats": res}, ensure_ascii=False, indent=2), "utf-8")
    (HERE / "rapport.md").write_text(rapport(questions, res, avec_agent), "utf-8")
    print("\nTerminé : evaluation/rapport.md et evaluation/resultats.json")


if __name__ == "__main__":
    main()
