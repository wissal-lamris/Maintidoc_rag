"""Poser des questions à l'agent LangChain (Ollama doit être ouvert).

La 2e question est une question de suite : l'agent doit comprendre grâce à l'historique.

Lancement :
    python -m scripts.ask_agent
    python -m scripts.ask_agent "Montre-moi le synoptique de la station"
"""
import sys
import time

from app.agent import ask

questions = sys.argv[1:] or [
    "Quelle est la puissance du moteur de la pompe HM-160 ?",
    "Et sa vitesse de rotation ?",
]

history = []
for q in questions:
    start = time.time()
    result = ask(q, history)
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": result["answer"]}]
    print(f"\n=== Question : {q}")
    print(f"Outils choisis par l'agent : {', '.join(result['tools']) or 'aucun'}")
    print(f"\nRéponse ({time.time() - start:.0f} s) :\n{result['answer']}")
    print("\nSources :")
    for s in result["sources"]:
        print(f"  [{s['num']}] {s['type']:<7} | page {s['page']:<2} | {s['via']:<6} | score {s['score']:.3f}")
    for path in result["images"]:
        print(f"  Image citée : {path}")
