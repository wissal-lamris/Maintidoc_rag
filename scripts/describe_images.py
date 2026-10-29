"""Faire décrire les images du PDF par Qwen2.5-VL et afficher les descriptions.

Lancement (Ollama doit être ouvert) :
    python -m scripts.describe_images                     # décrit les images pas encore décrites
    python -m scripts.describe_images page2_image1.png    # refait la description de cette image
"""
import sys
import time

from app.config import PDF_DIR
from app.describe import describe_image
from app.extraction import extract_images

to_redo = set(sys.argv[1:])
nodes = [n for pdf in sorted(PDF_DIR.glob("*.pdf")) for n in extract_images(pdf)]
for node in nodes:
    m = node.metadata
    if to_redo and m["image_name"] not in to_redo:
        continue
    print(f"\n=== {m['image_name']} (page {m['page']}) : {m['caption'][:60]}")
    start = time.time()
    print(f"Position dans la figure : {m['position'] or '-'}")
    print(describe_image(m["image_path"], m["caption"], force=m["image_name"] in to_redo,
                         position=m["position"]))
    print(f"--- {time.time() - start:.0f} s")
