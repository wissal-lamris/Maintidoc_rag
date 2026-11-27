# MaintiDoc — RAG agentique multimodal (partie 1)

Projet intégrateur « IA générative multimodale et agentique » (EMSI).
L'application répond à des questions sur un PDF technique (texte, tableaux et images),
en s'appuyant uniquement sur le contenu du document, avec des modèles open source qui tournent en local.

## Corpus de test : la station de pompage SP-Oulad Saleh

Le dossier `data/pdfs/` contient **10 documents techniques** d'une station de pompage d'eau potable **fictive**
de la région de Casablanca (station, équipements, valeurs et incidents inventés). Ils couvrent les principaux
documents d'un service de maintenance, avec au total **17 tableaux** et **12 schémas et graphiques** :

| N° | Document | Contenu multimodal |
|---|---|---|
| 01 | Dossier technique de la station | synoptique hydraulique, schéma électrique unifilaire, liste des équipements |
| 02 | Notice constructeur de la pompe HM-160 | courbe caractéristique, coupe de la pompe, caractéristiques, pièces de rechange |
| 03 | Notice du variateur de fréquence VF-45 | schéma de raccordement, paramètres, codes de défaut |
| 04 | Plan de maintenance préventive 2026 | périodicités, graissage, calendrier annuel |
| 05 | Procédure de consignation électrique | étapes, schéma de l'armoire, équipements de protection |
| 06 | Procédure de remplacement de la garniture mécanique | vue éclatée, couples de serrage, outillage |
| 07 | Guide des alarmes de l'automate | codes d'alarme, seuils de vibration, logigramme de diagnostic |
| 08 | Rapport d'exploitation d'octobre 2026 | volumes pompés par jour, heures de marche, énergie |
| 09 | Rapport d'incident : cavitation du groupe P2 | courbe de pression, chronologie, mesures avant / après |
| 10 | Analyse vibratoire de septembre 2026 | spectre vibratoire, mesures par groupe |

Les documents sont générés par `scripts/generer_documents.py` (ReportLab et matplotlib).

## Architecture

```
PDF ──> extraction (LlamaIndex / PyMuPDF) ──> texte, tableaux, images (PNG)
            │
            ├── texte + tableaux + descriptions d'images (Qwen2.5-VL) ──> BGE-M3 ──> Qdrant « chunks »
            └── images ──────────────────────────────────────────────────> CLIP ───> Qdrant « images_clip »

Interface Streamlit ──HTTP──> API FastAPI ──> agent LangChain (Qwen 3.5 via Ollama, température 0)
               ├── chercher_texte       (BGE-M3, passages)
               ├── chercher_tableaux    (BGE-M3, tableaux)
               ├── chercher_images      (BGE-M3 sur les descriptions + CLIP texte -> image)
               ├── chercher_par_photo   (CLIP image -> image, photo jointe par l'utilisateur)
               ├── lire_page            (tout le contenu d'une page)
               └── inventaire_document  (liste des tableaux et des figures)
           ──> réponse dans la langue de la question + sources + image d'origine
               + trace du raisonnement de l'agent (Streamlit)
```

| Rôle | Outil |
|---|---|
| Extraction du PDF | LlamaIndex (`pymupdf4llm`), PyMuPDF |
| Embeddings du texte | `BAAI/bge-m3` (1024 dimensions, multilingue) |
| Embeddings des images | CLIP `clip-ViT-B-32` (512 dimensions) |
| Descriptions des images | `qwen2.5vl:3b` via Ollama |
| Base vectorielle | Qdrant (distance cosinus), lancé avec Docker |
| Génération des réponses | `qwen3.5:4b` via Ollama, température 0, mode « réflexion » désactivé |
| Agent | LangChain (`create_agent`) |
| API | FastAPI |
| Interface | Streamlit (ne fait qu'afficher : tout passe par l'API) |

## Structure

```
app/
  config.py        réglages (lus depuis .env)
  extraction.py    extraction du texte, des tableaux et des images
  stores.py        connexion à Qdrant (collections, lecture complète)
  indexing.py      embeddings BGE-M3, indexation, recherche, inventaire, suppression
  clip_images.py   vecteurs CLIP des images, recherche texte -> image et image -> image
  describe.py      descriptions des images par Qwen2.5-VL
  rag.py           RAG simple (récupération, prompt, génération)
  agent.py         agent LangChain : 6 outils, gestion du contexte, trace du raisonnement
  client.py        client de l'API, utilisé par l'interface
api.py             API FastAPI (questions, documents, indexation, suppression, images)
scripts/           scripts de test de chaque étape
evaluation/        jeu de questions, script d'évaluation et rapport des résultats
streamlit_app.py   interface utilisateur
.streamlit/        thème de l'interface
docker-compose.yml base Qdrant
```

## Installation

Prérequis : Python 3.11 ou plus, Docker, Ollama.

```bash
# 1. Base vectorielle Qdrant
docker compose up -d

# 2. Modèles Ollama
ollama pull qwen3.5:4b
ollama pull qwen2.5vl:3b

# 3. Environnement Python
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux / macOS : source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # Linux / macOS : cp .env.example .env
```

## Utilisation

```bash
# Indexer les 10 PDF (texte, tableaux, descriptions d'images, vecteurs CLIP)
python -m scripts.index_pdf
python -m scripts.index_clip

# Lancer l'application : deux programmes, dans deux terminaux
uvicorn api:app --port 8000          # terminal 1 : l'API (documentation sur http://127.0.0.1:8000/docs)
streamlit run streamlit_app.py       # terminal 2 : l'interface
```

### L'API

| Route | Rôle |
|---|---|
| `GET /sante` | état de l'API, base et modèles utilisés |
| `GET /documents` | documents indexés et nombre de passages, tableaux et images |
| `POST /documents` | envoyer un PDF : il est indexé (texte, tableaux, descriptions d'images, CLIP) |
| `DELETE /documents/{nom}` | supprimer un document de la base |
| `POST /questions` | poser une question à l'agent (historique, document ciblé et photo facultatifs) |
| `GET /fichiers/{chemin}` | une image du dossier `data/` (figure extraite ou photo jointe) |

## L'interface

- **Zone de question** : on pose sa question en français ou en anglais. Le bouton **+** permet de joindre
  un **PDF** (il est indexé directement dans la discussion) ou une **image** (l'agent cherche les figures
  du document qui lui ressemblent).
- **Questions suggérées** au démarrage, cliquables.
- **Sous chaque réponse** : le document utilisé, les outils choisis, le temps de réponse, et trois onglets :
  *Raisonnement* (chaque étape de l'agent), *Images* (les figures d'origine) et *Sources* (extraits avec page et score).
- **Barre latérale** : nombre de passages, tableaux et images dans la base, choix du document utilisé pour
  répondre (un document précis ou tous), liste des documents indexés avec un bouton de suppression,
  et effacement de la conversation.

Scripts de test : `test_extraction`, `test_search`, `test_clip`, `describe_images`, `ask` (RAG simple)
et `ask_agent` (agent), à lancer avec `python -m scripts.<nom>`.

## L'agent

| Point de la consigne | Mise en œuvre |
|---|---|
| Sélection d'outils | 6 outils ; l'agent choisit selon la question (texte, tableaux, figures, photo, page, inventaire) |
| Interrogation de l'index | Qdrant (BGE-M3 et CLIP) avec filtres par document, type de morceau et page |
| Gestion du contexte | 3 derniers échanges gardés, extraits sans doublons, budget de 7 000 caractères, 6 étapes au maximum (sinon RAG simple en secours) |
| Génération | Qwen 3.5 à température 0, réponse dans la langue de la question, sources citées, images d'origine affichées |

Quand plusieurs images partagent une même légende (par exemple « (left) … (right) … »), l'extraction
indique à chacune sa position, pour que Qwen2.5-VL décrive la bonne partie de la figure.

## Évaluation

Le dossier `evaluation/` contient 10 questions sur le corpus (7 en français, 3 en anglais), avec la
réponse attendue, le document, la page et le type de source où elle se trouve, dont une question hors
document pour vérifier que l'agent n'invente pas.

```bash
python -m evaluation.evaluate --recherche-seule   # recherche seulement, sans LLM (~1 min)
python -m evaluation.evaluate                     # recherche + réponses de l'agent (~2 à 3 min par question)
```

Indicateurs mesurés :
- **recherche** : bonne page dans les 4 premiers résultats (Hit@4), en 1re position (Hit@1), rang moyen inverse (MRR), bon type de source ;
- **agent** : réponse juste (mots-clés attendus), langue de la question respectée, bon outil choisi, sources citées, temps de réponse.

Les résultats sont écrits dans `evaluation/rapport.md` (lisible sur GitHub) et `evaluation/resultats.json`.

## Limites connues

- Sans GPU, une description d'image prend 1 à 3 minutes et une réponse de l'agent 1 à 3 minutes.

