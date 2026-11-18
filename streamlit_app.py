"""Interface Streamlit de MaintiDoc : elle affiche, et demande tout le travail à l'API (api.py).

Lancement, dans deux terminaux (depuis le dossier du projet, Docker et Ollama ouverts) :
    uvicorn api:app --port 8000          (1. l'API)
    streamlit run streamlit_app.py       (2. l'interface)
Le thème (couleurs) est dans .streamlit/config.toml, le style (CSS) dans ce fichier.
"""
import html
import time

import streamlit as st

from app import client
from app.client import APIError

st.set_page_config(page_title="MaintiDoc", page_icon="🛠️", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Style ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0A1020; --panel: #111A2E; --panel-2: #16223B; --line: #22314F;
  --text: #E2E8F0; --muted: #8B9BB8; --sky: #38BDF8; --amber: #FBBF24; --violet: #A78BFA;
  --green: #34D399;
}
html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }
h1, h2, h3, .md-title { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

/* Fond "plan technique" : quadrillage léger */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(56,189,248,0.10), transparent 60%),
    radial-gradient(900px 400px at -10% 110%, rgba(167,139,250,0.08), transparent 60%),
    linear-gradient(rgba(56,189,248,0.035) 1px, transparent 1px) 0 0 / 32px 32px,
    linear-gradient(90deg, rgba(56,189,248,0.035) 1px, transparent 1px) 0 0 / 32px 32px,
    var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1150px; }

/* En-tête */
.md-hero {
  position: relative; padding: 28px 32px; border-radius: 20px; margin-bottom: 22px;
  background: linear-gradient(135deg, rgba(17,26,46,0.95), rgba(22,34,59,0.85));
  border: 1px solid var(--line); overflow: hidden;
}
.md-hero::before {
  content: ""; position: absolute; inset: 0; border-radius: 20px; padding: 1px;
  background: linear-gradient(120deg, var(--sky), transparent 35%, transparent 65%, var(--violet));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
}
.md-kicker { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--sky);
  text-transform: uppercase; letter-spacing: 0.18em; }
.md-title { font-size: 40px; font-weight: 700; margin: 6px 0 4px; color: #F8FAFC; }
.md-title span { background: linear-gradient(90deg, var(--sky), var(--violet));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.md-sub { color: var(--muted); font-size: 15px; margin-bottom: 16px; max-width: 720px; }
.md-pills { display: flex; flex-wrap: wrap; gap: 8px; }
.md-pill { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; padding: 5px 10px;
  border-radius: 999px; border: 1px solid var(--line); background: rgba(10,16,32,0.6); color: #CBD5E1; }
.md-pill b { color: var(--sky); font-weight: 500; }

/* Barre latérale */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1528, #0A1020);
  border-right: 1px solid var(--line); }
.md-side-title { font-family: 'Space Grotesk', sans-serif; font-size: 13px; text-transform: uppercase;
  letter-spacing: 0.14em; color: var(--muted); margin: 18px 0 10px; }
.md-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.md-stat { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 10px 8px; text-align: center; }
.md-stat .n { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; }
.md-stat .l { font-size: 11px; color: var(--muted); }
.md-doc { display: flex; align-items: center; gap: 8px; padding: 8px 10px; margin-bottom: 6px;
  border-radius: 10px; background: var(--panel); border: 1px solid var(--line); font-size: 13px;
  min-width: 0; }
.md-doc span.name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
[data-testid="stPopover"] button { padding: 0.25rem 0.4rem; }
.md-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green);
  box-shadow: 0 0 8px var(--green); flex: none; }

/* Boutons */
.stButton > button { border-radius: 12px; border: 1px solid var(--line); background: var(--panel);
  color: var(--text); transition: all .15s ease; }
.stButton > button:hover { border-color: var(--sky); color: var(--sky);
  box-shadow: 0 0 0 3px rgba(56,189,248,0.12); }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #0EA5E9, #8B5CF6);
  border: none; color: white; font-weight: 600; }

/* Messages */
[data-testid="stChatMessage"] { background: rgba(17,26,46,0.75); border: 1px solid var(--line);
  border-radius: 16px; padding: 14px 18px; backdrop-filter: blur(6px); }
[data-testid="stChatInput"] textarea { font-size: 15px; }

/* Puces : outils, durée */
.md-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 4px; }
.md-chip { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; padding: 4px 9px;
  border-radius: 8px; background: rgba(56,189,248,0.10); color: var(--sky);
  border: 1px solid rgba(56,189,248,0.25); }
.md-chip.doc { background: rgba(167,139,250,0.10); color: var(--violet); border-color: rgba(167,139,250,0.25); }
.md-chip.time { background: rgba(52,211,153,0.10); color: var(--green); border-color: rgba(52,211,153,0.25); }

/* Cartes de sources */
.md-src { background: var(--panel-2); border: 1px solid var(--line); border-left: 3px solid var(--c);
  border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; }
.md-src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 13px; }
.md-num { font-family: 'JetBrains Mono', monospace; font-weight: 500; color: var(--c); }
.md-badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; color: var(--c);
  background: color-mix(in srgb, var(--c) 14%, transparent); }
.md-meta { color: var(--muted); font-size: 12px; margin-left: auto; font-family: 'JetBrains Mono', monospace; }
.md-bar { height: 4px; border-radius: 4px; background: var(--line); margin: 4px 0 8px; overflow: hidden; }
.md-bar > div { height: 100%; background: var(--c); }
.md-txt { color: #CBD5E1; font-size: 13px; line-height: 1.5; white-space: pre-wrap; }

/* Trace du raisonnement de l'agent */
.md-step { display: flex; gap: 12px; align-items: flex-start; padding: 10px 12px; margin-bottom: 8px;
  border-radius: 12px; background: var(--panel-2); border: 1px solid var(--line); }
.md-step-n { flex: none; width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
  font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--bg);
  background: linear-gradient(135deg, var(--sky), var(--violet)); }
.md-step-tool { font-family: 'JetBrains Mono', monospace; color: var(--sky); font-size: 13px; }
.md-step-arg { color: #CBD5E1; font-size: 13px; }
.md-step-res { color: var(--green); font-size: 12px; font-family: 'JetBrains Mono', monospace; }

/* Cartes de suggestions */
.md-empty { text-align: center; color: var(--muted); padding: 18px 0 6px; }
</style>
""", unsafe_allow_html=True)

ALL_DOCS = "Tous les documents"
TYPE_COLORS = {"texte": "#38BDF8", "tableau": "#FBBF24", "image": "#A78BFA"}
SUGGESTIONS = [
    "Montre-moi le synoptique de la station",
    "Quel est le débit nominal de la pompe HM-160 ?",
    "Que faire si les vibrations dépassent 7,1 mm/s ?",
]

# ---------- Configuration de l'API ----------
try:
    CONFIG = client.sante()
except APIError as e:
    st.error(f"⚠️ {e}")
    st.stop()
BASE = CONFIG["base_vectorielle"]
LLM_MODEL, VISION_MODEL = CONFIG["llm"], CONFIG["vision"]

# ---------- En-tête ----------
st.markdown(f"""
<div class="md-hero">
  <div class="md-kicker">RAG agentique · multimodal · 100 % local</div>
  <div class="md-title">Mainti<span>Doc</span></div>
  <div class="md-sub">Posez vos questions sur un document technique : l'agent choisit où chercher
  (texte, tableaux ou schémas), répond dans votre langue et montre ses sources.</div>
  <div class="md-pills">
    <span class="md-pill"><b>extraction</b> LlamaIndex</span>
    <span class="md-pill"><b>vecteurs</b> {BASE}</span>
    <span class="md-pill"><b>texte</b> BGE-M3</span>
    <span class="md-pill"><b>images</b> CLIP · {VISION_MODEL}</span>
    <span class="md-pill"><b>LLM</b> {LLM_MODEL} · T=0</span>
    <span class="md-pill"><b>agent</b> LangChain</span>
    <span class="md-pill"><b>API</b> FastAPI</span>
  </div>
</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=20, show_spinner=False)
def get_stats():
    try:
        return client.documents()
    except APIError:
        return None


# ---------- Barre latérale ----------
with st.sidebar:
    stats = get_stats()
    st.markdown('<div class="md-side-title">Base de connaissances</div>', unsafe_allow_html=True)
    if stats:
        t = stats["types"]
        st.markdown(f"""
        <div class="md-stats">
          <div class="md-stat"><div class="n" style="color:#38BDF8">{t.get('texte', 0)}</div><div class="l">passages</div></div>
          <div class="md-stat"><div class="n" style="color:#FBBF24">{t.get('tableau', 0)}</div><div class="l">tableaux</div></div>
          <div class="md-stat"><div class="n" style="color:#A78BFA">{t.get('image', 0)}</div><div class="l">images</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="md-side-title">Document utilisé pour répondre</div>',
                    unsafe_allow_html=True)
        options = stats["sources"] + ([ALL_DOCS] if len(stats["sources"]) > 1 else [])
        if "next_doc" in st.session_state:  # document qui vient d'être indexé
            st.session_state.doc_choice = st.session_state.pop("next_doc")
        if options:
            if st.session_state.get("doc_choice") not in options:
                st.session_state.doc_choice = options[0]
            st.selectbox("Document", options, key="doc_choice", label_visibility="collapsed")

        st.markdown('<div class="md-side-title">Documents indexés</div>', unsafe_allow_html=True)
        for name in stats["sources"]:
            col_name, col_del = st.columns([4, 1], vertical_alignment="center")
            col_name.markdown(f'<div class="md-doc" title="{html.escape(name)}"><span class="md-dot"></span>'
                              f'<span class="name">{html.escape(name)}</span></div>',
                              unsafe_allow_html=True)
            with col_del.popover("🗑️", help=f"Supprimer {name}"):
                st.write(f"Supprimer **{name}** de la base ? Ses passages, tableaux, images et "
                         "le fichier PDF seront effacés.")
                if st.button("Supprimer définitivement", key=f"del_{name}", type="primary"):
                    with st.spinner("Suppression..."):
                        client.supprimer(name)
                    get_stats.clear()
                    st.rerun()
    else:
        st.warning(f"Base injoignable : vérifiez que Docker ({BASE}) est lancé.")

    st.markdown('<div class="md-side-title">Conversation</div>', unsafe_allow_html=True)
    if st.button("🗑️ Effacer la conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ---------- Conversation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


def source_card(s: dict) -> str:
    color = TYPE_COLORS.get(s["type"], "#38BDF8")
    width = max(4, min(100, int(s["score"] * 100)))
    text = html.escape(s["text"][:420] + ("…" if len(s["text"]) > 420 else ""))
    return f"""
    <div class="md-src" style="--c:{color}">
      <div class="md-src-head">
        <span class="md-num">[{s['num']}]</span>
        <span class="md-badge">{s['type']}</span>
        <span>{html.escape(s.get('source', ''))}{f" · page {s['page']}" if s['page'] else ""}</span>
        <span class="md-meta">{s['via']} · {s['score']:.3f}</span>
      </div>
      <div class="md-bar"><div style="width:{width}%"></div></div>
      <div class="md-txt">{text}</div>
    </div>"""


def show_details(msg: dict) -> None:
    """Sous une réponse : outils choisis, durée, images d'origine et sources."""
    chips = ""
    if msg.get("doc"):
        chips += f'<span class="md-chip doc">📄 {html.escape(msg["doc"])}</span>'
    chips += "".join(f'<span class="md-chip">🔧 {html.escape(t)}</span>' for t in msg.get("tools", []))
    if msg.get("seconds") is not None:
        chips += f'<span class="md-chip time">⏱ {msg["seconds"]:.0f} s</span>'
    if chips:
        st.markdown(f'<div class="md-chips">{chips}</div>', unsafe_allow_html=True)

    images, sources, trace = msg.get("images", []), msg.get("sources", []), msg.get("trace", [])
    if not images and not sources and not trace:
        return
    tabs = st.tabs([f"🧠 Raisonnement ({len(trace)} étapes)", f"🖼️ Images ({len(images)})",
                    f"📚 Sources ({len(sources)})"])
    with tabs[0]:
        steps = "".join(
            f'<div class="md-step"><div class="md-step-n">{i}</div><div>'
            f'<div class="md-step-tool">{html.escape(t["outil"])}</div>'
            f'<div class="md-step-arg">{html.escape(str(t["argument"]))}</div>'
            f'<div class="md-step-res">→ {html.escape(t["resultat"])}</div></div></div>'
            for i, t in enumerate(trace, start=1))
        steps += (f'<div class="md-step"><div class="md-step-n">{len(trace) + 1}</div><div>'
                  '<div class="md-step-tool">réponse</div><div class="md-step-arg">'
                  'Qwen rédige la réponse à partir des extraits, dans la langue de la question</div></div></div>')
        st.markdown(steps, unsafe_allow_html=True)
    with tabs[1]:
        if images:
            cols = st.columns(2)  # 2 images par ligne au maximum
            for i, path in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(client.url_fichier(path), caption=path.split("/")[-1], width="stretch")
        else:
            st.caption("Aucune image utilisée pour cette réponse.")
    with tabs[2]:
        st.markdown("".join(source_card(s) for s in sources), unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🔧" if msg["role"] == "user" else "🛠️"):
        st.markdown(msg["content"])
        if msg.get("photo"):
            st.image(client.url_fichier(msg["photo"]), caption="Photo jointe", width=180)
        if msg["role"] == "assistant":
            show_details(msg)

# Suggestions quand la conversation est vide
pending = st.session_state.pop("pending", None)
if not st.session_state.messages and not pending:
    st.markdown('<div class="md-empty">Essayez une de ces questions, ou écrivez la vôtre en bas.</div>',
                unsafe_allow_html=True)
    cols = st.columns(3)
    for col, suggestion in zip(cols, SUGGESTIONS):
        if col.button(suggestion, width="stretch"):
            st.session_state.pending = suggestion
            st.rerun()

# Zone de question : le + permet de joindre un PDF (indexé) ou une image (recherche par photo)
entry = st.chat_input("Votre question, ou + pour joindre un PDF ou une image", accept_file=True,
                      file_type=[".pdf", ".png", ".jpg", ".jpeg"])
question, photo, forced_doc = pending, None, None
if entry:
    question = entry.text.strip() or None
    pdf_file = next((f for f in entry.files if f.name.lower().endswith(".pdf")), None)
    image_file = next((f for f in entry.files if not f.name.lower().endswith(".pdf")), None)

    if pdf_file:  # 1. PDF joint : l'API l'indexe
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(f"📄 **{pdf_file.name}**")
        st.session_state.messages.append({"role": "user", "content": f"📄 **{pdf_file.name}**"})
        with st.chat_message("assistant", avatar="🛠️"):
            try:
                with st.status("Indexation du document par l'API...", expanded=True) as status:
                    st.write("Extraction du texte, des tableaux et des images")
                    st.write(f"Descriptions des images par {VISION_MODEL} (1 à 3 min par image sur CPU)")
                    st.write("Vecteurs BGE-M3 et CLIP")
                    bilan = client.indexer(pdf_file.name, pdf_file.getvalue())
                    status.update(label="Document indexé", state="complete")
                done = (f"Document **{pdf_file.name}** indexé en {bilan['secondes']} s : "
                        f"{bilan.get('texte', 0)} passages, {bilan.get('tableau', 0)} tableaux, "
                        f"{bilan.get('image', 0)} images. Vous pouvez poser vos questions.")
                st.session_state.next_doc = pdf_file.name
                forced_doc = pdf_file.name
            except APIError as e:
                done = f"⚠️ L'indexation a échoué. {e}"
            st.markdown(done)
        st.session_state.messages.append({"role": "assistant", "content": done})
        get_stats.clear()

    if image_file:  # 2. Image jointe : elle accompagne la question (recherche par photo)
        photo = (image_file.name, image_file.getvalue())
        question = question or "Quelle figure du document ressemble à cette photo ?"

    if pdf_file and not question:
        st.rerun()  # document seulement : on met à jour la barre latérale

if question:
    with st.chat_message("user", avatar="🧑‍🔧"):
        st.markdown(question)
        if photo:
            st.image(photo[1], caption="Photo jointe", width=180)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)

    with st.chat_message("assistant", avatar="🛠️"):
        choice = forced_doc or st.session_state.get("doc_choice")
        try:
            with st.spinner("L'agent choisit ses outils et cherche dans le document..."):
                result = client.poser_question(question, history,
                                               None if choice in (None, ALL_DOCS) else choice, photo)
            user_msg["photo"] = result.get("photo")  # la photo est gardée par l'API
            msg = {"role": "assistant", "content": result["answer"], "sources": result["sources"],
                   "images": result["images"], "tools": result["tools"], "seconds": result["secondes"],
                   "doc": choice or ALL_DOCS, "trace": result.get("trace", [])}
        except APIError as e:
            msg = {"role": "assistant", "content": f"⚠️ {e}"}
        st.markdown(msg["content"])
        show_details(msg)
    st.session_state.messages.append(msg)
