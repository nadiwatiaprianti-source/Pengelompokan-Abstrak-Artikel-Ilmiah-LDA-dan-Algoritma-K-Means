# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os, hashlib, html, pickle
import numpy as np
import pandas as pd
import streamlit as st
import engine

st.set_page_config(
    page_title="Scientific Article Abstract Clustering",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIMARY    = "#1a2e4a"
ACCENT     = "#2563eb"
ACCENT_LT  = "#dbeafe"
ACCENT_MID = "#bfdbfe"
BG         = "#f0f4f8"
BORDER     = "#dde3ed"
TEXT       = "#0f172a"
MUTED      = "#64748b"
WHITE      = "#ffffff"

CSV_PATH  = "contoh_korpus.csv"
MODEL_CACHE_DIR = ".model_cache"
# Metadata lengkap (Jurnal, Judul Paper, Keyword, Volume, Tahun) untuk TAMPILAN saja.
# Baris di file ini SEJAJAR posisi 1:1 dengan CSV_PATH (sama-sama 1800 baris, urutan
# paper sama), jadi digabung berdasarkan posisi -- bukan pencocokan teks abstrak,
# karena abstrak di sini belum semuanya diterjemahkan sedangkan CSV_PATH sudah.
# Kolom ABSTRAK di CSV_PATH TIDAK diubah/diganti, sehingga hasil clustering tetap
# identik dengan paper 2.
META_PATH = "prosesing-20082025 - Sheet18 (1).csv"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT};
}}
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1180px !important;
}}
[data-testid="collapsedControl"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ display: none !important; }}

/* ── input card ── */
.input-card {{
    background: {WHITE};
    border: 1px solid {ACCENT};
    border-radius: 14px;
    padding: 2.8rem 2.8rem 2rem 2.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 8px 28px rgba(0,0,0,0.05);
    margin-bottom: 1.4rem;
}}
.card-title {{
    text-align: center;
    font-weight: 800;
    font-size: 1.55rem;
    line-height: 1.3;
    margin: 0;
    letter-spacing: -0.03em;
}}
.field-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: {PRIMARY};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 0.4rem;
}}
.field-hint {{
    font-size: 0.78rem;
    color: {MUTED};
    margin: 0 0 0.5rem 0;
    line-height: 1.5;
}}

/* ── textarea ── */
.stTextArea > div > div > textarea {{
    border: 1.5px solid {ACCENT} !important;
    border-radius: 8px !important;
    background: #f8fafc !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    color: {TEXT} !important;
    line-height: 1.65 !important;
    resize: vertical !important;
}}
.stTextArea > div > div > textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    outline: none !important;
    background: {WHITE} !important;
}}
/* Textarea yang di-disable (mis. "Abstrak yang Dianalisis") tetap harus terbaca gelap */
.stTextArea textarea:disabled,
.stTextArea textarea[disabled] {{
    color: {TEXT} !important;
    -webkit-text-fill-color: {TEXT} !important;
    opacity: 1 !important;
    background: #f8fafc !important;
    -webkit-opacity: 1 !important;
}}

/* ── panel biru bersatu (header + isi jadi satu kotak) ── */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1.5px solid {ACCENT} !important;
    border-radius: 14px !important;
    background: {WHITE} !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.06) !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border-radius: 14px !important;
}}
.panel-hd {{
    color: {PRIMARY};
    font-weight: 700;
    font-size: 0.92rem;
    padding: 0 0 0.7rem 0;
    margin-bottom: 1rem;
    border-bottom: 1.5px solid {ACCENT};
    display: flex;
    align-items: center;
    gap: 0.55rem;
}}
.panel-hd::before {{
    content: "";
    display: inline-block;
    width: 4px;
    height: 17px;
    background: {ACCENT};
    border-radius: 2px;
}}
.panel-sub {{
    color: {MUTED};
    font-size: 0.78rem;
    margin: -0.6rem 0 1rem 0;
}}

/* dataframe & expander ikut dibingkai biru tipis supaya senada */
div[data-testid="stDataFrame"] {{
    border: 1.5px solid {ACCENT} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
div[data-testid="stExpander"] {{
    border: 1.5px solid {ACCENT} !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden;
}}

/* ── primary button ── */
.stButton > button {{
    background: {ACCENT} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1.6rem !important;
    letter-spacing: 0.015em !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.3) !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background: #1d4ed8 !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.38) !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{
    transform: translateY(0px) !important;
}}

/* ── result cards ── */
.result-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.8rem 2rem 1.6rem 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 8px 28px rgba(0,0,0,0.05);
}}
.result-card-hd {{
    color: {PRIMARY};
    font-weight: 700;
    font-size: 0.95rem;
    margin: 0 0 1.2rem 0;
    padding-bottom: 0.85rem;
    border-bottom: 1.5px solid {BORDER};
    line-height: 1.4;
    letter-spacing: -0.01em;
}}
.paper-item {{ margin-top: 0.4rem; }}
.paper-num {{
    color: {ACCENT};
    font-size: 0.68rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}}
.paper-num::before {{
    content: "";
    display: inline-block;
    width: 14px;
    height: 2px;
    background: {ACCENT};
    border-radius: 1px;
}}
.f-label {{
    font-size: 0.67rem;
    color: {MUTED};
    font-weight: 700;
    margin-bottom: 0.22rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
.f-box {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 0.48rem 0.72rem;
    font-size: 0.84rem;
    color: {TEXT};
    background: #f8fafc;
    word-break: break-word;
    margin-bottom: 0.65rem;
    line-height: 1.5;
    min-height: 2.1rem;
}}
.f-box-lg {{ min-height: 3.8rem; }}
hr.sep {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 0.9rem 0;
}}

/* ── upload screen ── */
.upload-card {{
    background: {WHITE};
    border: 1px solid {ACCENT};
    border-radius: 14px;
    padding: 3rem 3rem 2.5rem 3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 8px 28px rgba(0,0,0,0.05);
    text-align: center;
}}
.upload-icon {{
    font-size: 2.8rem;
    margin-bottom: 1rem;
    display: block;
}}
.upload-title {{
    color: {PRIMARY};
    font-weight: 800;
    font-size: 1.3rem;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}}
.upload-desc {{
    color: {MUTED};
    font-size: 0.85rem;
    line-height: 1.6;
    margin: 0 0 1.8rem 0;
}}
.upload-note {{
    color: {MUTED};
    font-size: 0.75rem;
    margin-top: 1rem;
    line-height: 1.5;
}}

/* ── expander ── */
.streamlit-expanderHeader {{
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
    color: {MUTED} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Corpus loading ─────────────────────────────────────────────────────────────
@st.cache_data
def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    abs_col = next((c for c in df.columns if c.lower() in ("abstrak", "abstract", "text")), None)
    if abs_col:
        df = df.dropna(subset=[abs_col])
    return df.reset_index(drop=True)


@st.cache_data
def _load_metadata(path: str, n_rows: int):
    """Metadata (Jurnal, Judul Paper, Keyword, Volume, Tahun) untuk ditempel ke korpus
    berdasarkan posisi baris. None kalau file tidak ada atau jumlah barisnya tidak sejajar."""
    if not os.path.exists(path):
        return None
    meta = pd.read_csv(path).reset_index(drop=True)
    if len(meta) != n_rows:
        return None
    return meta


def _with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    meta = _load_metadata(META_PATH, len(df))
    if meta is None:
        return df
    cols = [c for c in ("Jurnal", "Judul Paper", "Keyword", "Volume", "Tahun")
            if c in meta.columns and c not in df.columns]
    if not cols:
        return df
    return pd.concat([df, meta[cols].reset_index(drop=True)], axis=1)


if "corpus" not in st.session_state:
    if os.path.exists(CSV_PATH):
        st.session_state["corpus"] = _with_metadata(_load_csv(CSV_PATH))
    else:
        st.session_state["corpus"] = None

corpus = st.session_state["corpus"]

if "page" not in st.session_state:
    st.session_state["page"] = "input"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SCREEN — Upload Data (tampil jika belum ada data)                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if corpus is None:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("""
<div class="upload-card">
  <span class="upload-icon">📂</span>
  <h2 class="upload-title">No Article Data Yet</h2>
  <p class="upload-desc">
    Upload a CSV file containing your scientific article data.<br>
    Make sure the CSV has these columns: <strong>Title</strong>, <strong>ABSTRACT</strong>, <strong>Keyword</strong>.
  </p>
</div>
""", unsafe_allow_html=True)

        up = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
        if up:
            try:
                nd = pd.read_csv(up)
                nd.to_csv(CSV_PATH, index=False)
                st.session_state["corpus"] = _with_metadata(nd)
                _load_csv.clear()
                st.success(f"✅ {len(nd)} articles successfully loaded and saved.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        st.markdown("""
<p class="upload-note">
  Data only needs to be uploaded once. The system will save it automatically<br>
  so you won't need to upload it again every time you open the app.
</p>
""", unsafe_allow_html=True)

    st.stop()


# ── Column detection & model ───────────────────────────────────────────────────
all_cols = list(corpus.columns)

def _pick(*names: str):
    name_set = {n.lower() for n in names}
    for c in all_cols:
        if c.lower() in name_set:
            return c
    return None

col_abs   = _pick("abstrak", "abstract", "text") or all_cols[0]
col_jud   = _pick("judul", "title", "judul paper")
col_jur   = _pick("jurnal", "journal")
col_kw    = _pick("keyword", "keywords")
col_vol   = _pick("volume", "vol")
col_tahun = _pick("tahun", "year")

def _fmt_val(v) -> str:
    """String rapi utk nilai metadata; buang '.0' akibat kolom numerik ber-NaN di CSV."""
    if pd.isna(v):
        return "-"
    s = str(v)
    return s[:-2] if s.endswith(".0") else s

def get_model(df: pd.DataFrame, col: str):
    key = hashlib.md5("||".join(df[col].fillna("").astype(str)).encode()).hexdigest()
    if st.session_state.get("_mkey") != key:
        cache_path = os.path.join(MODEL_CACHE_DIR, f"{key}.pkl")
        with st.spinner("Preparing clustering model..."):
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = engine.fit_corpus(df, col)
                os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(model, f)
            st.session_state["_model"] = model
            st.session_state["_mkey"]  = key
    return st.session_state["_model"]

model = get_model(corpus, col_abs)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE — Input                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if st.session_state["page"] == "input":

    _, center, _ = st.columns([1, 2.2, 1])
    with center:
        # Title card
        st.markdown(f"""
<div class="input-card">
  <h2 class="card-title" style="color:{ACCENT};">Scientific Article<br>Abstract Clustering</h2>
</div>
""", unsafe_allow_html=True)

        # Input label
        st.markdown("""
<span class="field-label">Input Abstract</span>
""", unsafe_allow_html=True)

        text = st.text_area(
            "Input Abstract",
            height=260,
            label_visibility="collapsed",
            placeholder="Paste the scientific article abstract here...",
            key="q_abstrak",
        )
        go = st.button("Check Abstract Clustering →", use_container_width=True, key="btn_go")

    if go:
        if not text.strip():
            _, c, _ = st.columns([1, 2.2, 1])
            with c:
                st.warning("Please paste an abstract first.")
        else:
            with st.spinner("Analyzing abstract..."):
                res = engine.assign_and_search(model, text, top_n=None)
            if res is None:
                _, c, _ = st.columns([1, 2.2, 1])
                with c:
                    st.warning("Abstract is too short. Please use a longer text.")
            else:
                st.session_state["res"]        = res
                st.session_state["input_text"] = text
                st.session_state["page"]       = "result"
                st.rerun()



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE — Hasil Pengelompokan                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
elif st.session_state["page"] == "result":

    res       = st.session_state.get("res", {})
    cluster_n = res.get("cluster", "?")
    theme_str = res.get("theme", "-")
    n_members = res.get("n_members", 0)

    # ── Daftar artikel dalam klaster yang sama, terurut dari paling relevan ke paling tidak relevan ──
    hits = res.get("hits", [])  # hanya anggota klaster hasil assign_and_search, sudah terurut

    # Tidak ada satu baris pun yang dibuang -> total tetap sama dengan jumlah anggota klaster (n_members).
    # Kalau ada abstrak yang teksnya persis sama, kemunculan PERTAMA (paling relevan) tetap
    # di posisi atas; kemunculan berikutnya yang identik digeser ke bagian bawah daftar,
    # jadi bagian teratas/paling relevan dijamin tidak ada dua baris berisi teks yang sama persis.
    import re as _re
    def _norm_abs(t: str) -> str:
        # samakan spasi/enter berlebih (mis. sisa \r\n di CSV) supaya abstrak yang
        # isinya sama persis tetap terdeteksi sebagai duplikat walau beda whitespace.
        return _re.sub(r"\s+", " ", t.strip()).lower()

    seen_abs = set()
    primary, later_dupes = [], []
    for idx, sim in hits:
        row = model["df"].iloc[idx]
        abs_text = str(row[col_abs])
        key = _norm_abs(abs_text)
        item = (idx, sim, row, abs_text)
        if key in seen_abs:
            later_dupes.append(item)
        else:
            seen_abs.add(key)
            primary.append(item)
    ordered = primary + later_dupes

    # ── Banner judul ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a2e4a 0%,#2563eb 100%);
            border-radius:14px;padding:1.3rem 2rem;margin-bottom:1.6rem;
            text-align:center;
            box-shadow:0 4px 20px rgba(37,99,235,0.28)">
  <div style="color:white;font-size:1.25rem;font-weight:800;letter-spacing:-0.02em;line-height:1.3">
    Scientific Article Abstract Clustering Results
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Abstrak yang dianalisis ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<div class="panel-hd">Analyzed Abstract</div>', unsafe_allow_html=True)
        st.text_area(
            "abstract echo",
            value=st.session_state.get("input_text", ""),
            height=180,
            label_visibility="collapsed",
            disabled=True,
            key="echo_abstrak",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Daftar seluruh dokumen dalam klaster yang sama dengan abstrak, terurut dari yang
    # paling relevan ke yang paling tidak relevan terhadap abstrak yang dianalisis (No urut 1..N).
    rank_rows = []
    for no, (idx, sim, row, abs_text) in enumerate(ordered, 1):
        entry = {"No": no, "Cluster": int(model["labels"][idx]) + 1}
        if col_jur:
            entry["Journal"] = _fmt_val(row[col_jur])
        if col_jud:
            entry["Title"] = _fmt_val(row[col_jud])
        entry["Abstract"] = abs_text
        if col_kw:
            entry["Keyword"] = _fmt_val(row[col_kw])
        if col_vol:
            entry["Volume"] = _fmt_val(row[col_vol])
        if col_tahun:
            entry["Year"] = _fmt_val(row[col_tahun])
        rank_rows.append(entry)

    with st.container(border=True):
        st.markdown(f'<div class="panel-hd">{len(rank_rows)} Relevant Articles</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-sub">Sorted from most relevant to least relevant '
            'to the analyzed abstract (based on TF-IDF + LDA vector similarity).</div>',
            unsafe_allow_html=True,
        )

        df_rank = pd.DataFrame(rank_rows)
        col_cfg = {
            "No":      st.column_config.NumberColumn("No", width=40),
            "Cluster": st.column_config.NumberColumn("Cluster", width=55),
        }
        if col_jur:
            col_cfg["Journal"] = st.column_config.TextColumn("Journal", width="medium")
        if col_jud:
            col_cfg["Title"] = st.column_config.TextColumn("Title", width="large")
        col_cfg["Abstract"] = st.column_config.TextColumn("Abstract", width=900)
        if col_kw:
            col_cfg["Keyword"] = st.column_config.TextColumn("Keyword", width="medium")
        if col_vol:
            col_cfg["Volume"] = st.column_config.TextColumn("Volume", width=90)
        if col_tahun:
            col_cfg["Year"] = st.column_config.TextColumn("Year", width=80)

        st.dataframe(
            df_rank,
            use_container_width=True,
            hide_index=True,
            height=720,
            row_height=68,
            column_config=col_cfg,
        )
        st.caption("Click any Abstract cell to read its full content.")

    # ── Data Artikel (admin) ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Article Data", expanded=False):
        st.caption(f"Current data: **{len(corpus)} articles**")
        up = st.file_uploader("Replace article data (CSV: Title, ABSTRACT, Keyword)", type=["csv"])
        if up:
            try:
                nd = pd.read_csv(up)
                nd.to_csv(CSV_PATH, index=False)
                st.session_state["corpus"] = _with_metadata(nd)
                _load_csv.clear()
                st.success(f"✅ {len(nd)} articles successfully saved.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    # ── Tombol Kembali ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        if st.button("← Back", key="btn_back", use_container_width=True):
            st.session_state["page"] = "input"
            st.rerun()
