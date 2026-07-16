# -*- coding: utf-8 -*-
"""
pipeline_core.py
Pipeline clustering abstrak (faithful ke tesis):
Preprocessing (EN) -> TF-IDF -> LDA (Gensim) -> Early Fusion ->
SVD+Normalizer -> KMeans (evaluasi K & final) -> Top-10 terms + WordCloud.

Versi ini di-refactor dari pipeline CLI agar bisa dipanggil oleh aplikasi web
(Streamlit): bekerja in-memory, mengembalikan hasil sebagai dict, dan
opsional menulis artefak ke folder. Logika & parameter dipertahankan sama.
"""

from __future__ import annotations
import io
import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from wordcloud import WordCloud

# ============================================================
# KONFIGURASI (sama dengan pipeline asli)
# ============================================================
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8

K_LDA = 10
LDA_NO_BELOW = 5
LDA_NO_ABOVE = 0.5
LDA_PASSES = 12
LDA_ITERATIONS = 200
LDA_CHUNKSIZE = 200

ALPHA_LDA = 1.0
SVD_COMPONENTS = 100
K_MIN = 2
K_MAX = 20

EXTRA_STOPWORDS = {
    "research", "study", "paper", "article", "method", "methods",
    "result", "results", "data", "dataset", "using", "use", "used",
    "based", "approach", "proposed", "analysis", "system", "information",
    "technology", "model", "models",
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "atau",
    "dalam", "ini", "itu", "tidak", "ya", "jadi", "lebih", "dapat",
    "akan", "juga", "sebagai", "tersebut",
}
STOPWORDS_FINAL = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS
STOPWORDS_WC = set(ENGLISH_STOP_WORDS) | {
    "yang", "dan", "di", "ke", "dari", "pada",
    "untuk", "dengan", "atau", "dalam", "ini", "itu",
}
STOPWORDS_TOP10 = set(ENGLISH_STOP_WORDS) | {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "atau", "dalam",
    "ini", "itu", "ada", "tidak", "ya", "jadi", "lebih", "dapat", "akan", "juga",
    "oleh", "sebagai", "bahwa", "tersebut", "dilakukan", "melakukan", "baik",
    "adalah", "yaitu", "salah", "satu", "secara", "terhadap", "berdasarkan",
    "hasil", "penelitian", "studi", "paper", "artikel", "metode", "menggunakan",
    "data", "informasi", "sistem", "aplikasi", "proses", "pengguna", "uji", "nilai", "hal",
}

RE_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_ALPHA_ONLY = re.compile(r"^[a-z]+$")


def _noop(*_a, **_k):
    pass


def clean_text_en(text: str) -> str:
    t = str(text).lower()
    t = RE_NON_ALPHA.sub(" ", t)
    t = RE_MULTI_SPACE.sub(" ", t).strip()
    return t


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(df: pd.DataFrame, abstract_col: str):
    texts_raw = df[abstract_col].fillna("").astype(str).tolist()
    stemmer = PorterStemmer()
    cache: dict[str, str] = {}

    def stem(w):
        s = cache.get(w)
        if s is None:
            s = stemmer.stem(w)
            cache[w] = s
        return s

    final_rows, prev = [], []
    for i, raw in enumerate(texts_raw, start=1):
        cleaned = clean_text_en(raw)
        tokens = cleaned.split() if cleaned else []
        tokens_sw = [w for w in tokens if len(w) > 2 and w not in STOPWORDS_FINAL]
        tokens_stem = [stem(w) for w in tokens_sw]
        final_rows.append({
            "No": i,
            "ABSTRAK_ORIGINAL": raw,
            "ABSTRAK_FINAL_CLEAN": " ".join(tokens_stem),
        })
        if tokens_stem:
            prev.append({
                "No": i,
                "Case_Folding": cleaned,
                "Tokenization": ", ".join(tokens[:40]),
                "Stopword_Removal": ", ".join(tokens_sw[:40]),
                "Stemming": ", ".join(tokens_stem[:40]),
            })
    return pd.DataFrame(final_rows), pd.DataFrame(prev)


# ============================================================
# TF-IDF
# ============================================================
def build_tfidf(df_clean: pd.DataFrame, min_df: int):
    texts = df_clean["ABSTRAK_FINAL_CLEAN"].fillna("").astype(str).reset_index(drop=True)
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
        min_df=min_df,
        max_df=TFIDF_MAX_DF,
    )
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    tokens_lda = [t.split() for t in texts.tolist()]
    return X, vocab, df_clean.reset_index(drop=True), tokens_lda


# ============================================================
# LDA (Gensim) — dengan penyesuaian otomatis untuk data kecil
# ============================================================
def run_lda(tokens_lda, n_docs: int):
    cleaned, valid = [], []
    for i, toks in enumerate(tokens_lda):
        c = [t for t in toks if len(t) >= 3 and RE_ALPHA_ONLY.match(t)]
        if c:
            cleaned.append(c)
            valid.append(i)

    # Penyesuaian otomatis bila dataset kecil
    no_below = LDA_NO_BELOW if n_docs >= 100 else 1
    k_lda = K_LDA if n_docs >= 20 else max(2, min(K_LDA, n_docs // 2))

    dictionary = Dictionary(cleaned)
    dictionary.filter_extremes(no_below=no_below, no_above=LDA_NO_ABOVE)
    if len(dictionary) == 0:  # fallback ekstrem untuk data sangat kecil
        dictionary = Dictionary(cleaned)
    corpus = [dictionary.doc2bow(doc) for doc in cleaned]

    lda = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=k_lda,
        random_state=RANDOM_STATE, passes=LDA_PASSES,
        iterations=LDA_ITERATIONS, chunksize=LDA_CHUNKSIZE,
        alpha="symmetric", eta=None,
    )

    theta = np.zeros((len(tokens_lda), k_lda), dtype=np.float32)
    for out_i, in_i in enumerate(valid):
        for tid, p in lda.get_document_topics(corpus[out_i], minimum_probability=0):
            theta[in_i, tid] = p

    df_lda = pd.DataFrame(theta, columns=[f"LDA_TOPIC_{i+1:02d}" for i in range(k_lda)])
    top_words = []
    for t in range(k_lda):
        words = lda.show_topic(t, topn=10)
        top_words.append({"topic": t + 1, "terms": [w for w, _ in words]})
    return df_lda, top_words, len(dictionary), k_lda


# ============================================================
# FUSION + EVALUASI K + KMEANS FINAL
# ============================================================
def fuse_and_cluster(X_tfidf, df_lda, df_tfidf, k_max: int):
    X_lda = csr_matrix(df_lda.values * ALPHA_LDA)
    combined = hstack([X_tfidf, X_lda], format="csr")
    n_docs = combined.shape[0]

    n_comp = min(SVD_COMPONENTS, combined.shape[1] - 1, n_docs - 1)
    svd_pipe = make_pipeline(
        TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE),
        Normalizer(copy=False),
    )
    X_eval = svd_pipe.fit_transform(combined)

    max_k = min(k_max, n_docs - 1)
    k_range = list(range(K_MIN, max_k + 1))
    sil_scores, dbi_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    max_iter=300, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_eval)
        sil_scores.append(float(silhouette_score(X_eval, labels)))
        dbi_scores.append(float(davies_bouldin_score(X_eval, labels)))

    df_eval = pd.DataFrame({
        "K": k_range, "Silhouette_Score": sil_scores,
        "Davies_Bouldin_Index": dbi_scores,
    })

    # K optimal: 3 Silhouette tertinggi -> pilih DBI terendah di antaranya
    top3_idx = np.argsort(sil_scores)[::-1][:3]
    top3_k = [k_range[i] for i in top3_idx]
    top3_dbi = [dbi_scores[i] for i in top3_idx]
    best_k = int(top3_k[int(np.argmin(top3_dbi))])

    km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                      max_iter=300, tol=1e-4, random_state=RANDOM_STATE)
    labels_final = km_final.fit_predict(X_eval)
    sil_final = float(silhouette_score(X_eval, labels_final))
    dbi_final = float(davies_bouldin_score(X_eval, labels_final))

    df_result = df_tfidf.copy()
    df_result["CLUSTER"] = labels_final + 1
    df_dist = df_result["CLUSTER"].value_counts().sort_index().reset_index()
    df_dist.columns = ["CLUSTER", "NUM_DOCUMENTS"]

    info = {
        "best_k_final": best_k,
        "silhouette_final": sil_final,
        "dbi_final": dbi_final,
        "iterations": int(km_final.n_iter_),
    }
    return df_result, df_eval, df_dist, info


# ============================================================
# TOP TERMS + WORDCLOUD (gambar in-memory)
# ============================================================
def top_terms(df_result, X_tfidf, vocab):
    terms = np.array(vocab)
    rows = []
    for c in sorted(df_result["CLUSTER"].unique()):
        idx = np.where(df_result["CLUSTER"].values == c)[0]
        mean_tfidf = np.asarray(X_tfidf[idx].mean(axis=0)).ravel()
        cand = mean_tfidf.argsort()[::-1][:200]
        top = []
        for i in cand:
            term = terms[i]
            if mean_tfidf[i] <= 0 or term in STOPWORDS_TOP10 or len(term) <= 2:
                continue
            if not term.replace(" ", "").isalpha():
                continue
            top.append((term, float(mean_tfidf[i])))
            if len(top) == 10:
                break
        for rank, (t, w) in enumerate(top, 1):
            rows.append({"CLUSTER": int(c), "RANK": rank, "TERM": t, "MEAN_TFIDF": round(w, 4)})
    return pd.DataFrame(rows)


def wordcloud_image(text: str):
    """Kembalikan PNG bytes dari WordCloud satu klaster (atau None bila kurang data)."""
    if len(text.split()) < 10:
        return None
    wc = WordCloud(width=1200, height=700, background_color="white",
                   stopwords=STOPWORDS_WC, collocations=False,
                   min_font_size=8, max_words=100).generate(text)
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# ORCHESTRATOR untuk aplikasi web
# ============================================================
def run_pipeline(df: pd.DataFrame, abstract_col: str, k_max: int = K_MAX,
                 progress=None) -> dict:
    """Jalankan seluruh pipeline. `progress(frac, label)` opsional untuk UI."""
    progress = progress or _noop
    t0 = time.time()

    progress(0.10, "Prapemrosesan teks...")
    df_clean, df_prev = preprocess(df, abstract_col)
    n_docs = len(df_clean)
    min_df = TFIDF_MIN_DF if n_docs >= 10 else 1

    progress(0.30, "Ekstraksi fitur TF-IDF...")
    X_tfidf, vocab, df_tfidf, tokens_lda = build_tfidf(df_clean, min_df)

    progress(0.50, "Pemodelan topik LDA...")
    df_lda, lda_top, vocab_lda, k_lda = run_lda(tokens_lda, n_docs)

    progress(0.70, "Penggabungan fitur & evaluasi K (clustering)...")
    df_result, df_eval, df_dist, info = fuse_and_cluster(X_tfidf, df_lda, df_tfidf, k_max)

    progress(0.88, "Istilah dominan & WordCloud...")
    df_top10 = top_terms(df_result, X_tfidf, vocab)
    wc_images = {}
    for cid in sorted(df_result["CLUSTER"].unique()):
        blob = " ".join(df_result.loc[df_result["CLUSTER"] == cid, "ABSTRAK_FINAL_CLEAN"]).strip()
        img = wordcloud_image(blob)
        if img is not None:
            wc_images[int(cid)] = img

    progress(1.0, "Selesai.")
    return {
        "df_clean": df_clean,
        "df_preview": df_prev,
        "df_result": df_result,
        "df_eval": df_eval,
        "df_dist": df_dist,
        "df_top10": df_top10,
        "lda_top": lda_top,
        "wc_images": wc_images,
        "vocab": list(vocab),
        "tfidf_shape": list(X_tfidf.shape),
        "combined_shape": [X_tfidf.shape[0], X_tfidf.shape[1] + k_lda],
        "k_lda": k_lda,
        "vocab_lda": vocab_lda,
        "elapsed": round(time.time() - t0, 1),
        **info,
    }


# ============================================================
# CLI (kompatibel dengan versi lama)
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--col", default="ABSTRAK")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input).dropna(subset=[args.col]).reset_index(drop=True)
    res = run_pipeline(df, args.col, progress=lambda f, l: print(f"[{int(f*100):3d}%] {l}"))
    res["df_result"][["No", "ABSTRAK_ORIGINAL", "CLUSTER"]].to_csv(
        outdir / "15_Final_Clustering_Results.csv", index=False)
    res["df_eval"].to_csv(outdir / "14_K_Evaluation.csv", index=False)
    res["df_top10"].to_csv(outdir / "21_Top10_Terms_Per_Cluster.csv", index=False)
    print(json.dumps({k: res[k] for k in
                      ("best_k_final", "silhouette_final", "dbi_final", "iterations")},
                     indent=2))
    print(f"[OK] artefak di {outdir.resolve()}")
