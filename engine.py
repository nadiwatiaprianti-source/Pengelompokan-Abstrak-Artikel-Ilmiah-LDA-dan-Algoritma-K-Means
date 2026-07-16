# -*- coding: utf-8 -*-
"""
engine.py — Mesin "cari artikel sejenis" untuk desain Paper 3.
Melatih pipeline pada korpus (TF-IDF + LDA Gensim + Early Fusion + SVD/L2 + K-Means),
lalu menempatkan SATU abstrak baru ke klaster terdekat dan mengembalikan
artikel-artikel paling mirip dalam klaster tersebut.

Konsisten dengan pipeline_core (parameter & preprocessing sama).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

import pipeline_core as pc  # pakai config & pembersih yang sama

_STEM = PorterStemmer()


def _tokens(text: str):
    """Preprocessing satu teks -> daftar token (sama dengan pipeline_core)."""
    cleaned = pc.clean_text_en(text)
    toks = [w for w in cleaned.split() if len(w) > 2 and w not in pc.STOPWORDS_FINAL]
    return [_STEM.stem(w) for w in toks]


def _lda_tokens(stem_tokens):
    return [t for t in stem_tokens if len(t) >= 3 and pc.RE_ALPHA_ONLY.match(t)]


def fit_corpus(df: pd.DataFrame, col_abstrak: str, k_max: int = pc.K_MAX):
    """Latih pipeline pada korpus, simpan objek terlatih untuk pencarian."""
    raw = df[col_abstrak].fillna("").astype(str).tolist()
    stem_tokens = [_tokens(t) for t in raw]
    cleaned = [" ".join(s) for s in stem_tokens]
    n = len(cleaned)

    # TF-IDF
    min_df = pc.TFIDF_MIN_DF if n >= 10 else 1
    vec = TfidfVectorizer(max_features=pc.TFIDF_MAX_FEATURES, ngram_range=pc.TFIDF_NGRAM,
                          min_df=min_df, max_df=pc.TFIDF_MAX_DF)
    X_tfidf = vec.fit_transform(cleaned)
    vocab = np.array(vec.get_feature_names_out())

    # LDA (Gensim)
    lda_docs = [_lda_tokens(s) for s in stem_tokens]
    valid = [i for i, d in enumerate(lda_docs) if d]
    no_below = pc.LDA_NO_BELOW if n >= 100 else 1
    k_lda = pc.K_LDA if n >= 20 else max(2, min(pc.K_LDA, n // 2))
    dictionary = Dictionary([lda_docs[i] for i in valid])
    dictionary.filter_extremes(no_below=no_below, no_above=pc.LDA_NO_ABOVE)
    if len(dictionary) == 0:
        dictionary = Dictionary([lda_docs[i] for i in valid])
    corpus_bow = [dictionary.doc2bow(lda_docs[i]) for i in valid]
    lda = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=k_lda,
                   random_state=pc.RANDOM_STATE, passes=pc.LDA_PASSES,
                   iterations=pc.LDA_ITERATIONS, chunksize=pc.LDA_CHUNKSIZE, alpha="symmetric")
    theta = np.zeros((n, k_lda), dtype=np.float32)
    for out_i, in_i in enumerate(valid):
        for tid, p in lda.get_document_topics(corpus_bow[out_i], minimum_probability=0):
            theta[in_i, tid] = p

    # Fusion + SVD + L2
    combined = hstack([X_tfidf, csr_matrix(theta * pc.ALPHA_LDA)], format="csr")
    n_comp = min(pc.SVD_COMPONENTS, combined.shape[1] - 1, n - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=pc.RANDOM_STATE)
    X_red = svd.fit_transform(combined)
    norm = Normalizer(norm="l2")
    X_eval = norm.fit_transform(X_red)

    # Pilih K (3 Silhouette tertinggi -> DBI terendah)
    max_k = min(k_max, n - 1)
    k_range = list(range(pc.K_MIN, max_k + 1))
    sils, dbis = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300,
                    random_state=pc.RANDOM_STATE)
        lab = km.fit_predict(X_eval)
        sils.append(silhouette_score(X_eval, lab))
        dbis.append(davies_bouldin_score(X_eval, lab))
    top3 = np.argsort(sils)[::-1][:3]
    best_k = int([k_range[i] for i in top3][int(np.argmin([dbis[i] for i in top3]))])
    kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, max_iter=300,
                    tol=1e-4, random_state=pc.RANDOM_STATE)
    labels = kmeans.fit_predict(X_eval)

    # Istilah dominan & label tema per klaster
    X_arr = X_tfidf.toarray()
    themes, top_terms = {}, {}
    for c in range(best_k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            themes[c] = "-"; top_terms[c] = []; continue
        mean_w = X_arr[idx].mean(axis=0)
        order = mean_w.argsort()[::-1]
        terms = [vocab[i] for i in order if mean_w[i] > 0 and vocab[i] not in pc.STOPWORDS_TOP10
                 and vocab[i].replace(" ", "").isalpha()][:8]
        top_terms[c] = terms
        themes[c] = " · ".join(terms[:3]) if terms else "-"

    return {
        "vec": vec, "dictionary": dictionary, "lda": lda, "svd": svd, "norm": norm,
        "kmeans": kmeans, "X_eval": X_eval, "labels": labels, "k_lda": k_lda,
        "best_k": best_k, "themes": themes, "top_terms": top_terms,
        "df": df.reset_index(drop=True), "col_abstrak": col_abstrak,
        "silhouette": float(silhouette_score(X_eval, labels)),
        "dbi": float(davies_bouldin_score(X_eval, labels)),
    }


def assign_and_search(model, text: str, top_n: int | None = None):
    """Tempatkan abstrak baru ke klaster terdekat & rangking HANYA anggota
    klaster tersebut berdasarkan kemiripan (dari yang paling relevan ke yang
    paling tidak relevan). top_n=None -> kembalikan seluruh anggota klaster."""
    stem = _tokens(text)
    if not stem:
        return None
    cleaned = " ".join(stem)

    x_tfidf = model["vec"].transform([cleaned])
    theta = np.zeros((1, model["k_lda"]), dtype=np.float32)
    bow = model["dictionary"].doc2bow(_lda_tokens(stem))
    for tid, p in model["lda"].get_document_topics(bow, minimum_probability=0):
        theta[0, tid] = p
    combined = hstack([x_tfidf, csr_matrix(theta * pc.ALPHA_LDA)], format="csr")
    x_red = model["svd"].transform(combined)
    x_eval = model["norm"].transform(x_red)

    cluster = int(model["kmeans"].predict(x_eval)[0])
    members = np.where(model["labels"] == cluster)[0]

    # Kemiripan HANYA terhadap dokumen dalam klaster yang sama (bukan seluruh korpus)
    # cosine == dot product (vektor sudah L2-normal)
    sims_members = model["X_eval"][members] @ x_eval[0]
    order = np.argsort(sims_members)[::-1]
    if top_n is not None:
        order = order[:top_n]
    hits = [(int(members[i]), float(sims_members[i])) for i in order]

    return {
        "cluster": cluster + 1,
        "theme": model["themes"].get(cluster, "-"),
        "n_members": int(len(members)),
        "hits": hits,  # (row_index_in_df, similarity), terurut dari paling relevan, hanya anggota klaster
    }
