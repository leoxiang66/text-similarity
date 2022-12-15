from typing import  List
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import pandas as pd


def tfidf_sim_pipeline(
        corpus_a: List[str],
        corpus_b: List[str],
        top_k_similarity: int = 3,
        similarity_lower_bound: float = 0.
):
    '''
    pipe to compute tfidf similarity of given two corpora

    :param similarity_lower_bound:
    :param top_k_similarity:
    :param corpus_a: list[str], N samples
    :param corpus_b: list[str], M samples
    :return:
    '''
    assert top_k_similarity < min(len(corpus_a),len(corpus_b))

    corpora = corpus_a + corpus_b
    vectorizer = TfidfVectorizer()
    temp = vectorizer.fit_transform(corpora)

    m1 = temp[:len(corpus_a)] # N,D
    m2 = temp[len(corpus_a):] # M,D


    sim = awesome_cossim_topn(m1,m2.transpose(),top_k_similarity, similarity_lower_bound) # NxM
    nonzero_x , nonzero_y = sim.nonzero()
    data = []
    columns = ["Text"]
    for j in range(top_k_similarity):
        columns.append(f"Top-{j+1} Similarity")

    for i in range(len(corpus_a)):
        tmp = [corpus_a[i]]
        for j in range(top_k_similarity):
            tmp.append(corpus_b[nonzero_y[i*top_k_similarity+j]])
        data.append(tmp)


    return pd.DataFrame(data, columns=columns)

