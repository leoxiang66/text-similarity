from typing import  List
from sklearn.feature_extraction.text import TfidfVectorizer



from ._comp import cossim_topN, get_matches_df

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

    corpora = corpus_a + corpus_b
    vectorizer = TfidfVectorizer()
    temp = vectorizer.fit_transform(corpora)

    m1 = temp[:len(corpus_a)] # N,D
    m2 = temp[len(corpus_a):] # M,D


    matches = cossim_topN(m1,m2.transpose(),top_k_similarity, similarity_lower_bound)
    return matches

