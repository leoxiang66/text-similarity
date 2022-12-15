import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import pandas as pd


def cossim_topN(A: csr_matrix, B:csr_matrix, topN:int = 3, lower_bound: float = 0.):
    '''
        We can theoretically calculate the cosine similarity of all items in our dataset with all other items in scikit-learn by using the cosine_similarity function, however the Data Scientists at ING found out this has some disadvantages:

        The sklearn version does a lot of type checking and error handling.
        The sklearn version calculates and stores all similarities in one go, while we are only interested in the most similar ones. Therefore it uses a lot more memory than necessary.
        To optimize for these disadvantages they created their own library which stores only the num_matches N highest matches in each row, and only the similarities above an (optional) threshold.

    :param A: CSR matrix of shape NxD
    :param B: CSR matrix of shape DxM
    :param topN: only keep the samples with num_matches-n scores
    :param lower_bound: specifies the threshold above which the samples are stored
    :return: CSR matrix of shape NxM, with entries being the cosine similarity scores
    '''

    N, _ = A.shape
    _, M = B.shape

    idx_dtype = np.int32

    nnz_max = N * topN

    indptr = np.zeros(N + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        N, M, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        topN,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(N, M))


def get_matches_df(sparse_matrix, name_vector, num_matches = None):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if num_matches:
        nr_matches = num_matches
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]


    matches_df = pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similairity': similairity})
    matches_df = matches_df[matches_df['similairity'] < 1]  # Remove all exact matches
    return matches_df