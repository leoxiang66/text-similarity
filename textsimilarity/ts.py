import textdistance as td
import numpy as np

class TextSimilarityAlgo:
    def __init__(self):
        self._all_algos = {
            'hamming': td.hamming,
            'mlipns': td.mlipns,
            'levenshtein': td.levenshtein,
            'damerau_levenshtein': td.damerau_levenshtein,
            'jaro_winkler': td.jaro_winkler,
            'strcmp95': td.strcmp95,
            'needleman_wunsch': td.needleman_wunsch,
            'gotoh': td.gotoh,
            'smith_waterman': td.smith_waterman,
            'jaccard': td.jaccard,
            'sorensen': td.sorensen,
            'tversky': td.tversky,
            'overlap': td.overlap,
            'tanimoto': td.tanimoto,
            'cosine': td.cosine,
            'monge_elkan': td.monge_elkan,
            'bag': td.bag,
            'ratcliff_obershelp': td.ratcliff_obershelp,
            'arith_ncd': td.arith_ncd,
            'rle_ncd': td.rle_ncd,
            'bwtrle_ncd': td.bwtrle_ncd,
            'sqrt_ncd': td.sqrt_ncd,
            'entropy_ncd': td.entropy_ncd,
            'bz2_ncd': td.bz2_ncd,
            'zlib_ncd': td.zlib_ncd,
            'editex': td.editex,
            'prefix': td.prefix,
            'postfix': td.postfix,
            'length': td.length,
            'identity': td.identity,
            'matrix': td.matrix,

        }

        self._current_algo = td.jaccard

    def print_all_algorithms(self):
        print(list(self._all_algos.keys()))
        print(f'Current algorithm is {self._current_algo}')


    def change_algorithm(self,algorithm:str):
        if algorithm in self._all_algos.keys():
            self._current_algo = self._all_algos[algorithm]
        else:
            raise NotImplementedError('This algorithm is not implemented yet.')

    def compute_similarity(self, text1, text2 ):
        return self._current_algo(text1,text2)

    def _create_one2many(self,dist_func):
        def new_func(mention, candidates: list):
            f = lambda candidate: dist_func(mention, candidate)
            return np.array(list(map(f, candidates)))

        return new_func

    def top_K_similarity_between_one_mention_and_many_candidates(self,k:int, mention, candidates:list):
        algo_one2many = self._create_one2many(self._current_algo)

        similarities = algo_one2many(mention, candidates)
        ids = np.argpartition(similarities, -k)[-k:]

        result = [candidates[a] for a in ids]
        f = lambda x: td.jaccard(mention, x)
        return sorted(result, key=f, reverse=True)

tsalgo = TextSimilarityAlgo()