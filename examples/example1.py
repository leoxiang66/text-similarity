if __name__ == '__main__':
    from textsimi import textsimilarity

    print(textsimilarity.compute_similarity('hello','hi'))
    print(textsimilarity.top_K_similarity_between_one_mention_and_many_candidates(2,'hello',['hi','hihi','have fun','hey']))

    from textsimi import textsimilarity_model_based
    import numpy as np

    a = np.array([0.5,0.5])
    b = np.array([0.6,0.7])
    c = np.array([0.7,1])
    d = np.array([0.1,0.2])
    print(textsimilarity_model_based.compute_similarity(a,b))
    print(textsimilarity_model_based.top_K_similarity_between_one_mention_and_many_candidates(2,a,[b,c,d]))