import textsimilarity.TfIdf as tfidf

if __name__ == '__main__':
    ca = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    cb = [
        'This is the first document. and yes why',
        'This document is the second document. and yes whyand yes why',
        'And this is the third one. and yes whyand yes why',
        'Is this the first document? and yes whyand yes why',
    ]

    matches = tfidf.tfidf_sim_pipeline(ca, cb, 2)
    print(matches)