from nlgeval import compute_metrics

methods = ['Levenshtein', 'Jaccard', 'VSM', 'LSI', 'BM25', 'NNGen', 'W2V']

for m in methods:
    print(m)
    compute_metrics('result/bash/'+m+'.csv', ['data/bash/test/test.nl.tgt'], no_skipthoughts=True, no_glove=True)
    print('---------------------------')