# encoding=utf-8
import pandas as pd
import time

from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import load_dataset

corpus_type = 'bash'

def find_mixed_nn(simi, diffs, test_diff, bleu_thre):
    """Find the nearest neighbor using cosine simialrity and bleu score"""
    candidates = simi.argsort()[-bleu_thre:][::-1]
    max_score = 0
    max_idx = 0
    for j in candidates:
        score = sentence_bleu([diffs[j].split()], test_diff.split())
        if score > max_score:
            max_score = score
            max_idx = j
    return max_idx

def nngen(train_codes, train_nls, test_codes):
    counter = CountVectorizer()
    train_matrix = counter.fit_transform(train_codes)
    test_matrix = counter.transform(test_codes)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_nls = []
    for idx, test_simi in tqdm(enumerate(similarities), total=len(similarities)):
        max_idx = find_mixed_nn(test_simi, train_codes, test_codes[idx], bleu_thre=5)
        test_nls.append(train_nls[max_idx])
    return test_nls

train_codes = load_dataset("./data/" + corpus_type + "/train/train.code.src")
test_codes = load_dataset("./data/" + corpus_type + "/test/test.code.src")
train_nls = load_dataset("./data/" + corpus_type + "/train/train.nl.tgt")
out_nls = nngen(train_codes, train_nls, test_codes)
df = pd.DataFrame(out_nls)
df.to_csv("./result/"+corpus_type+"/NNGen.csv", index=False, header=None)