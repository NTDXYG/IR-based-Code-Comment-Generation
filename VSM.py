# encoding=utf-8
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils import load_dataset

corpus_type = 'bash'

def vsm(train_codes, train_nls, test_codes):
    counter = TfidfVectorizer()
    train_matrix = counter.fit_transform(train_codes)
    test_matrix = counter.transform(test_codes)
    similarities = cosine_similarity(test_matrix, train_matrix)
    test_nls = []
    for idx, test_simi in tqdm(enumerate(similarities), total=len(similarities)):
        max_idx = test_simi.argsort()[-1]
        test_nls.append(train_nls[max_idx])
    return test_nls

train_codes = load_dataset("./data/" + corpus_type + "/train/train.code.src")
test_codes = load_dataset("./data/" + corpus_type + "/test/test.code.src")
train_nls = load_dataset("./data/" + corpus_type + "/train/train.nl.tgt")
out_nls = vsm(train_codes, train_nls, test_codes)
df = pd.DataFrame(out_nls)
df.to_csv("./result/"+corpus_type+"/VSM.csv", index=False, header=None)