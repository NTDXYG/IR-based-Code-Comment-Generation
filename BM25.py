import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np

from utils import load_dataset

corpus_type = 'bash'
train_code_list = load_dataset("./data/"+corpus_type+"/train/train.code.src")
test_code_list = load_dataset("./data/"+corpus_type+"/test/test.code.src")
train_comment_list = load_dataset("./data/"+corpus_type+"/train/train.nl.tgt")

corpus = train_code_list
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def get_top_index(code, n):
    scores = bm25.get_scores(code)
    top_n = np.argsort(scores)[::-1][:n]
    return top_n

result_list = []
for i in tqdm(range(len(test_code_list))):
    index = get_top_index(test_code_list[i].split(" "), n=1)[0]
    result_list.append(train_comment_list[index])

df = pd.DataFrame(result_list)
df.to_csv("./result/"+corpus_type+"/BM25.csv", index=False, header=None)