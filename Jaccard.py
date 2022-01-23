import pandas as pd
from tqdm import tqdm

from utils import load_dataset

corpus_type = 'bash'
train_code_list = load_dataset("./data/"+corpus_type+"/train/train.code.src")
test_code_list = load_dataset("./data/"+corpus_type+"/test/test.code.src")
train_comment_list = load_dataset("./data/"+corpus_type+"/train/train.nl.tgt")

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

data_list = []
for i in tqdm(range(len(test_code_list))):
    result_list = []
    for j in range(len(train_code_list)):
        score = sim_jaccard(train_code_list[j].split(), test_code_list[i].split())
        result_list.append((score, j))
    result_list.sort(reverse=True)
    max_score = 0
    index = 0
    result = result_list[index]
    suggest_nl = train_comment_list[result[1]]
    data_list.append(suggest_nl)

df = pd.DataFrame(data_list)
df.to_csv("./result/"+corpus_type+"/Jaccard.csv", index=False, header=None)