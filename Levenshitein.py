import pandas as pd
from tqdm import tqdm
from textdistance import levenshtein
from utils import load_dataset

corpus_type = 'bash'
train_code_list = load_dataset("./data/"+corpus_type+"/train/train.code.src")
test_code_list = load_dataset("./data/"+corpus_type+"/test/test.code.src")
train_comment_list = load_dataset("./data/"+corpus_type+"/train/train.nl.tgt")


data_list = []
for i in tqdm(range(len(test_code_list))):
    result_list = []
    for j in range(len(train_code_list)):
        score = levenshtein.normalized_similarity(train_code_list[j], test_code_list[i])
        result_list.append((score, j))
    result_list.sort(reverse=True)
    max_score = 0
    index = 0
    result = result_list[index]
    suggest_nl = train_comment_list[result[1]]
    data_list.append(suggest_nl)

df = pd.DataFrame(data_list)
df.to_csv("./result/"+corpus_type+"/Levenshtein.csv", index=False, header=None)