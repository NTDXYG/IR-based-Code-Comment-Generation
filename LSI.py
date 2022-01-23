import gensim
from gensim import corpora
from gensim import similarities
import pandas as pd
from tqdm import tqdm

from utils import load_dataset

corpus_type = 'bash'
train_code_list = load_dataset("./data/"+corpus_type+"/train/train.code.src")
test_code_list = load_dataset("./data/"+corpus_type+"/test/test.code.src")
train_comment_list = load_dataset("./data/"+corpus_type+"/train/train.nl.tgt")


corpus = train_code_list
clean_list = [doc.split(" ") for doc in corpus]

# 下一步准备 Document-Term 矩阵
# 创建语料的词语词典，每个单独的词语都会被赋予一个索引
dictionary = corpora.Dictionary(clean_list)

# 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_list]

# 创建tfidf对象
# LSI LDA 等模型通常用bow向量或是tfidf向量作为语料输入，上面的doc_term_matrix就是bow向量
tfidf = gensim.models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]

# 构建LSI模型进行训练
Lsi = gensim.models.LsiModel
lsimodel = Lsi(corpus_tfidf, id2word=dictionary, num_topics=100)

# 用待检索的文档向量初始化一个相似度计算的对象
index = similarities.MatrixSimilarity(lsimodel[corpus_tfidf])

# 保存相似度矩阵，index中每一行表示一篇文档，列表示主题，列数与前面定义的num_topics数是一致的
# index.save('./sim_mat.index')
# index = similarities.MatrixSimilarity.load('./sim_mat.index')

# 计算一篇文档与现有语料中所有文档的（余弦）相似度
# 这里先取语料中的第一篇文档试验一下

test_list = [doc.split(" ") for doc in test_code_list]

data_list = []
for i in tqdm(range(len(test_code_list))):
    query_bow = dictionary.doc2bow(test_list[i])

    # tfidf向量化
    query_tfidf = tfidf[query_bow]

    # 用之前训练好的LSI模型将其映射到topic空间
    query_vec = lsimodel[query_tfidf]

    # 检查query在index中的相似度
    sim = index[query_vec]

    firstMax = max(list(enumerate(sim)),key=lambda x:x[1])
    suggest_nl = train_comment_list[firstMax[0]]
    data_list.append(suggest_nl)

df = pd.DataFrame(data_list)
df.to_csv("./result/" + corpus_type + "/LSI.csv", index=False, header=None)
