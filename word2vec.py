import os.path
from gensim.models import Word2Vec
import pickle
import faiss
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import load_dataset

corpus_type = 'bash'
train_code_list = load_dataset("./data/"+corpus_type+"/train/train.code.src")
test_code_list = load_dataset("./data/"+corpus_type+"/test/test.code.src")
train_nl_list = load_dataset("./data/"+corpus_type+"/train/train.nl.tgt")


model = Word2Vec(sentences=[d.split() for d in train_code_list], vector_size=256, window=4, min_count=1, workers=-1, sg=1, epochs=20)
dim = 256

def averageVector(many_vectors, column_num):
    """
    求多个向量的权值向量
    :param many_vector:
    :column_num:向量列数
    """
    average_vector = []
    for i in range(0, column_num, 1):
        average_vector.append(0)
    row_num = len(many_vectors)
    # 先求出各个列权重之和  后面再求平均值
    row_index = 0
    for weight_index, vector in enumerate(many_vectors):
        for i in range(0, column_num, 1):
            average_vector[i] += float(vector[i])
        row_index += 1
    for i in range(0, column_num, 1):
        if(row_num == 0):
            row_num = 1
        average_vector[i] = average_vector[i] / row_num
    return average_vector

# 根据word2vec词向量均值
# splited_words是形如[word1 word2 word3 ... wordn]的一个句子
def get_sentence_matrix(splited_words, w2v_model):
    sentences_matrix = []
    index = 0
    # 平均特征矩阵
    while index < len(splited_words):
        words_matrix = []
        words = splited_words[index].split(" ")
        # 得出各个词的特征向量  并形成一个矩阵  然后计算平均值  就得到该句子的特征向量
        for word in words:
            # 当前词是在Word2vec模型中，self.model为词向量模型
            if word in w2v_model:
                words_matrix.append(np.array(w2v_model[word]))
        # 将words_matrix求均值
        feature = averageVector(many_vectors=words_matrix,
                                column_num=w2v_model.vector_size)
        sentences_matrix.append(feature)
        index += 1
    return sentences_matrix

class Retrieval(object):
    def __init__(self, vec_file, model):
        w2v_model = model.wv
        if(os.path.isfile(vec_file)):
            print('loading ...')
            f = open(vec_file, 'rb')
            self.vec = pickle.load(f)
            f.close()
        else:
            print('transform vec ...')
            vecs = []
            for code in tqdm(train_code_list):
                vec = get_sentence_matrix([code], w2v_model)
                vecs.append(np.array(vec)[0])
            vecs = np.array(vecs)
            f = open(vec_file, 'wb')
            pickle.dump(vecs, f)
            f.close()
            self.vec = vecs
            print('save vec ...')

        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None
        self.w2v_model = w2v_model

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        for i in range(len(train_code_list)):
            all_texts.append(train_code_list[i])
            all_ids.append(i)
            all_vecs.append(self.vec[i].reshape(1,-1))
        all_vecs = np.concatenate(all_vecs, 0)
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, topK):
        body = get_sentence_matrix([code], self.w2v_model)
        vec = np.array(body)[0].reshape(1, -1).astype('float32')
        _, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        max_idx = sim_idx[0]
        return train_nl_list[max_idx]

if __name__ == '__main__':
    IR = Retrieval(vec_file= 'w2v.pkl', model=model)
    print("Sentences to vectors")
    IR.encode_file()
    print("加载索引")
    IR.build_index(n_list=1)
    IR.index.nprob = 1
    data_list = []
    for i in tqdm(range(len(test_code_list))):
        sim_nl = IR.single_query(test_code_list[i], topK=1)
        data_list.append(sim_nl)

    df = pd.DataFrame(data_list)
    df.to_csv("./result/"+corpus_type+"/W2V.csv", index=False,header=None)
