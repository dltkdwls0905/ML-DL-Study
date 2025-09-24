from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np

def make_dic(texts):

  words = []
  m = Mecab()

  for snt in texts:
    res = m.pos(snt)
    for (lex,pos) in res:
      if pos[0] == 'N' or pos[0] == 'V':
        words.append(lex)

  dic = FreqDist(np.hstack(words))
  print('사전 크기: {0:d}'.format(len(dic)))

  indexes = {}
  words = {}

  for num, word in enumerate(dic):
    idx = num+2
    indexes[word]= idx
    words[idx] = word

  indexes['pad'] = 1
  words[1] = 'pad'
  indexes['unk'] = 0
  words[0] = 'unk'

  return (indexes, words)

def word2index(indexes, word):
  idx = indexes[word] if word in indexes else indexes['unk']
  return idx

def index2word(words, index):
  w = words[index] if index in words else -1
  return w

texts = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를",
         "잎새에 이는 바람에도 나는 괴로워 했다.",
         "별을 노래하는 마음으로 모든 죽어 가는 것을 사랑해야지",
         "그리고 나한테 주어진 길을 걸어가야겠다.",
         "오늘 밤에도 별이 바람에 스치운다."]

(indexes, words) = make_dic(texts)

print(word2index(indexes,'하늘'), word2index(indexes,'학교'))
print(index2word(words,4), index2word(words,0))

rom konlpy.tag import Mecab
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

texts = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를",
         "잎새에 이는 바람에도 나는 괴로워 했다.",
         "별을 노래하는 마음으로 모든 죽어 가는 것을 사랑해야지",
         "그리고 나한테 주어진 길을 걸어가야겠다.",
         "오늘 밤에도 별이 바람에 스치운다."]

# Word2Vec 학습에 사용할 데이터 만들기

m = Mecab()
result = []

for sent in texts:
  tag = m.pos(sent)
  words = []
  for (lex,pos) in tag:
    if pos[0] == 'N':
      words.append(lex)
  result.append(words)

print(result,"\n")

# Word2Vec 학습시키기
model = Word2Vec(sentences=result, window=1, min_count=1, workers=1, sg=0)

# 값 읽어오기
print(model.wv['하늘'])
# 유사한 단어 가져오기
print(model.wv.most_similar("하늘"),"\n")

# Word2Vec 모델 저장하기
model.wv.save_word2vec_format('c:/Users/82108/Desktop/스터디 폴더/Textrepresentation/test_w2v')

# Word2Vec 모델 로드하기
loaded_model = KeyedVectors.load_word2vec_format("c:/Users/82108/Desktop/스터디 폴더/Textrepresentation/test_w2v")

# 값 읽어오기 (wv는 현 버전에서 더 이상 필요하지 않아 제거함)
print(loaded_model['하늘'])
# 유사한 단어 가져오기
print(loaded_model.most_similar("하늘"))

import torch
import torch.nn as nn

train_data = 'I like deep learning I like NLP I enjoy flying'


word_set = set(train_data.split())


vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['unk'] = 0
vocab['pad'] = 1
print(vocab)


embedding_table = torch.FloatTensor([[ 0.0,  0.0,  0.0],
                                     [ 0.0,  0.0,  0.0],
                                     [ 0.1,  0.8,  0.3],
                                     [ 0.7,  0.8,  0.2],
                                     [ 0.1,  0.8,  0.7],
                                     [ 0.9,  0.2,  0.1],
                                     [ 0.1,  0.1,  0.9],
                                     [ 0.2,  0.1,  0.7],
                                     [ 0.3,  0.1,  0.1]])


input_snt = 'I like football'.split()


idxes=[]

for word in input_snt:
  idx = vocab[word] if word in vocab else vocab['unk']
  idxes.append(idx)

idxes = torch.LongTensor(idxes)


lookup_result = embedding_table[idxes, :]
print(lookup_result)


embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3, padding_idx=1)
print(embedding_layer.weight)
