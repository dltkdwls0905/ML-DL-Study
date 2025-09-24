from konlpy.tag import Mecab
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
model = Word2Vec(sentences=result, size=10, window=1, min_count=1, workers=1, sg=0)

# 값 읽어오기
print(model.wv['하늘'])
# 유사한 단어 가져오기
print(model.wv.most_similar("하늘"),"\n")

# Word2Vec 모델 저장하기
model.wv.save_word2vec_format('C:/Users/82108/Desktop/스터디 폴더/Text Representation/test_w2v')

# Word2Vec 모델 로드하기
loaded_model = KeyedVectors.load_word2vec_format("C:/Users/82108/Desktop/스터디 폴더/Text Representation/test_w2v")

# 값 읽어오기
print(loaded_model.wv['하늘'])
# 유사한 단어 가져오기
print(loaded_model.wv.most_similar("하늘"))
