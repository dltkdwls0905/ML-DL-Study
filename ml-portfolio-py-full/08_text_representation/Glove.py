from konlpy.tag import Mecab
import numpy as np
from glove import Corpus, Glove

texts = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를",
         "잎새에 이는 바람에도 나는 괴로워 했다.",
         "별을 노래하는 마음으로 모든 죽어 가는 것을 사랑해야지",
         "그리고 나한테 주어진 길을 걸어가야겠다.",
         "오늘 밤에도 별이 바람에 스치운다."]

# GloVe 학습에 사용할 데이터 만들기

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

# Co-Occurrence Matrix 생성
corpus = Corpus() 
corpus.fit(result, window=1)

# GloVe 학습시키기
glove = Glove(no_components=10, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=1, verbose=True)
glove.add_dictionary(corpus.dictionary)

# 값 읽어오기
print(glove.word_vectors[glove.dictionary['하늘']])
# 유사한 단어 가져오기
print(glove.most_similar("하늘"),"\n")

# GloVe 모델 저장하기
glove.save('C:/Users/82108/Desktop/스터디 폴더/Text Representation/test_glove')

# GloVe 모델 로드하기
loaded_model = glove.load("C:/Users/82108/Desktop/스터디 폴더/Text Representation/test_glove")

# 값 읽어오기
print(loaded_model.word_vectors[glove.dictionary['하늘']])
# 유사한 단어 가져오기
print(loaded_model.most_similar("하늘"))
