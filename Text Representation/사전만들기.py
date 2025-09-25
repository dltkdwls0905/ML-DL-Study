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

  # 단어의 빈도수 계산
  dic = FreqDist(np.hstack(words))
  print('사전 크기: {0:d}'.format(len(dic)))

  # 인덱스를 저장할 변수 초기화
  indexes = {}
  words = {}

  # 단어에 고유 번호(인덱스) 부여
  for num, word in enumerate(dic):
    idx = num+2
    indexes[word]= idx
    words[idx] = word

  # 문장 길이 정규화에 사용한 패딩 인덱스
  indexes['pad'] = 1
  words[1] = 'pad'
  # 사전에 없는 단어에 대한 인덱스
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