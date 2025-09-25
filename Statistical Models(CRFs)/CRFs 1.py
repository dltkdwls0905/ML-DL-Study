import os
import sklearn_crfsuite
from sklearn_crfsuite import metrics


# 파일 경로
file_path = "C:/Users/82108\Desktop/스터디 폴더/Statistical Models(CRFs)/spacing_data.txt"

# "spacing_data.txt" 파일을 읽고 lines에 읽은 데이터를 저장
with open(file_path, "r", encoding="utf8") as inFile:
    lines = inFile.readlines()

# 데이터를 음절로 이루어진 문장과 정답 값으로 나누어 저장
datas = []
for line in lines:
    pieces = line.strip().split("\t")
    eumjeol_sequence, label = pieces[0].split(), pieces[1].split()
    datas.append((eumjeol_sequence, label))

number_of_train_datas = int(len(datas)*0.9)

train_datas = datas[:number_of_train_datas]
test_datas = datas[number_of_train_datas:]

print("train_datas 개수 : " + str(len(train_datas)))
print("test_datas 개수 : " + str(len(test_datas)))

def sent2feature(eumjeol_sequence):
  features = []
  sequence_length = len(eumjeol_sequence)
  for index, eumjeol in enumerate(eumjeol_sequence):
      feature = { "BOS":False, "EOS":False, "WORD":eumjeol, "IS_DIGIT":eumjeol.isdigit() }

      if(index == 0):
          feature["BOS"] = True
      elif(index == sequence_length-1):
          feature["EOS"] = True

      if(index-1 >= 0):
          feature["-1_WORD"] = eumjeol_sequence[index-1]
      if(index-2 >= 0):
          feature["-2_WORD"] = eumjeol_sequence[index-2]

      if(index+1 <= sequence_length-1):
          feature["+1_WORD"] = eumjeol_sequence[index+1]
      if(index+2 <= sequence_length-1):
          feature["+2_WORD"] = eumjeol_sequence[index+2]

      features.append(feature)

  return features


train_x, train_y = [], []
for eumjeol_sequence, label in train_datas:
    train_x.append(sent2feature(eumjeol_sequence))
    train_y.append(label)

test_x, test_y = [], []
for eumjeol_sequence, label in test_datas:
    test_x.append(sent2feature(eumjeol_sequence))
    test_y.append(label)

crf = sklearn_crfsuite.CRF()
crf.fit(train_x, train_y)

def show_predict_result(test_datas, predict):
  for index_1 in range(len(test_datas)):
      eumjeol_sequence, correct_labels = test_datas[index_1]
      predict_labels = predict[index_1]

      correct_sentence, predict_sentence = "", ""
      for index_2 in range(len(eumjeol_sequence)):
          if(index_2 == 0):
              correct_sentence += eumjeol_sequence[index_2]
              predict_sentence += eumjeol_sequence[index_2]
              continue

          if(correct_labels[index_2] == "B"):
              correct_sentence += " "
          correct_sentence += eumjeol_sequence[index_2]

          if (predict_labels[index_2] == "B"):
              predict_sentence += " "
          predict_sentence += eumjeol_sequence[index_2]

      print("정답 문장 : " + correct_sentence)
      print("출력 문장 : " + predict_sentence)
      print()

predict = crf.predict(test_x)

print("Accuracy score : " + str(metrics.flat_accuracy_score(test_y, predict)))
print()

print("10개의 데이터에 대한 모델 출력과 실제 정답 비교")
print()

show_predict_result(test_datas[:10], predict[:10])