import numpy as np

file_path = "c:/Users/82108/Desktop/스터디 폴더/Support vector machine/SVM 1.py"

x_data, y_data = [], []
with open(file_path, 'r', encoding='utf8') as inFile:
    lines = inFile.readlines()

lines = lines[:100]

for line in lines:
    line = line.strip().split('\t')
    sentence, label = line[1], line[0]
    x_data.append(sentence)
    y_data.append(label)

print("x_data의 개수 : " + str(len(x_data)))
print("y_data의 개수 : " + str(len(y_data)))

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

label2index_dict = {'spam':0, 'ham':1}

indexing_x_data, indexing_y_data = [], []

for label in y_data:
    indexing_y_data.append(label2index_dict[label])

tokenizer.fit_on_texts(x_data)

indexing_x_data = tokenizer.texts_to_sequences(x_data)

print("x_data indexing 하기 전 : " + str(x_data[0]))
print("x_data indexing 하기 후 : " + str(indexing_x_data[0]))
print("y_data indexing 하기 전 : " + str(y_data[0]))
print("y_data indexing 하기 후 : " + str(indexing_y_data[0]))

from sklearn.svm import SVC

max_length = 60
for index in range(len(indexing_x_data)):
    length = len(indexing_x_data[index])

    if(length > max_length):
        indexing_x_data[index] = indexing_x_data[index][:max_length]
    elif(length < max_length):
        indexing_x_data[index] = indexing_x_data[index] + [0]*(max_length-length)

number_of_train = int(len(indexing_x_data)*0.9)

train_x = indexing_x_data[:number_of_train]
train_y = indexing_y_data[:number_of_train]
test_x = indexing_x_data[:number_of_train:]
test_y = indexing_y_data[:number_of_train:]

svm = SVC(kernel='linear', C=1e10)
svm.fit(train_x,train_y)

predict = svm.predict(test_x)

correct_count = 0
for index in range(len(predict)):
    if(test_y[index] == predict[index]):
        correct_count += 1

accuracy = 100.0*correct_count/len(test_y)

print("Accuracy: " + str(accuracy))

index2label = {0:"spam", 1:"ham"}

test_x_word = tokenizer.sequences_to_texts(test_x)

for index in range(len(test_x_word)):
    print()
    print("문장 : ", test_x_word[index])
    print("정답 : ", index2label[test_y[index]])
    print("모델 출력 : ", index2label[predict[index]])
