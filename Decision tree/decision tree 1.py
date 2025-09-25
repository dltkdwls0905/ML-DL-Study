import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

file_path = "C:/Users/82108/Desktop/스터디 폴더/Decision tree/PlayTennis.csv"
datas = pd.read_csv(file_path)

print(datas)

label_encoder = LabelEncoder()

target_names = label_encoder.fit(datas['play']).classes_

print("target_names : {}".format(target_names))

datas['outlook'] = label_encoder.fit_transform(datas['outlook'])
datas['temp'] = label_encoder.fit_transform(datas['temp'])
datas['humidity'] = label_encoder.fit_transform(datas['humidity'])
datas['windy'] = label_encoder.fit_transform(datas['windy'])
datas['play'] = label_encoder.fit_transform(datas['play'])

print(datas)

x_data, y_data = datas.drop(['play'], axis =1), datas['play']

print(x_data)
print()
print(y_data)

decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy')
train_result = decision_tree.fit(x_data, y_data)

graph = graphviz.Source(tree.export_graphviz(train_result, out_file=None, feature_names=x_data.columns, class_names=target_names))

print(graph)

predict_result = decision_tree.predict(x_data)

print(predict_result == y_data)
