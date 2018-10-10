import json
import os
import random
import pickle as pl
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

working_dir = "./data/"
log_dir = "./logs"
train_filename = os.path.join(working_dir, "df_train.bin")
test_filename = os.path.join(working_dir, "df_test.bin")
emb_filename = os.path.join(working_dir, "emb_matrix.bin")

df_train = pd.read_pickle(train_filename)
df_test = pd.read_pickle(test_filename)
(emb_matrix, word2index, index2word) = pl.load(open(emb_filename, "rb"))


def get_x_from_df(df_data):
    docs = df_data['doc'].tolist()
    print('len(docs) :', len(docs))
    x = []
    for doc in docs:
        item = ''
        for sen in doc:
            for word in sen:
                item += index2word[word] + ' '
                # item.append(emb_matrix[word].tolist())

        x.append(item)

    return x


train_x = get_x_from_df(df_train)
train_y = df_train['label'].tolist()

test_x = get_x_from_df(df_test)
test_y = df_test['label'].tolist()

count_vec = CountVectorizer(lowercase=False, ngram_range=(1, 2))
count_vec.fit(train_x + test_x)

train_x = count_vec.transform(train_x)
test_x = count_vec.transform(test_x)

cls = LogisticRegression()
cls.fit(train_x, train_y)

predictions = cls.predict(test_x)

# ['긴장도 상', '긴장도 중', '긴장도 하']
# ['부정', '중립', '긍정']
# ['호응유도', '의견','인용','일화', '사실']

print(classification_report(test_y, predictions, target_names=['호응유도', '의견', '인용', '일화', '사실']))
score = cls.score(test_x, test_y)
print("accuracy: {}".format(score))
print('confusion: ', confusion_matrix(test_y, predictions))
