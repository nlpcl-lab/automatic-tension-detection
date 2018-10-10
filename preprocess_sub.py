import os, re
import argparse
import numpy as np
import pickle as pl
from os import walk
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from tensorflow.contrib.keras import preprocessing

from tqdm import tqdm
from konlpy.tag import Twitter

twitter = Twitter()

from libs import hangle, mongo, analytics

seungwon_user_id = '5b856c9d995fc115c6659c04'


def build_emb_matrix_and_vocab(embedding_model, keep_in_dict=100000, embedding_size=200):
    # 0 th element is the default vector for unknowns.

    emb_matrix = np.zeros((keep_in_dict + 2, embedding_size))
    word2index = {}
    index2word = {}
    for k in range(1, keep_in_dict + 1):
        word = embedding_model.wv.index2word[k - 1]
        # print('word: {}'.format(word))

        emb_matrix[k] = embedding_model[word]
        word2index[word] = k
        index2word[k] = word
    word2index['UNK'] = 0
    index2word[0] = 'UNK'
    word2index['STOP'] = keep_in_dict + 1
    index2word[keep_in_dict + 1] = 'STOP'
    return emb_matrix, word2index, index2word


def sent2index(sent, word2index):
    words = sent.strip().split(' ')
    sent_index = [word2index[word] if word in word2index else 0 for word in words]
    return sent_index


def get_sentence(index2word, sen_index):
    return ' '.join([index2word[index] for index in sen_index])


def gen_data(word2index, type='tension'):
    data = []

    db = mongo.get_db()
    videos = mongo.to_dicts(db.video.find({}))
    pbar = tqdm(total=len(videos))

    for video in videos:
        video = mongo.to_dict(db.video.find_one({'video_id': video['video_id']}))
        subs = mongo.to_dicts(db.sub.find({'video_id': video['video_id']}).sort('index', 1))
        users = mongo.to_dicts(db.user.find({}))

        video['sub_total'] = len(subs)

        target_users = []
        for user in users:
            context_no_label_total = db.label.find({
                'video_id': video['video_id'],
                'context': 0,
                'user_id': user['_id'],
            }).count()

            context_yes_label_total = db.label.find({
                'video_id': video['video_id'],
                'context': 1,
                'user_id': user['_id'],
            }).count()

            if video['sub_total'] == context_yes_label_total and context_no_label_total == context_yes_label_total:
                target_users.append(user)

        if len(target_users) < 2:
            pbar.update(1)
            continue

        offset = 1

        for i, sub in enumerate(subs):
            sub['category'] = 0

            if type == 'tension':
                for user in target_users:
                    label1 = db['label'].find_one({
                        'video_id': video['video_id'],
                        'context': 0,
                        'sub_index': sub['index'],
                        'user_id': user['_id'],
                    })

                    label2 = db['label'].find_one({
                        'video_id': video['video_id'],
                        'context': 1,
                        'sub_index': sub['index'],
                        'user_id': user['_id'],
                    })

                    sub['category'] = analytics.get_category(label1, label2)
            elif type == 'sentiment':
                label = db['label'].find_one({
                    'video_id': video['video_id'],
                    'context': 1,
                    'sub_index': sub['index'],
                    'user_id': seungwon_user_id,
                })

                sentiment = label['sentiment_label']
                if sentiment == '부정':
                    sub['category'] = 0
                elif sentiment == '중립':
                    sub['category'] = 1
                elif sentiment == '긍정':
                    sub['category'] = 2

            elif type == 'intent':
                label = db['label'].find_one({
                    'video_id': video['video_id'],
                    'context': 1,
                    'sub_index': sub['index'],
                    'user_id': seungwon_user_id,
                })

                sentiment = label['intent_label']
                if sentiment == '호응유도':
                    sub['category'] = 0
                elif sentiment == '의견':
                    sub['category'] = 1
                elif sentiment == '인용':
                    sub['category'] = 2
                elif sentiment == '일화':
                    sub['category'] = 3
                elif sentiment == '팩트':
                    sub['category'] = 4

            def text_to_indexes(text):
                n_text = hangle.normalize(text, english=True, number=True, punctuation=True)
                tokens = twitter.pos(n_text, stem=True)

                indexes = []
                for token in tokens:
                    word = '{}/{}'.format(token[0], token[1])

                    if word in word2index:
                        indexes.append(word2index[word])
                    else:
                        print('no word : {}'.format(word))
                        indexes.append(0)

                return indexes

            doc = []
            for j in range(max(0, i - offset), i):
                doc.append(text_to_indexes(subs[j]['text']))
            doc.append(text_to_indexes(sub['text']))
            for j in range(i + 1, min(i + offset + 1, len(subs))):
                doc.append(text_to_indexes(subs[j]['text']))

            # print('doc :', doc)

            data.append({
                'doc': doc,
                'start_ts': sub['start_ts'],
                'end_ts': sub['end_ts'],
                'category': sub['category'],
            })

        pbar.update(1)
    pbar.close()
    return data


def preprocess_sub(data, sent_length, max_rev_len, keep_in_dict=10000):
    ## As the result, each review will be composed of max_rev_len sentences. If the original review is longer than that, we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the word of 'UNK' to it. Also, we keep track of the actual number of sentences each review contains.
    data_formatted = []
    review_lens = []
    for i, item in enumerate(data):
        review = item['doc']
        review_formatted = preprocessing.sequence.pad_sequences(review, maxlen=sent_length, padding="post", truncating="post", value=keep_in_dict + 1)
        review_len = review_formatted.shape[0]
        review_lens.append(review_len if review_len <= max_rev_len else max_rev_len)
        lack_len = max_rev_length - review_len
        review_formatted_right_len = review_formatted
        if lack_len > 0:
            # extra_rows = np.zeros([lack_len, sent_length], dtype=np.int32)
            extra_rows = np.full((lack_len, sent_length), keep_in_dict + 1)
            review_formatted_right_len = np.append(review_formatted, extra_rows, axis=0)
        elif lack_len < 0:
            row_index = [max_rev_length + i for i in list(range(0, -lack_len))]
            review_formatted_right_len = np.delete(review_formatted, row_index, axis=0)
        data_formatted.append(review_formatted_right_len)

        data[i]['doc'] = review_formatted_right_len
        data[i]['sub_lens'] = review_formatted.shape[0]


def divide(data, train_prop):
    import random
    # 예시
    # x = [1, 2, 3, 4, 5]
    # y = [0, 0, 1, 0, 0]
    # train_prop = 0.2

    random.seed(1234)

    # tmp: [3 4 1 0 2], 랜덤순열
    random.shuffle(data)

    train = data[:round(train_prop * len(data))]
    test = data[-(len(data) - round(train_prop * len(data))):]
    return train, test


def get_df(data):
    label = []
    doc = []
    length = []

    for item in data:
        label.append(item['category'])
        doc.append(item['doc'])
        length.append(item['sub_lens'])

    return pd.DataFrame({'label': label, 'doc': doc, 'length': length})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some important parameters.')
    parser.add_argument('-s', '--sent_length', type=int, default=20,
                        help='fix the sentence length in all reviews')
    parser.add_argument('-r', '--max_rev_length', type=int, default=5,
                        help='fix the maximum review length')

    args = parser.parse_args()
    sent_length = args.sent_length
    max_rev_length = args.max_rev_length

    print('sent length is set as {}'.format(sent_length))
    print('rev length is set as {}'.format(max_rev_length))
    working_dir = "./data/"
    fname = os.path.join(working_dir, "embedding.bin")
    embedding_model = Word2Vec.load(fname)

    print("generate word to index dictionary and inverse dictionary...")

    keep_in_dict = len(embedding_model.wv.vocab)
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model, keep_in_dict=keep_in_dict)
    print("format each review into sentences, and also represent each word by index...")

    print("preprocess each sub...")

    print("save word embedding matrix ...")
    emb_filename = os.path.join(working_dir, "emb_matrix.bin")
    # emb_matrix.dump(emb_filename)
    pl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb"))

    data = gen_data(word2index, type='intent')
    preprocess_sub(data, keep_in_dict=keep_in_dict, sent_length=sent_length, max_rev_len=max_rev_length)

    train, test = divide(data, 0.9)

    df_train = get_df(train)
    df_train_filename = os.path.join(working_dir, "df_train.bin")
    df_train.to_pickle(df_train_filename)

    df_test = get_df(test)
    df_test_filename = os.path.join(working_dir, "df_test.bin")
    df_test.to_pickle(df_test_filename)
