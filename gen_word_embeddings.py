import os
from tqdm import tqdm
from gensim.models import Word2Vec

from konlpy.tag import Twitter

twitter = Twitter()

from libs import hangle, mongo, analytics


def get_train():
    train = []
    db = mongo.get_db()

    videos = mongo.to_dicts(db.video.find({}))

    sub_total = db.sub.find({}).count()
    pbar = tqdm(sub_total)

    print('total sub: {}'.format(sub_total))

    offset = 3
    for video in videos:
        subs = mongo.to_dicts(db.sub.find({'video_id': video['video_id']}).sort('index', 1))
        for i, sub in enumerate(subs):
            text = ''

            for j in range(max(0, i - offset), i):
                text += subs[j]['text'] + ' '
            text += sub['text'] + ' '
            for j in range(i + 1, min(i + offset, len(subs))):
                text += subs[j]['text'] + ' '

            text = hangle.normalize(text, english=True, number=True, punctuation=True)
            tokens = twitter.pos(text, stem=True)

            doc = []
            for token in tokens:
                doc.append('{}/{}'.format(token[0], token[1]))

            train.append(doc)
            pbar.update(1)

    pbar.close()

    return train


if __name__ == "__main__":
    working_dir = "./data"
    embedding_size = 200
    fname = os.path.join(working_dir, "embedding.bin")

    train = get_train()
    embedding_model = Word2Vec(train, size=embedding_size, window=5, min_count=5)
    embedding_model.save(fname)

    word1 = "우리/Noun"
    word2 = "악보/Noun"
    print("similar words11 of {}:".format(word1))
    print(embedding_model.most_similar(word1))
    print("similar words of {}:".format(word2))
    print(embedding_model.most_similar(word2))

    pass
