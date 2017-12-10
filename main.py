import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import requests
import requests.auth
import pickle
import praw
import pprint
import markovify
import markovify
import re


def login():
    client_id = 'WrEgtxUIZDh1gA'
    client_secret = 'Mv4coL8Plg9NsngSQE5Y1nSLrpE'
    user_agent = 'Swoleacceptance_bot:v0.1 by /u/swoleacceptance_bot'

    return praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)

def login_and_pull(r, limit=100):
    all_comments = ""
    submissions = r.subreddit('swoleacceptance').top(limit=limit)
    print(vars(submissions))
    # print('Got {} submissions'.format(len(submissions)))
    sub_count = 1
    for submission in submissions:
        print('Submission number ' + str(sub_count))
        sub_count += 1
        if submission.is_self:
            all_comments += submission.selftext
        for comment in submission.comments:
            if comment.score > 40:
                all_comments += '.'
                all_comments += comment.body
            # pprint.pprint(vars(comment))

    pickle.dump(all_comments, open("all_comments_v1.pickle", "wb"))
    return all_comments

def make_model(text, save_name):
    # Build the model.
    text_model = markovify.Text(text)

    # # Print five randomly-generated sentences
    # for i in range(5):
    #     print(text_model.make_sentence())
    # print('----------------------------------')

    # Print three randomly-generated sentences of no more than 140 characters
    for i in range(3):
        print(text_model.make_short_sentence(140))
    print('---------------------------------')
    pickle.dump(text_model, open(save_name, "wb"))

    return text_model

def make_current_top_model(r):
    submissions = r.subreddit('swoleacceptance').hot(limit=5)
    for submission in submissions:
        # pprint(vars(submission))
        if submission.is_self:
            pprint.pprint(submission.selftext)
            return submission.selftext

def do_markov():
    reddit = login()
    # all_text = login_and_pull(reddit, limit=600)
    topic_text = make_current_top_model(reddit)
    # model_comments = make_model(all_text, 'markov_comments_model_v1.p')
    model_comments = pickle.load(open("markov_comments_model_v1.p", "rb"))
    model_topic = make_model(topic_text, 'markov_topic_model_v1.p')

    model_combo = markovify.combine([ model_comments, model_topic ], [ 5, 1 ])
    for i in range(15):
        print(model_combo.make_sentence())

def do_ltsm():
    raw_text = pickle.load(open("all_comments_v1.pickle", "rb"))
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', raw_text, flags=re.MULTILINE)

    # create mapping of unique chars to integers
    chars = sorted(list(set(text)))
    print("Total chars before cleaning: ", len(chars))

    for char in chars:
        # print(char + '  ' + str(text.count(char)))
        if text.count(char) < 10:
            print('    Removed ' + char)
            text = text.replace(char, "")

    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    n_chars = len(text)
    n_vocab = len(chars)
    print("Total chars after cleaning: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(n_vocab) # normalize
    y = np_utils.to_categorical(dataY) # one hot encode the output variable

    # define the LSTM model
    model = Sequential()
    print(y.shape, X.shape)
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))

    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

do_ltsm()