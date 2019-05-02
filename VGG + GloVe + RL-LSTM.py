import string
import numpy as np
import pandas as pd
from numpy import array
from pickle import dump, load
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.models import Model
from keras import Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint



# Glove vectors

embeddings_index = {} 
f = open('glove.6B.200d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))



def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
                
                
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

                
def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def data_generator(descriptions, photos, wordtoix, max_length):
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        photo = photos[key]
        for desc in desc_list:
            seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)



train = load_set('/scratch/user/mahin/ML/Flickr8k_text/Flickr_8k.trainImages.txt')
train_descriptions = load_clean_descriptions('/scratch/user/mahin/run6ml/data/descriptions.txt', train)
train_features = load_photo_features('/scratch/user/mahin/ML/features.pkl', train)

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)


# Consider only words which occur at least 10 times in the corpus
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1
    
vocab_size = len(ixtoword) + 1 

max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)



val = load_set('/scratch/user/mahin/ML/Flickr8k_text/Flickr_8k.devImages.txt')
val_descriptions = load_clean_descriptions('/scratch/user/mahin/run6ml/data/descriptions.txt', val)
val_features = load_photo_features('/scratch/user/mahin/ML/features.pkl', val)


# Embbedding 

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def bleu_score(hypotheses, references):
	return 1 - nltk.translate.bleu_score.corpus_bleu(references, hypotheses)

# build NN
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.summary()
    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    model.compile(loss=bleu_score, optimizer='sgd')

    return model



model = define_model(vocab_size, max_length)

X1train, X2train, ytrain = data_generator(train_descriptions, train_features, wordtoix, max_length)
X1val, X2val, yval = data_generator(val_descriptions, val_features, wordtoix, max_length)


X1train = X1train.reshape(546, 4096)
X1val = X1val.reshape(174, 4096)


# train NN

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit([X1train, X2train],
          ytrain,
          epochs=2,
          verbose=1,
          callbacks=[checkpoint],
          validation_data=([X1val, X2val], yval))



# Evaluation

from nltk.translate.bleu_score import corpus_bleu

def evaluate_model(model, descriptions, photos, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        image = encoding_test[key].reshape((1,4096))
        yhat = greedySearch(image)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
    
    
test = load_set('/home/mahin/Desktop/ML-Project/practice/Flickr8k_text/Flickr_8k.testImages.txt')
print('test: test=%d' % len(test)) 

test_descriptions = load_clean_descriptions('/home/mahin/Desktop/ML-Project/practice/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))

with open("/home/mahin/Desktop/ML-Project/practice/features.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)
    
model = load_model('/home/mahin/Desktop/ML-Project/run6/result/vgg.h5')

evaluate_model(model, test_descriptions, encoding_test, max_length)

