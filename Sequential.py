from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.callbacks import ModelCheckpoint



def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# load a pre-defined list of photo identifiers
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

def create_sequences(tokenizer, descriptions, photos):
    x, y = list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0] 
            seq = to_categorical([seq], num_classes=vocab_size)[0]
            x.append(photos[key][0])  
            y.append(seq)
    return array(x), array(y)



# load training set

filename = '/scratch/user/mahin/ML/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('/scratch/user/mahin/ML/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features('/scratch/user/mahin/ML/features.pkl', train)
print('Photos: train=%d' % len(train_features))

tokenizer = load(open('/scratch/user/mahin/run2ml/tokenizer.pkl', 'rb'))
#tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)


# load validation set 

filename = '/scratch/user/mahin/ML/Flickr8k_text/Flickr_8k.devImages.txt'
val = load_set(filename)
print('Dataset: %d' % len(val))

val_descriptions = load_clean_descriptions('/scratch/user/mahin/ML/descriptions.txt', val)
print('Descriptions: val=%d' % len(val_descriptions))

val_features = load_photo_features('/scratch/user/mahin/ML/features.pkl', val)
print('Photos: val=%d' % len(val_features))



# build NN

def define_model(max_length, vocab_size):
    model = Sequential()
    model.add(Embedding(max_length, 256))
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# train NN

model = define_model(max_length, vocab_size)
print(model.summary())


xtrain, ytrain = create_sequences(tokenizer, train_descriptions, train_features)
xval,yval = create_sequences(val_descriptions, val_features)

filepath = '/scratch/user/mahin/run3ml/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(xtrain,
          ytrain,
          epochs=10,
          verbose=1,
          callbacks=[checkpoint],
          validation_data=(xval,yval))

