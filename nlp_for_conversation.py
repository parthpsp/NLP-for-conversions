import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import json

#load jason file
with open("datasets_conversation.json", 'r') as f:
    datastore = json.load(f)
    
vocab_size = 10000
embedding_dim = 16
max_length = 99
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000


conversations = []
training_sentences = []
training_labels = []


#getting conversions from json object and converting into list
for item in (datastore["conversations"]):
	for sentence in item:
		sentence = sentence.lower()
		training_sentences.append(sentence)
		conversations.append(sentence)


for item in (datastore["conversations"]):
	flag = 0
	for sentence in item:
		if flag:
			sentence = sentence.lower()
			training_labels.append(sentence)
		flag = 1
	#adding hmmm in list for last statement
	training_labels.append("hmmm")


total_op = len(training_labels)

#creating tokens from the list of words
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_labels)

word_index = tokenizer.word_index
total_words = len(tokenizer.word_index) + 1

# converting conversion list into sequence of tokens and fitting them with padding
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_labels_ = tokenizer.texts_to_sequences(training_labels)
training_label_padded = pad_sequences(training_labels_, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#converting into numpy array
training_padded = np.array(training_padded)
training_labels = np.array(training_label_padded)

#creatting a list of classes index
lable_classes_category = []
i = 0
for label in training_labels:
	lable_classes_category.append(i)
	i += 1


ys = tf.keras.utils.to_categorical(lable_classes_category, num_classes=total_op)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_length))
model.add(Bidirectional(LSTM(150)))
#model.add(GlobalAveragePooling1D())
#model.add(Dense(1024, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(1024, activation='relu'))
model.add(Dense(total_op, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#model.summary()
history = model.fit(training_padded, ys, epochs=100, verbose=1)

model.save('convo_model.h5')