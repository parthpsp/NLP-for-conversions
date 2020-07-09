import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from tensorflow.keras import models

#load jason file
with open("datasets_conversation.json", 'r') as f:
    datastore = json.load(f)

#get model object from local
model = models.load_model('convo_model.h5')

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

testing_sentences = []
testing_labels = []

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



#creating tokens from the list of words
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_labels)

#starting conversion from user
print("please type.........")
user_input = input()
while(user_input!="bye"):
	#converting userinput into sequence of tokens and fitting them with padding
	training_sequences = tokenizer.texts_to_sequences([user_input])
	training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

	#pridicting the classes and getting index value from list of sentences
	predicted = model.predict_classes(training_padded)
	print("=> ", training_labels[predicted[0]])
	user_input = input("=> ")

print("bye, cya!!")
#print(training_padded[0])
#predicted1 = model.predict(training_padded)

#print(predicted)
#print(predicted1)