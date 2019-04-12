from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# save tokens to file, one dialog per line
def save_doc(line, filename):
	data = str(line)+'\n'
	file = open(filename, 'a')
	file.write(data)
	file.close()

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

# load cleaned text sequences
in_filename = 'republic_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# define the number that goes into range function so we can calculate % complete
iterations = 100000

for x in range(iterations):
        # define text
        text = []
        # select a seed text
        seed_text = lines[randint(0,len(lines))]
        # generate new text
        generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
        # combine seed text and generated text
        seed_generated = str(seed_text) + '|' + str(generated)
        # append the final product to the text list
        text.append(seed_generated)
        # save text
        final_text = ' '.join(str(e) for e in text)
        save_doc(final_text, 'final_text.txt')
        if(x % (int(iterations/100)+(iterations%100>0))) == 0:
                print(str(int(round(100*(x/iterations))))+"% complete")
print("100% complete")
