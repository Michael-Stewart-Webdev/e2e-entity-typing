from bert_serving.client import BertClient
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np

sys.path.append('bert')
import tokenization

import codecs, jsonlines

''' logger '''
import logging as logger
import codecs
import sys, torch, os
from datetime import datetime
from colorama import Fore, Back, Style
# logger.basicConfig(format=Fore.CYAN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.DEBUG)
# logger.basicConfig(format=Fore.GREEN + '%(levelname)s: ' + Style.RESET_ALL + '%(message)s', level=logger.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGFILE = None

class LoggingFormatter(logger.Formatter):	

	def format(self, record):
		#compute s according to record.levelno
		#for example, by setting self._fmt
		#according to the levelno, then calling
		#the superclass to do the actual formatting
		
		if LOGFILE:
			message = record.msg.replace(Fore.GREEN, "")
			message = message.replace(Fore.RED, "")
			message = message.replace(Fore.YELLOW, "")
			message = message.replace(Style.RESET_ALL, "")
			LOGFILE.write("%s %s %s\n" % (datetime.now().strftime('%d-%m-%Y %H:%M:%S'), record.levelname.ljust(7), message))

			

		if record.levelno == 10:
			return Fore.CYAN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL,  record.msg)
		elif record.levelno == 20:
			return Fore.GREEN + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg) 
		elif record.levelno == 30:
			return Fore.YELLOW + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg)
		else:
			return Fore.RED + "%s %s%s" % (record.levelname.ljust(5), Style.RESET_ALL, record.msg)
		#return s

hdlr = logger.StreamHandler(sys.stdout)
hdlr.setFormatter(LoggingFormatter())
logger.root.addHandler(hdlr)
logger.root.setLevel(logger.DEBUG)


MAX_SENT_LEN = 50
tokenizer = tokenization.FullTokenizer(
    vocab_file='./data/bert/vocab.txt', do_lower_case=False)


bc = BertClient()

# A class for holding one sentence from the dataset.
# Each object stores the original tokens, original labels, as well as the wordpiece-tokenized versions of those tokens.
# It also stores a map that maps the indexes from the tokens to their position in wordpieces.
# vocab stores info on word_to_ix, wordpiece_to_ix etc, which the Sentence class updates upon creating a sentence.
class Sentence():
	def __init__(self, tokens, labels, vocab):
	
		self.tokens = tokens
		self.labels = labels

		self.mentions = self.get_mentions_vector(self.labels)
		self.wordpieces, self.token_idxs_to_wp_idxs = self.get_wordpieces(tokens)

		if len(self.wordpieces) > MAX_SENT_LEN:
			logger.debug("A sentence in the dataset exceeds MAX_SENT_LEN (%d): %s" % (MAX_SENT_LEN, " ".join(self.wordpieces)))

		self.wordpiece_labels = self.get_wordpiece_labels(self.wordpieces, self.labels, self.token_idxs_to_wp_idxs)

		# Pad the wordpieces and wordpiece labels so that the DataLoader interprets them correctly.
		self.pad_wordpieces()
		self.pad_wordpiece_labels()

		self.pad_tokens()
		self.pad_labels()

		self.pad_token_map()

		self.wordpiece_mentions = self.get_mentions_vector(self.wordpiece_labels)

		# Add every word and wordpiece in this sentence to the Vocab object.
		for word in self.tokens:
			vocab.add_word(word)
		for wordpiece in self.wordpieces:
			vocab.add_wordpiece(wordpiece)

		# Generate the token indexes and wordpiece indexes.
		self.token_indexes = [vocab.word_to_ix[token] for token in self.tokens]
		self.wordpiece_indexes = [vocab.wordpiece_to_ix[wordpiece] for wordpiece in self.wordpieces]


	# Transforms a list of original tokens into a list of wordpieces using the Bert tokenizer.
	# Returns two lists:
	# - bert_tokens, the wordpieces corresponding to orig_tokens,
	# - orig_to_token_map, which maps the indexes from orig_tokens to their positions in bert_tokens.
	#   for example, if orig_tokens = ["hi", "michael"], and bert_tokens is ["[CLS]", "hi", "mich", "##ael", "[SEP]"],
	#	then orig_to_token_map becomes [[1], [2, 3]].
	def get_wordpieces(self, orig_tokens):
		bert_tokens = []
		orig_to_tok_map = []
		bert_tokens.append("[CLS]")
		for orig_token in orig_tokens:
			word_pieces = tokenizer.tokenize(orig_token)
			orig_to_tok_map.append([len(bert_tokens) + x for x in range(len(word_pieces))])
			bert_tokens.extend(word_pieces)
		bert_tokens.append("[SEP]")
		#print orig_to_tok_map, "<<<"
		return bert_tokens, orig_to_tok_map

	# Pad the wordpieces to MAX_SENT_LEN
	def pad_wordpieces(self):
		for x in range(MAX_SENT_LEN - len(self.wordpieces)):
			self.wordpieces.append("[PAD]")

	# Pad the wordpiece_labels to MAX_SENT_LEN
	def pad_wordpiece_labels(self):
		for x in range(MAX_SENT_LEN - len(self.wordpiece_labels)):
			self.wordpiece_labels.append([0] * len(self.wordpiece_labels[0]))

	# Pad the tokens to MAX_SENT_LEN
	def pad_tokens(self):
		for x in range(MAX_SENT_LEN - len(self.tokens)):
			self.tokens.append("[PAD]")

	# Pad the labels to MAX_SENT_LEN
	def pad_labels(self):
		for x in range(MAX_SENT_LEN - len(self.labels)):
			self.labels.append([0] * len(self.labels[0]))

	# Pad the token to wordpiece map to MAX_SENT_LEN
	def pad_token_map(self):
		for x in range(MAX_SENT_LEN - len(self.token_idxs_to_wp_idxs)):
			self.token_idxs_to_wp_idxs.append([-1])

	# Retrieve the wordpiece labels, which are the same as their corresponding tokens' labels.
	# This is performed using the token_idxs_to_wp_idxs map.
	def get_wordpiece_labels(self, wordpieces, labels, token_idxs_to_wp_idxs):
		wordpiece_labels = []
		padding_labels = [0] * len(labels[0])
		for i, idx in enumerate(token_idxs_to_wp_idxs):

			for ix in idx:
				wordpiece_labels.append(labels[i])

			'''

			if i == len(token_idxs_to_wp_idxs) - 1:
				max_idx = len(wordpieces)
			else:
				max_idx = token_idxs_to_wp_idxs[i + 1]
			for x in range(idx, max_idx):
				wordpiece_labels.append(labels[i])
			'''
		#print wordpiece_labels, "<<"

		return [padding_labels] + wordpiece_labels + [padding_labels] # Add 'padding_labels' for the [CLS] and [SEP] wordpieces

	# Retrieve the mentions vector, a list of 0s and 1s, where 1s represent that the token at that index is an entity.
	def get_mentions_vector(self, labels):
		return [1 if 1 in x else 0 for x in labels]

	# Return the data corresponding to this sentence.
	def data(self):
		return self.tokens, self.labels, self.mentions, self.wordpieces, self.wordpiece_labels, self.wordpiece_mentions, self.token_idxs_to_wp_idxs

	# Returns True when this sentence is valid (i.e. its length is <= MAX_SENT_LEN.)
	def is_valid(self):
		return len(self.wordpieces) <= MAX_SENT_LEN

	# Print out a summary of the sentence.
	def __repr__(self):
		s =  "Tokens:             %s\n" % self.tokens
		s += "Labels:             %s\n" % self.labels 
		s += "Mentions:           %s\n" % self.mentions 
		s += "Wordpieces:         %s\n" % self.wordpieces 
		s += "Wordpiece labels:   %s\n" % self.wordpiece_labels 
		s += "Wordpiece mentions: %s\n" % self.wordpiece_mentions
		s += "Token map:          %s\n" % self.token_idxs_to_wp_idxs
		return s

# An EntityTypingDataset, comprised of multiple Sentences.
class EntityTypingDataset(Dataset):
	def __init__(self, x, y, z, i, tx, ty, tm, ):
		super(EntityTypingDataset, self).__init__()
		self.x = x # wordpiece indexes
		self.y = y # wordpiece labels
		self.z = z # mentions

		self.i = i # sentence indexes
		self.tx = tx # token indexes
		self.ty = ty # token labels
		self.tm = tm # token to wordpiece map

	def __getitem__(self, ids):
		return self.x[ids], self.y[ids], self.z[ids], self.i[ids], self.tx[ids], self.ty[ids], self.tm[ids]

	def __len__(self):
		return len(self.x)

# A Class to store the category hierarchy.
# May be initialised with a list, or built and then frozen with the freeze_categories() method.
class CategoryHierarchy():

	def __init__(self, categories=None):		
		if categories:
			self.categories = categories 
			self.freeze_categories()
		else:
			self.categories = set()

	def add_category(self, category):
		if type(self.categories) == tuple:
			raise Exception("Cannot add more categories to the hierarchy after it has been frozen via freeze_categories().")
		self.categories.add(category)

	# Freeze the hierarchy, converting it to a list and sorting it in alphabetical order.
	def freeze_categories(self):
		self.categories = tuple(sorted(self.categories))
		self.category2idx = {self.categories[i] : i for i in range(len(self.categories)) }

	def get_category_index(self, category):
		try:
			return self.category2idx[category]
		except KeyError as e:			
			logger.error("Category '%s' does not appear in the hierarchy." % category)

	# Transform a list of categories into a one-hot vector, where a 1 represents that category existing in the list.
	def categories2onehot(self, categories):
		categories_onehot = [0] * len(self.categories)
		for category in categories:
			categories_onehot[self.get_category_index(category)] = 1
		return categories_onehot

	def __len__(self):
		return len(self.categories)

	# Retrieve all categories in the hierarchy.
	def get_categories(self):
		if type(self.categories) == set:
			raise Exception("Categories have not yet been frozen and sorted. Please call the freeze_categories() method first.")
		return self.categories

# A class to store the vocabulary, i.e. word_to_ix, wordpiece_to_ix, and their reverses ix_to_word and ix_to_wordpiece.
# Can be used to quickly convert an index of a word/wordpiece to its corresponding term.
class Vocab():
	def __init__(self):
		self.word_to_ix = {}
		self.ix_to_word = []
		self.wordpiece_to_ix = {}
		self.ix_to_wordpiece = []
		self.add_word("[PAD]")
		self.add_wordpiece("[PAD]")

	def add_word(self, word):
		if word not in self.word_to_ix:
			self.word_to_ix[word] = len(self.ix_to_word)
			self.ix_to_word.append(word)
			

	def add_wordpiece(self, wordpiece):
		if wordpiece not in self.wordpiece_to_ix:
			self.wordpiece_to_ix[wordpiece] = len(self.ix_to_wordpiece)
			self.ix_to_wordpiece.append(wordpiece)
			

# Builds the hierarchy given a list of file paths (training and test sets, for example).
def build_hierarchy(filepaths):
	# Load the hierarchy
	hierarchy = CategoryHierarchy()
	for filepath in filepaths:
		with jsonlines.open(filepath, "r") as reader:
			for line in reader:
				for m in line['mentions']:
					labels = set(m['labels'])
					for l in labels:
						hierarchy.add_category(l)
	hierarchy.freeze_categories() # Convert the set to a list, and sort it
	logger.info("Category hierarchy contains %d categories." % len(hierarchy))

	return hierarchy

def build_dataset(filepath, hierarchy, vocab, ds_name):
	sentences = []
	invalid_sentences_count = 0
	total_sents = 0
	
	# Generate the Sentences
	with jsonlines.open(filepath, "r") as reader:
		for line in reader:
			tokens = [w for w in line['tokens']]			
			#tokens = tokens + ["[PAD]"] * (MAX_SENT_LEN - len(tokens))
			labels = [[0] * len(hierarchy) for x in range(len(tokens))]
			for m in line['mentions']:
				for i in range(m['start'], m['end']):
					labels[i] = hierarchy.categories2onehot(m['labels'])
			sent = Sentence(tokens, labels, vocab)
			if sent.is_valid():
				sentences.append(sent)
			else:
				invalid_sentences_count += 1
			total_sents += 1

	if invalid_sentences_count > 0:
		logger.warn("%d of %d sentences in the %s dataset were not included in the dataset due to exceeding the MAX_SENT_LEN of %s after wordpiece tokenization." % (invalid_sentences_count, total_sents, ds_name, MAX_SENT_LEN))

	data_x, data_y, data_z, data_i, data_tx, data_ty, data_tm,  = [], [], [], [], [], [], []
	for i, sent in enumerate(sentences):
		data_x.append(np.asarray(sent.wordpiece_indexes))
		data_y.append(np.asarray(sent.wordpiece_labels))
		data_z.append(np.asarray(sent.wordpiece_mentions))
		data_i.append(np.asarray(i))
		data_tx.append(np.asarray(sent.token_indexes))
		data_ty.append(np.asarray(sent.labels))
		data_tm.append(np.asarray(sent.token_idxs_to_wp_idxs))

	dataset = EntityTypingDataset(data_x, data_y, data_z, data_i, data_tx, data_ty, data_tm)
	return dataset, sentences


def main():
	dataset_filepaths = {
		#"train": "data/datasets/ontonotes_small/train.json",
		"test": "data/datasets/ontonotes_small/test.json"
	}

	hierarchy = build_hierarchy(dataset_filepaths.values())
	vocab = Vocab()
	data_loaders = {}
	for ds_name in dataset_filepaths:
		filepath = dataset_filepaths[ds_name]
		logger.info("Loading %s dataset from %s." % (ds_name, filepath))
		dataset, sentences = build_dataset(filepath, hierarchy, vocab, ds_name)
		#for x in dataset:
		#	print x
		#	exit()
		data_loader = DataLoader(dataset, batch_size=3, pin_memory=True)
		data_loaders[ds_name] = data_loader
		logger.info("The %s dataset was built successfully." % ds_name)

		for batch_x, batch_y, batch_z, batch_i, batch_tx, batch_ty, batch_tm,  in data_loader:
		 	print "x:", batch_x 
		 	print "y:", batch_y
		 	print "z:", batch_z

		 	print "i:", batch_i
		 	print "tx:", batch_tx 
		 	print "ty:", batch_ty 
		 	print "tm:", batch_tm 
		 	print "==="
			

	# TODO: Save data loaders, vocab and hierarchy to files

if __name__ == "__main__":
	main()

	# TODO: save data loaders to a pickle file
	# TODO: save vocab and sentences to a pickle file as well? vocab yes, sentences maybe not

	# print "===="


	
	

	#for row in data_iterator:
	#	print row

	# print "---"


#print vocab.word_to_ix
#print vocab.wordpiece_to_ix

'''

x = [["hello"], ["hi", "commisserations"], ["hi"]]
y = [[[0,0,0]], [[0, 0, 0], [1,1,1]], [[0, 0, 0]]]

sentences = []
for xi, yi in zip(x, y):
	sentences.append(Sentence(xi, yi))





'''







''' Build data needs to ...


["Michael", "was", "here"]

[
 ["Michael", ["/person"]],
 ["was", []],
 ["here", []]
]






x = ["[CLS]", Mich", "##ael", "was", "here", "[SEP]"]
y = [ [], [0], [0], [], [], [], [] ]
z = [ 0, 1, 1, 0, 0, 0, 0 ]

(z can be done automatically ? )


how to match wordpieces with mention indexes?

need function to convert x, y back to sentence



d.wordpieces 				= ["[CLS]", Mich", "##ael", "was", "here", "[SEP]"]

d.categories 				= [ [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0] ]
d.mentions   				= [ 0, 1, 1, 0, 0, 0, 0 ]	# [0 if all(x) == 0 for x in self.categories else 1]

d.tokens 			 		= ["Michael", "was", "here"]
d.categories_tokens 		= [ [1, 0, 0], [0, 0, 0], [0, 0, 0]]

d.token_idxs_to_wp_idxs		= [[1, 2], [3], [4]] 

# step 1 -> tokenize
"Michael was here".encode()[1] = ["[CLS]", "Mich", "##ael"...]

# step 2 -> map back to original indexes

self.token_idxs_to_wp_idxs = [ [] ]
current_token_idx = 0
for wp_idx in range(1, len(self.wordpieces)-1):
	wp = self.wordpieces[wp_idx]
	if self.tokens[current_token_idx].startswith(wp):
		self.token_idxs_to_wp_idxs.append( [] )
		self.token_idxs_to_wp_idxs[-1].append(wp_idx)
		current_token_idx += 1 
	if wp.startswith("##"):
		self.token_idxs_to_wp_idxs[-1].append(wp_idx)
		
		



x = d.wordpieces
y = d.categories
z = d.mentions

in eval, iterate over each batch
get model to predict Y given x, i.e. given d.wordpieces above, returns [[0, 0, 0], [1, 0, 0], [1, 0, 0], ... ]

what if pred for ##ael is different from pred for Mich ??

In model.predict_categories(x), it needs to predict per token, not per wordpiece. i.e. needs to return
[1, 0, 0], [0, 0, 0], [0, 0, 0]



def predict_labels(y_hat):
	# y_hat = get top k or sigmoid or whatever
	return y_hat

def evaluate(x):
	y_hat = self.forward(x)
	return self.predict_labels(y_hat)

def evaluate_wps_to_tokens(x, token_idxs_to_wp_idxs)

	y_hat = self.forward(x)

	avg_preds = [None] * len(token_ids_to_wp_ids)
	for wp_idxs in token_idxs_to_wp_idxs:
		avg_preds[-1] = y_hat[wp_idxs].mean(axis=0)  			# -> preds[[1, 2]]

	return self.predict_labels(avg_preds)





# eval.py

for x_wp, y_wp, _, x_tokens, y_tokens, token_idxs_to_wp_idxs in dataloader.get():
	...

	token_preds = model.evaluate_wps_to_tokens(x, token_idxs_to_wp_idxs)

	# compare to y_tokens



WordPieceSentence
	toTokenSentence() -> converts back to token sentence

TokenSentence







'''