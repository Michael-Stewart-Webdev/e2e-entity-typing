import torch
from torch.utils.data import DataLoader
import sys
import numpy as np
import pickle as pkl


import codecs, jsonlines

from logger import logger



import data_utils as dutils
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset, MentionTypingDataset, tokens_to_wordpieces

from load_config import load_config
cf = load_config()

from train import train_without_loading
from evaluate import evaluate_without_loading

torch.manual_seed(123)
torch.backends.cudnn.deterministic=True

MAX_SENT_LEN = cf.MAX_SENT_LEN

# A class for holding one sentence from the dataset.
# Each object stores the original tokens, original labels, as well as the wordpiece-tokenized versions of those tokens.
# It also stores a map that maps the indexes from the tokens to their position in wordpieces.
# vocab stores info on word_to_ix, wordpiece_to_ix etc, which the Sentence class updates upon creating a sentence.
class Sentence(object):
	def __init__(self, tokens, labels, word_vocab, wordpiece_vocab, build_labels=True):
	
		self.tokens = tokens[:]
		self.labels = labels[:]

		
		self.wordpieces, self.token_idxs_to_wp_idxs = self.get_wordpieces(tokens)

		#if len(self.wordpieces) > MAX_SENT_LEN:
		#	logger.debug("A sentence in the dataset exceeds MAX_SENT_LEN (%d): %s" % (MAX_SENT_LEN, " ".join(self.wordpieces)))

		

		# Pad the wordpieces and wordpiece labels so that the DataLoader interprets them correctly.
		self.pad_wordpieces()

		#print self.wordpieces, "<<"
		
		self.pad_tokens()
		
		self.pad_token_map()


		if build_labels:
			self.mentions = self.get_mentions_vector(self.labels)
			
			self.wordpiece_mentions = self.get_mentions_vector(self.wordpiece_labels)
			self.wordpiece_labels = self.get_wordpiece_labels(self.wordpieces, self.labels, self.token_idxs_to_wp_idxs)
			self.pad_wordpiece_labels()
			self.pad_labels()


		# Add every word and wordpiece in this sentence to the Vocab object.
		for word in self.tokens:
			word_vocab.add_token(word)
		for wordpiece in self.wordpieces:
			wordpiece_vocab.add_token(wordpiece)

		# Generate the token indexes and wordpiece indexes.
		self.token_indexes = [word_vocab.token_to_ix[token] for token in self.tokens]
		self.wordpiece_indexes = [wordpiece_vocab.token_to_ix[wordpiece] for wordpiece in self.wordpieces]




	
	def get_wordpieces(self, orig_tokens):
		#print orig_to_tok_map, "<<<"
		return tokens_to_wordpieces(orig_tokens)

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
			self.token_idxs_to_wp_idxs.append(-1)

	# Retrieve the wordpiece labels, which are the same as their corresponding tokens' labels.
	# This is performed using the token_idxs_to_wp_idxs map.
	def get_wordpiece_labels(self, wordpieces, labels, token_idxs_to_wp_idxs):
		wordpiece_labels = []
		padding_labels = [0] * len(labels[0])
		for i, idx in enumerate(token_idxs_to_wp_idxs):

			#for ix in idx:
			#	wordpiece_labels.append(labels[i])
			if i == len(token_idxs_to_wp_idxs) - 1:
				max_idx = len(wordpieces)
			else:
				max_idx = token_idxs_to_wp_idxs[i + 1]
			for x in range(idx, max_idx):
				wordpiece_labels.append(labels[i])

		return [padding_labels] + wordpiece_labels + [padding_labels] # Add 'padding_labels' for the [CLS] and [SEP] wordpieces

	# Retrieve the mentions vector, a list of 0s and 1s, where 1s represent that the token at that index is an entity.
	def get_mentions_vector(self, labels):
		return [1 if 1 in x else 0 for x in labels]

	# Return the data corresponding to this sentence.
	def data(self):
		return self.tokens, self.labels, self.mentions, self.wordpieces, self.wordpiece_labels, self.wordpiece_mentions, self.token_idxs_to_wp_idxs

	# Returns True when this sentence is valid (i.e. its length is <= MAX_SENT_LEN.)
	def is_valid(self):
		return len(self.wordpieces) == MAX_SENT_LEN and len(self.wordpiece_labels) == MAX_SENT_LEN

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


class Mention(Sentence):
	def __init__(self, tokens, labels, word_vocab, wordpiece_vocab, start, end):
		super(Mention, self).__init__(tokens, labels, word_vocab, wordpiece_vocab, build_labels = False)
		
		# Build left, right, and mention context using start and end
			
		self.labels = labels

		self.token_mention_start = start
		self.token_mention_end = end

		# A mention is out of bounds if it lies outside of max_sent_len.
		#self.mention_oob = False
		#if start > cf.MAX_SENT_LEN or end > cf.MAX_SENT_LEN:
		#	self.mention_oob = True
		#	self.mention_start = 0
		#	self.mention_end = 0			
		#else:

		# Determine the mention start and end with respect to the word pieces, rather than the tokens.
		# If the mention_end is -1, set mention_end to the index of the [SEP] character in the wordpieces (i.e. the last
		# wordpiece in the sentence).
		self.mention_start = self.token_idxs_to_wp_idxs[start]
		self.mention_end = self.token_idxs_to_wp_idxs[end]
		
		if self.mention_end == -1:
			self.mention_end = self.wordpieces.index(u"[SEP]")

		self.wordpiece_vocab = wordpiece_vocab

		self.build_context_wordpieces()
		
		#print ",,,"
		#print self.wordpiece_indexes_left
		#print self.wordpiece_indexes_right
		#print self.wordpiece_indexes_mention
		#print self.wordpiece_indexes_all
		#print "===="
		
		

	def build_context_wordpieces(self):
		maxlen 	 = cf.MODEL_OPTIONS['context_window']
		maxlen_m = cf.MODEL_OPTIONS['mention_window']
	
		start = self.mention_start
		end = self.mention_end

		ctx_left = self.wordpiece_indexes[max(0, start-maxlen):start]	
		ctx_right = self.wordpiece_indexes[end:min(end+maxlen, len(self.wordpiece_indexes))]
		ctx_mention = self.wordpiece_indexes[start:end]	
		ctx_all = ctx_left + ctx_mention + ctx_right
		
		self.wordpiece_indexes_left = ctx_left
		self.wordpiece_indexes_right = ctx_right	
		self.wordpiece_indexes_mention = ctx_mention	
		self.wordpiece_indexes_all = ctx_all
	
		
		self.wordpiece_indexes_left 	 = self.wordpiece_indexes_left[:maxlen] + [0] * (maxlen - len(self.wordpiece_indexes_left))
		self.wordpiece_indexes_right 	 = self.wordpiece_indexes_right[:maxlen] + [0] * (maxlen - len(self.wordpiece_indexes_right))
		self.wordpiece_indexes_mention	 = self.wordpiece_indexes_mention[:maxlen_m] + [0] * (maxlen_m - len(self.wordpiece_indexes_mention))
		self.wordpiece_indexes_all	 = self.wordpiece_indexes_all[:maxlen + maxlen_m + maxlen]	+ [0] * (((maxlen * 2) + maxlen_m) - len(self.wordpiece_indexes_all))

		
		# Truncate token indexes and token map to max_sent_len (purely so that every item is the same length in the data loader.
		# The wordpieces and token map are not used for training, merely for printing during the evaluation.)
		self.token_indexes = self.token_indexes[:cf.MAX_SENT_LEN]
		self.token_idxs_to_wp_idxs = self.token_idxs_to_wp_idxs[:cf.MAX_SENT_LEN]

		

	def is_valid(self):
		maxlen 	 = cf.MODEL_OPTIONS['context_window']
		maxlen_m = cf.MODEL_OPTIONS['mention_window'] 
		return len(self.wordpiece_indexes_left) == maxlen and len(self.wordpiece_indexes_right) == maxlen and (self.mention_end - self.mention_start) <= maxlen_m

	def __repr__(self):
		s =  "Tokens:             %s\n" % self.tokens	
		s += "Wordpieces:         %s\n" % self.wordpieces 
		s += "Mention start/end:  %s-%s\n" % (self.mention_start, self.mention_end)
		s += "Token mention start/end:  %s-%s\n" % (self.token_mention_start, self.token_mention_end)
		s += "Token map:  %s\n" % (self.token_idxs_to_wp_idxs)
		s += "=========================\n"
		s += "Left context:       %s\n" % (self.wordpiece_indexes_left)
		s += "Right context:      %s\n" % (self.wordpiece_indexes_right)
		s += "Mention context:    %s\n" % (self.wordpiece_indexes_mention)
		s += "=========================\n"
		s += "Left context:       %s\n" % ([self.wordpiece_vocab.ix_to_token[t] for t in self.wordpiece_indexes_left])
		s += "Right context:      %s\n" % ([self.wordpiece_vocab.ix_to_token[t] for t in self.wordpiece_indexes_right])
		s += "Mention context:    %s\n" % ([self.wordpiece_vocab.ix_to_token[t] for t in self.wordpiece_indexes_mention])
		s += "=========================\n\n\n"
				
		return s
		

# Builds the hierarchy given a list of file paths (training and test sets, for example).
def build_hierarchy(filepaths):
	# Load the hierarchy
	logger.info("Building category hierarchy.")
	hierarchy = CategoryHierarchy()
	for ds_name, filepath in filepaths.items():
		with jsonlines.open(filepath, "r") as reader:
			for i, line in enumerate(reader):
				for m in line['mentions']:
					labels = set(m['labels'])
					for l in labels:
						hierarchy.add_category(l, "test" if ds_name == "dev" else ds_name) # Treat the dev set as the test set for the purpose of the hierarchy categories
				if type(cf.MAX_SENTS[ds_name]) == int and i >= cf.MAX_SENTS[ds_name]:
					break
	hierarchy.freeze_categories() # Convert the set to a list, and sort it
	logger.info("Category hierarchy contains %d categories." % len(hierarchy))

	return hierarchy





def build_dataset(filepath, hierarchy, word_vocab, wordpiece_vocab, ds_name):
	sentences = []	
	invalid_sentences_count = 0
	total_sents = 0

	mentions = []
	invalid_mentions_count = 0
	total_mentions = 0

	total_wordpieces = 0
	# Generate the Sentences
	with jsonlines.open(filepath, "r") as reader:
		for line in reader:
			tokens = [w for w in line['tokens']]	
			total_sents += 1
			if cf.TASK == "end_to_end":
							
				labels = [[0] * len(hierarchy) for x in range(len(tokens))]
				for m in line['mentions']:
					for i in range(m['start'], m['end']):					
						labels[i] = hierarchy.categories2onehot(m['labels'])
				sent = Sentence(tokens, labels, word_vocab, wordpiece_vocab)
				total_wordpieces += len(sent.wordpieces)
				if sent.is_valid():
					sentences.append(sent)
					#if ds_name == "test":
					#	print sent
				else:
					invalid_sentences_count += 1
				
				
				if type(cf.MAX_SENTS[ds_name]) == int and len(sentences) >= cf.MAX_SENTS[ds_name]:
					break
			elif cf.TASK == "mention_level":
				for m in line['mentions']:
					start = m['start']
					end = m['end']
					labels = hierarchy.categories2onehot(m['labels'])
					mention = Mention(tokens, labels, word_vocab, wordpiece_vocab, start, end)
					mentions.append(mention)
					if not mention.is_valid():						
						invalid_mentions_count += 1
					total_mentions += 1
					total_wordpieces += len(mention.wordpieces)

			print "\r%s" % total_sents,
			if type(cf.MAX_SENTS[ds_name]) == int and len(mentions) >= cf.MAX_SENTS[ds_name]:
				break
				

	# If any sentences are invalid, log a warning message.
	if invalid_sentences_count > 0:
		logger.warn("%d of %d sentences in the %s dataset were not included in the dataset due to exceeding the MAX_SENT_LEN of %s after wordpiece tokenization." % (invalid_sentences_count, total_sents, ds_name, MAX_SENT_LEN))


	if invalid_mentions_count > 0:
			logger.warn("%d of %d mentions in the %s dataset were trimmed due to exceeding the mention_window of %s after wordpiece tokenization." % (invalid_mentions_count, total_mentions, ds_name, cf.MODEL_OPTIONS['mention_window']))

	logger.info("Building data loader...")

	if cf.TASK == "end_to_end":
		# Construct an EntityTypingDataset object.
		data_x, data_y, data_z, data_i, data_tx, data_ty, data_tm,  = [], [], [], [], [], [], []
		for i, sent in enumerate(sentences):
			data_x.append(np.asarray(sent.wordpiece_indexes))
			data_y.append(np.asarray(sent.wordpiece_labels))
			data_z.append(np.asarray(sent.wordpiece_mentions))
			data_i.append(np.asarray(i))
			data_tx.append(np.asarray(sent.token_indexes))
			data_ty.append(np.asarray(sent.labels))
			data_tm.append(np.asarray(sent.token_idxs_to_wp_idxs))
			sys.stdout.write("\r%i / %i" % (i, len(sentences)))
			sys.stdout.flush()
		print ""
		logger.info("Data loader complete.")	

		dataset = EntityTypingDataset(data_x, data_y, data_z, data_i, data_tx, data_ty, data_tm)
		return dataset, total_wordpieces

	
	elif cf.TASK == "mention_level":

		data_xl, data_xr, data_xa, data_xm, data_y = [], [], [], [], []

		for i, mention in enumerate(mentions):
			data_xl.append(np.asarray(mention.wordpiece_indexes_left))
			data_xr.append(np.asarray(mention.wordpiece_indexes_right))
			data_xa.append(np.asarray(mention.wordpiece_indexes_all))
			data_xm.append(np.asarray(mention.wordpiece_indexes_mention))
			data_y.append(np.asarray(mention.labels))
		

			#print len(mention.wordpiece_indexes_left), len(mention.wordpiece_indexes_right), len(mention.wordpiece_indexes_all), len(mention.wordpiece_indexes_mention), len(mention.labels)
			
			

		dataset = MentionTypingDataset(data_xl, data_xr, data_xa, data_xm, data_y)

		return dataset, total_wordpieces





def main(asset_path):
	#cf, logger = load_config()
	
	dataset_filenames = {
		"train": cf.TRAIN_FILENAME,
		"dev": cf.DEV_FILENAME,
		"test": cf.TEST_FILENAME,
	}

	# 1. Construct the Hierarchy by looking through each dataset for unique labels.
	hierarchy = build_hierarchy(dataset_filenames)

	# 2. Construct two empty Vocab objects (one for words, another for wordpieces), which will be populated in step 3.
	word_vocab = Vocab()
	wordpiece_vocab = Vocab()

	logger.info("Hierarchy contains %d categories unique to the test set." % len(hierarchy.get_categories_unique_to_test_dataset()))

	# 3. Build a data loader for each dataset (train, test).
	data_loaders = {}
	for ds_name, filepath in dataset_filenames.items():
		logger.info("Loading %s dataset from %s." % (ds_name, filepath))
		dataset, total_wordpieces = build_dataset(filepath, hierarchy, word_vocab, wordpiece_vocab, ds_name)
		data_loader = DataLoader(dataset, batch_size=cf.BATCH_SIZE, pin_memory=True)
		data_loaders[ds_name] = data_loader
		logger.info("The %s dataset was built successfully." % ds_name)

		logger.info("Dataset contains %i wordpieces (including overly long sentences)." % total_wordpieces)
		if ds_name == "train":
			total_wordpieces_train = total_wordpieces
		# for batch_x, batch_y, batch_z, batch_i, batch_tx, batch_ty, batch_tm,  in data_loader:
		#  	print "x:", batch_x 
		#  	print "y:", batch_y
		#  	print "z:", batch_z

		#  	print "i:", batch_i
		#  	print "tx:", batch_tx 
		#  	print "ty:", batch_ty 
		#  	print "tm:", batch_tm 
		#  	print "==="

	print hierarchy.category_counts['train']


	BYPASS_SAVING = True
	if BYPASS_SAVING:
		#logger.info("Bypassing file saving - training model directly")
		#train_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces_train)
		#return
		logger.info("Evaluating directly")
		evaluate_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces_train)
		return
	

	logger.info("Saving data loaders to file...")

	dutils.save_obj_to_pkl_file(data_loaders, 'data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')

	logger.info("Saving vocabs and hierarchy to file...")
	dutils.save_obj_to_pkl_file(word_vocab, 'word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	dutils.save_obj_to_pkl_file(wordpiece_vocab, 'wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	dutils.save_obj_to_pkl_file(hierarchy, 'hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')

	dutils.save_obj_to_pkl_file(total_wordpieces_train, 'total_wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')

	dutils.save_list_to_file(word_vocab.ix_to_token, 'word vocab', cf.DEBUG_FOLDER + '/word_vocab.txt')
	dutils.save_list_to_file(wordpiece_vocab.ix_to_token, 'wordpiece vocab', cf.DEBUG_FOLDER + '/wordpiece_vocab.txt')


if __name__ == "__main__":
	main(asset_path='models/ontonotes_small_test/asset/')
