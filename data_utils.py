import pickle as pkl 
from torch.utils.data import Dataset
from logger import logger
import torch
import codecs, sys
import numpy as np
from load_config import device


sys.path.append('bert')
import tokenization

tokenizer = tokenization.FullTokenizer(
	vocab_file='./data/bert/vocab.txt', do_lower_case=False)


torch.manual_seed(123)
torch.backends.cudnn.deterministic=True

# Save an object to a pickle file and provide a message when complete.
def save_obj_to_pkl_file(obj, obj_name, fname):
	with open(fname, 'wb') as f:
		pkl.dump(obj, f, protocol=2)
		logger.info("Saved %s to %s." % (obj_name, fname))


# Save a list to a file, with each element on a newline.
def save_list_to_file(ls, ls_name, fname):
	with codecs.open(fname, 'w', 'utf-8') as f:
		f.write("\n".join(ls))
		logger.debug("Saved %s to %s." % (ls_name, fname))		

# Load an object from a pickle file and provide a message when complete.
def load_obj_from_pkl_file(obj_name, fname):
	with open(fname, 'rb') as f:
		obj = pkl.load(f)
		logger.info("Loaded %s from %s." % (obj_name, fname))		
	return obj

# Transforms a list of original tokens into a list of wordpieces using the Bert tokenizer.
# Returns two lists:
# - bert_tokens, the wordpieces corresponding to orig_tokens,
# - orig_to_tok_map, which maps the indexes from orig_tokens to their positions in bert_tokens.
#   for example, if orig_tokens = ["hi", "michael"], and bert_tokens is ["[CLS]", "hi", "mich", "##ael", "[SEP]"],
#	then orig_to_tok_map becomes [1, 2].
def tokens_to_wordpieces(orig_tokens):
	bert_tokens = []
	orig_to_tok_map = []
	bert_tokens.append("[CLS]")
	for orig_token in orig_tokens:
		word_pieces = tokenizer.tokenize(orig_token)
		orig_to_tok_map.append(len(bert_tokens))# + x for x in range(len(word_pieces))])
		bert_tokens.extend(word_pieces)
	bert_tokens.append("[SEP]")
	return bert_tokens, orig_to_tok_map


# An EntityTypingDataset, comprised of multiple Sentences.
class EntityTypingDataset(Dataset):
	def __init__(self, x, y, z, i, tx, ty, tm, ):
		super(EntityTypingDataset, self).__init__()
		self.x = x # wordpiece indexes
		self.y = y # wordpiece labels
		self.z = z # mentions

		self.i = i   # sentence indexes
		self.tx = tx # token indexes
		self.ty = ty # token labels
		self.tm = tm # token to wordpiece map

	def __getitem__(self, ids):
		return self.x[ids], self.y[ids], self.z[ids], self.i[ids], self.tx[ids], self.ty[ids], self.tm[ids]

	def __len__(self):
		return len(self.x)

# An MentionTypingDataset, comprised of multiple Mentions.
class MentionTypingDataset(Dataset):
	def __init__(self, xl, xr, xa, xm, y):
		super(MentionTypingDataset, self).__init__()
		self.xl = xl # context (left)
		self.xr = xr # context (right)
		self.xa = xa # context (all)
		self.xm = xm # context (mention)
		self.y = y

	
	def __getitem__(self, ids):
		return self.xl[ids], self.xr[ids], self.xa[ids], self.xm[ids], self.y[ids]

	def __len__(self):
		return len(self.xl)

# A Class to store the category hierarchy.
class CategoryHierarchy():

	def __init__(self, categories=None):		
		self.categories = set()

		self.categories_segmented = {
			"train": set(),
			"test": set()
		}
		self.hierarchy_matrix = None
		
		self.train_category_ids = []

		self.category_counts = {
			"train": {},
			"test": {}
		}
				

	# Build the hierarchy matrix, e.g.
	# "person/actor":  [1, 1, 0, 0, 0]
	# "person/actor/shakespearian":  [1, 1, 1, 0, 0]
	# "location": [0, 0, 0, 1, 0]
	# "location/body_of_water": [0, 0, 0, 1, 1]
	def build_hierarchy_matrix(self):
		subcat2idx = {}
		hierarchy_matrix = []
		for category in self.categories:
			splitcats = category.split('/')
			parent = splitcats[1]
			subcats = splitcats[1:]
			for i in range(len(subcats)):
				sc = "/".join(subcats[:i+1])
				if sc not in subcat2idx:
					subcat2idx[sc] = len(subcat2idx)
			subcat_idxs = [subcat2idx["/".join(subcats[:i+1])] for i in range(len(subcats))]
			subcat_idxs_onehot = [0] * len(self.categories)
			for subcat_idx in subcat_idxs:
				subcat_idxs_onehot[subcat_idx] = 1
			hierarchy_matrix.append(subcat_idxs_onehot)

		for i, row in enumerate(hierarchy_matrix):
			print(self.categories[i].ljust(40), row[:30])

		print(self.categories)
		return torch.from_numpy(np.asarray(hierarchy_matrix)).float().to(device)

	def get_categories_unique_to_test_dataset(self):
		return [c for c in self.categories_segmented['test'] if c not in self.categories_segmented['train']]

	def get_overlapping_category_ids(self):
		return [self.category2idx[c] for c in self.get_overlapping_categories()]

	def get_overlapping_categories(self):
		train_set = set(self.categories_segmented['train'])
		return sorted([c for c in self.categories_segmented['test'] if c in train_set])

	def add_category(self, category, ds_name):
		if type(self.categories) == tuple:
			raise Exception("Cannot add more categories to the hierarchy after it has been frozen via freeze_categories().")
		self.categories.add(category)
		self.categories_segmented[ds_name].add(category)
		if category in self.category_counts[ds_name]:
			self.category_counts[ds_name][category] += 1
		else:
			self.category_counts[ds_name][category] = 1

	def get_train_category_counts(self):
		return [self.category_counts['train'][category] if category in self.category_counts['train'] else 0 for category in self.categories ]

	# Freeze the hierarchy, converting it to a list and sorting it in alphabetical order.
	def freeze_categories(self):
		self.categories = tuple(sorted(self.categories))
		self.hierarchy_matrix = self.build_hierarchy_matrix()
		self.category2idx = {self.categories[i] : i for i in range(len(self.categories)) }
		self.category_ids = sorted([self.category2idx[c] for c in self.categories])
		self.train_category_ids = sorted([self.category2idx[i] for i in self.categories_segmented['train']])
		self.test_category_ids = sorted([self.category2idx[i] for i in self.categories_segmented['test']])
		logger.info("Hierarchy contains %d categories total. [%d train, %d test]" % (len(self.categories), len(self.categories_segmented['train']), len(self.categories_segmented['test'])))
		self.unique_test_categories = self.get_categories_unique_to_test_dataset()
		logger.info("%d categories are unique to the test dataset." % len(self.unique_test_categories))	
		logger.info("%d categories are present in both train and test datasets." % len(self.get_overlapping_category_ids()))	
		

	def get_category_index(self, category):
		try:
			return self.category2idx[category]
		except KeyError as e:			
			logger.error("Category '%s' does not appear in the hierarchy." % category)

	# Transforms a one-hot vector of categories into a list of categories.
	def onehot2categories(self, onehot):
		return [self.categories[ix] for ix in range(len(onehot)) if onehot[ix] == 1]

	# Transform a list of categories into a one-hot vector, where a 1 represents that category existing in the list.
	def categories2onehot(self, categories):
		categories_onehot = [0] * len(self.categories)
		for category in categories:
			categories_onehot[self.get_category_index(category)] = 1
		return categories_onehot

	def __len__(self):
		return len(self.categories)

	def __repr__(self):
		return "\n".join(["%d: %s" % (i, category) for i, category in enumerate(self.categories)])

	# Retrieve all categories in the hierarchy.
	def get_categories(self):
		if type(self.categories) == set:
			raise Exception("Categories have not yet been frozen and sorted. Please call the freeze_categories() method first.")
		return self.categories

# A class to store the vocabulary, i.e. word_to_ix, wordpiece_to_ix, and their reverses ix_to_word and ix_to_wordpiece.
# Can be used to quickly convert an index of a word/wordpiece to its corresponding term.
class Vocab():
	def __init__(self):
		self.token_to_ix = {}
		self.ix_to_token = []
		self.add_token("[PAD]")

	def add_token(self, token):
		if token not in self.token_to_ix:
			self.token_to_ix[token] = len(self.ix_to_token)
			self.ix_to_token.append(token)

	def __len__(self):
		return len(self.token_to_ix)


# Convert an entire batch to wordpieces using the vocab object.
def batch_to_wordpieces(batch_x, vocab):
	wordpieces = []
	padding_idx = vocab.token_to_ix["[PAD]"]
	for sent in batch_x:
		sent_wordpieces = [vocab.ix_to_token[x] for x in sent if x != padding_idx]
		if sent_wordpieces == []:
			sent_wordpieces = ['[PAD]'] # Allows BERT as a service to still embed it as zeros
		wordpieces.append(sent_wordpieces)
	return wordpieces

def wordpieces_to_bert_embs(batch_x, bc):
	return torch.from_numpy(bc.encode(batch_x, is_tokenized=True))


# Takes a token to wordpiece vector and modifies it as follows:
#   [1, 3, 4] ->
# [ [1, 2], [3], [4] ]
def build_token_to_wp_mapping(batch_tm):
	token_idxs_to_wp_idxs = []
	for row in batch_tm:		
		ls = [i for i in row.tolist() if i >= 0]

		token_idxs_to_wp_idxs.append([None] * len(ls))

		for i, item in enumerate(ls):
					
			if i+1 > len(ls) - 1:
				m = ls[i] + 1
			else:
				m = ls[i+1]	

			token_idxs_to_wp_idxs[-1][i] = range(ls[i], m)

	return token_idxs_to_wp_idxs

# Load embeddings from file for Glove.
# Some code in this function was found at https://github.com/guillaumegenthial/sequence_wtagging
def load_embeddings(emb_model, word_vocab, embedding_dim):
	if emb_model == "glove":
		emb_file = 'glove/glove.6B.300d.txt'
	elif emb_model == "word2vec":
		#emb_file = 'word2vec/GoogleNews-vectors-negative300.txt'
		emb_file = 'word2vec/enwiki_20180420_300d.txt'
	else:
		raise Exception("Embedding model not supported")
		#with np.load(emb_file) as data:
		#	embs = data["embeddings"]

	print("Loading embeddings")
	print(word_vocab.ix_to_token[0])
	embeddings = np.zeros([len(word_vocab), embedding_dim])
	with open(emb_file) as f:
		for line in f:
			line = line.strip().split(' ')
			word = line[0]
			embedding = [float(x) for x in line[1:]]
			if word in word_vocab.token_to_ix:
				word_idx = word_vocab.token_to_ix[word.lower()]
				embeddings[word_idx] = np.asarray(embedding)

	print("Found %d embeddings" % len(embeddings))

	#for i, embedding in enumerate(embeddings):
	#	print(word_vocab.ix_to_token[i], embedding[:5])
	return embeddings
	

