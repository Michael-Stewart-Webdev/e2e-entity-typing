import pickle as pkl 
from torch.utils.data import Dataset
from logger import logger
import torch


# Save an object to a pickle file and provide a message when complete.
def save_obj_to_pkl_file(obj, obj_name, fname):
	with open(fname, 'w') as f:
		pkl.dump(obj, f, protocol=2)
		logger.info("Saved %s to %s." % (obj_name, fname))


# Save a list to a file, with each element on a newline.
def save_list_to_file(ls, ls_name, fname):
	with open(fname, 'w') as f:
		f.write("\n".join(ls))
		logger.debug("Saved %s to %s." % (ls_name, fname))		

# Load an object from a pickle file and provide a message when complete.
def load_obj_from_pkl_file(obj_name, fname):
	with open(fname, 'rb') as f:
		obj = pkl.load(f)
		logger.info("Loaded %s from %s." % (obj_name, fname))		
	return obj



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
		wordpieces.append([vocab.ix_to_token[x] for x in sent if x != padding_idx])
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