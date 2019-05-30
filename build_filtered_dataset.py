from torch.utils.data import DataLoader
import sys
import numpy as np
import pickle as pkl

from bert_serving.client import BertClient


import codecs, jsonlines

from logger import logger

import data_utils as dutils
from data_utils import tokens_to_wordpieces

from load_config import load_config
cf = load_config()

MAX_SENT_LEN = cf.MAX_SENT_LEN
#from gensim.models import KeyedVectors

batch_size = 100

from build_data import build_hierarchy

'''

				'''


def build_vec_file():
	bc = BertClient()

	valid_sent_idxs = set()


	logger.info("Encoding train dataset")

	valid_sents = 0


	np.set_printoptions(suppress=True)

	emb_dict = {}
	
	current_batch = []
	# Pass through the dataset again, this time encoding each valid sentence via BERT.
	with jsonlines.open(cf.TRAIN_FILENAME, "r") as reader:
		for i, line in enumerate(reader):
			#if i > 2000:
			#	continue

			tokens = [w for w in line['tokens']]
			bert_tokens = tokens_to_wordpieces(tokens)[0]

			if len(bert_tokens) < MAX_SENT_LEN:
				
				sent = " ".join(line['tokens'])
				current_batch.append((i, sent))
				#encoded_sent = bc.encode([" ".join(line['tokens'])])[0]
				#emb_dict[i] = encoded_sent
				valid_sents += 1

			if len(current_batch) == batch_size:

				sents = [x[1] for x in current_batch]
				encoded_batch = bc.encode(sents)
				for i, x in enumerate(encoded_batch):
					emb_dict[current_batch[i][0]] = x
				current_batch = []

			sys.stdout.write("\r%d processed, %d valid" % (i, valid_sents))
			sys.stdout.flush()


	print ""
	logger.info("Writing embeddings to file")
	# Generate embedding matrix
	with codecs.open(cf.FILTERED_TRAIN_VEC_FILENAME, "w", "utf-8") as f:
		f.write("%d %d\n" % (len(emb_dict), cf.EMBEDDING_DIM))
		for k in emb_dict:
			f.write("%s %s\n" % (k, " ".join(str(d) for d in emb_dict[k])))
			# np.array_str(emb_dict[k], float('inf'))[3:-3].replace("   ", " ").replace("  ", " ")))

	
	logger.info("Wrote %d embeddings to %s." % (len(emb_dict), cf.FILTERED_TRAIN_VEC_FILENAME))

	logger.info("Built training sent embeddings.")

def build_filtered_dataset(topn=200, max_train_docs_per_test_doc=100, threshold=0.9):
	kv = KeyedVectors.load_word2vec_format(cf.FILTERED_TRAIN_VEC_FILENAME, binary=False)

	bc = BertClient()


	logger.info("Encoding test dataset and retrieving similar training documents")
	filtered_train_idxs = set()

	current_batch = []

	with jsonlines.open(cf.TEST_FILENAME, "r") as reader:
		for i, line in enumerate(reader):
			#if i > 50:
			#	continue
			tokens = [w for w in line['tokens']]
			bert_tokens = tokens_to_wordpieces(tokens)[0]

			if len(bert_tokens) < MAX_SENT_LEN:
				#print " ".join(line['tokens'])
				#encoded_sent = bc.encode([" ".join(line['tokens'])])[0]
				sent = " ".join(line['tokens'])
				current_batch.append((i, sent))
				#print encoded_sent
				#print "hi2"

			if len(current_batch) == batch_size:
				sents = [x[1] for x in current_batch]
				encoded_batch = bc.encode(sents)
				for j, (i, sent) in enumerate(current_batch):

					most_similar = kv.similar_by_vector(encoded_batch[j], topn=topn, restrict_vocab=None)
					most_similar_k = set([int(k[0]) for k in most_similar if k[1] >= threshold])
					num_added = 0
					for k in most_similar_k:
						if num_added > max_train_docs_per_test_doc:
							break
						if k not in filtered_train_idxs:
							filtered_train_idxs.add(k)
							num_added += 1
				current_batch = []

		




	#print i
				#sys.stdout.write("\r%d processed" % (i))
				#sys.stdout.flush()



				#print [k for k in most_similar]
				#print most_similar_k

			
	print ""
	logger.info("Building filtered training set")
	filtered_train_set = []

	with jsonlines.open(cf.TRAIN_FILENAME, "r") as reader:

		for i, line in enumerate(reader):
			if i in filtered_train_idxs:
				filtered_train_set.append(line)
				sys.stdout.write("\r%d processed" % (i))
				sys.stdout.flush()

	print ""
	logger.info("Filtered train set contains %d documents." % len(filtered_train_set))

	with jsonlines.open(cf.FILTERED_TRAIN_FILENAME, 'w') as writer:
		writer.write_all(filtered_train_set)

	logger.info("Wrote filtered train set to %s." % cf.FILTERED_TRAIN_FILENAME)





def build_filtered_hierarchy_dataset():
	test_hierarchy = build_hierarchy({ "test": cf.TEST_FILENAME })

	filtered_train_set = []

	with jsonlines.open(cf.TRAIN_FILENAME, "r") as reader:
		for i, line in enumerate(reader):
			valid_line = True
			for m in line['mentions']:
				for label in m['labels']:					
					if label not in set(test_hierarchy.get_categories()):
						valid_line = False

			if valid_line:				
				filtered_train_set.append(line)
				sys.stdout.write("\r%d processed, %d valid" % (i, len(filtered_train_set)))
				sys.stdout.flush()

	print ""
	logger.info("Filtered train set contains %d documents." % len(filtered_train_set))

	with jsonlines.open(cf.FILTERED_HIERARCHY_TRAIN_FILENAME, 'w') as writer:
		writer.write_all(filtered_train_set)

	logger.info("Wrote filtered train set to %s." % cf.FILTERED_HIERARCHY_TRAIN_FILENAME)


def main():
	#build_vec_file()
	#build_filtered_dataset()
	build_filtered_hierarchy_dataset()
	


if __name__ == "__main__":
	main()
