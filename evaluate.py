from load_config import load_config, device
cf = load_config()
from logger import logger
import torch, json, sys
from data_utils import batch_to_wordpieces, wordpieces_to_bert_embs, build_token_to_wp_mapping

from sklearn.metrics import f1_score, classification_report

from colorama import Fore, Back, Style

import random




class ModelEvaluator():

	def __init__(self, model, test_loader, word_vocab, wordpiece_vocab, hierarchy, bc):
		self.model = model 
		self.test_loader = test_loader 
		self.word_vocab = word_vocab 
		self.wordpiece_vocab = wordpiece_vocab
		self.hierarchy = hierarchy
		self.bc = bc

		self.best_f1_and_epoch = [0.0, -1]


	# Evaluate a given model via F1 score over the entire test corpus.
	def evaluate_model(self, epoch):

		self.model.zero_grad()
		self.model.eval()


		micro_f1_scores = []
		macro_f1_scores = []

		filtered_micro_f1_scores = []	# f1 scores where the non-entities have been removed prior to calculation of f1 score
		filtered_macro_f1_scores = []

		predictable_micro_f1_scores = []	# f1 scores of labels that appear in the training set
		predictable_macro_f1_scores = []	

		filtered_predictable_micro_f1_scores = []	# f1 scores of labels that appear in the training set, filtered as above
		filtered_predictable_macro_f1_scores = []	
	
		for (i, (batch_x, batch_y, batch_z, _, batch_tx, batch_ty, batch_tm)) in enumerate(self.test_loader):

			# 1. Convert the batch_x from wordpiece ids into wordpieces
			wordpieces = batch_to_wordpieces(batch_x, self.wordpiece_vocab)
			
			# 2. Encode the wordpieces into Bert vectors
			bert_embs  = wordpieces_to_bert_embs(wordpieces, self.bc)

			bert_embs = bert_embs.to(device)
			batch_y = batch_y.float().to(device)

			# 3. Build the token to wordpiece mapping using batch_tm, built during the build_data stage.
			token_idxs_to_wp_idxs = build_token_to_wp_mapping(batch_tm)

			# 4. Retrieve the token predictions for this batch, from the model.
			token_preds = self.model.predict_token_labels(bert_embs, token_idxs_to_wp_idxs)
		
			# 5. Determine the micro and macro f1 scores for each sentence

			# token_preds_r and batch_ty_r are 'reshaped' versions of token_preds and batch_ty.
			# They are essentially 'flattened' versions of token_preds and batch_ty, i.e. a 2d tensor
			# of dim [max_sent_len, hierarchy_size].
			token_preds_r = token_preds.view(token_preds.shape[0] * token_preds.shape[1], -1)
			batch_ty_r    = batch_ty.view(batch_ty.shape[0] * batch_ty.shape[1], -1)

			micro_f1_scores.append(f1_score(batch_ty_r, token_preds_r, average="micro"))
			macro_f1_scores.append(f1_score(batch_ty_r, token_preds_r, average="macro"))

			# Filter out any completely-zero rows in batch_ty, i.e. the words that are not entities
			
			nonzeros = torch.nonzero(batch_ty_r)
			
			if nonzeros.size()[0] > 0:	# Ignore batches that have no entities in them at all			

				indexes = torch.index_select(nonzeros, dim=1, index=torch.tensor([0])).view(-1)
				indexes = torch.unique(indexes)
				f_batch_ty = batch_ty_r[indexes]
				f_token_preds = token_preds_r[indexes]

				# Calculate the micro f1 and macro f1 scores with the filtered rows removed, i.e.
				# only over the mentions
				filtered_micro_f1_scores.append(f1_score(f_batch_ty, f_token_preds, average="micro"))
				filtered_macro_f1_scores.append(f1_score(f_batch_ty, f_token_preds, average="macro"))

				
				# Calculate the micro f1 and macro f1 scores for labels appearing in the training dataset (ignore those unique to the test set).
				p_batch_ty = batch_ty_r[:, self.hierarchy.train_category_ids]
				p_token_preds = token_preds_r[:, self.hierarchy.train_category_ids]	
				
				predictable_micro_f1_scores.append(f1_score(p_batch_ty, p_token_preds, average="micro"))
				predictable_macro_f1_scores.append(f1_score(p_batch_ty, p_token_preds, average="macro"))


				# Calculate the micro f1 and macro f1 scores for labels appearing in the training dataset (ignore those unique to the test set).
				# Filter as above.

				pf_batch_ty 	= f_batch_ty[:, self.hierarchy.train_category_ids]
				pf_token_preds =  f_token_preds[:, self.hierarchy.train_category_ids]
							
				filtered_predictable_micro_f1_scores.append(f1_score(pf_batch_ty, pf_token_preds, average="micro"))
				filtered_predictable_macro_f1_scores.append(f1_score(pf_batch_ty, pf_token_preds, average="macro"))
				

			# For the first batch, print a classification report and a visual demonstration of a tagged sentence.			
			if i == 0:
				logger.info("\n" + classification_report(batch_ty_r, token_preds_r, target_names=self.hierarchy.categories))
				logger.info("\n" + self.get_tagged_sent_example(batch_tx, token_preds, batch_ty))

			print "\rCalculating F1 scores for batch %d/%d..." % (i + 1, len(self.test_loader)),
		print ""

		micro_f1 = sum(micro_f1_scores) / len(micro_f1_scores)
		macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores)

		filtered_micro_f1 = sum(filtered_micro_f1_scores) / len(filtered_micro_f1_scores)
		filtered_macro_f1 = sum(filtered_macro_f1_scores) / len(filtered_macro_f1_scores)

		predictable_micro_f1 = sum(predictable_micro_f1_scores) / len(predictable_micro_f1_scores)
		predictable_macro_f1 = sum(predictable_macro_f1_scores) / len(predictable_macro_f1_scores)

		filtered_predictable_micro_f1 = sum(filtered_predictable_micro_f1_scores) / len(filtered_predictable_micro_f1_scores)
		filtered_predictable_macro_f1 = sum(filtered_predictable_macro_f1_scores) / len(filtered_predictable_macro_f1_scores)

		logger.info("                  Micro F1: %.4f\tMacro F1: %.4f" % (micro_f1, macro_f1))
		logger.info("(Filtered)        Micro F1: %.4f\tMacro F1: %.4f" % (filtered_micro_f1, filtered_macro_f1))
		logger.info("(Predictable)     Micro F1: %.4f\tMacro F1: %.4f" % (predictable_micro_f1, predictable_macro_f1))
		logger.info("(F + Predictable) Micro F1: %.4f\tMacro F1: %.4f" % (filtered_predictable_micro_f1, filtered_predictable_macro_f1))

		return (micro_f1 + macro_f1 + filtered_micro_f1 + filtered_macro_f1 + predictable_micro_f1 + predictable_macro_f1 + filtered_predictable_micro_f1 + filtered_predictable_macro_f1) / 8

	# Get an example tagged sentence, returning it as a string.
	# It resembles the following:
	#
	# word_1		Predicted: /other		Actual: /other
	# word_2		Predicted: /person		Actual: /organization
	# ...
	#
	def get_tagged_sent_example(self, batch_tx, token_preds, batch_ty):

		# 1. Build a list of tagged_sents, in the form of:
		#    [[word, [pred_1, pred_2], [label_1, label_2]], ...]
		tagged_sents = []
		n = random.randint(0, len(token_preds) - 1)	# Pick a random sentence from the batch

		tagged_sent = []			
		for i, token_ix in enumerate(batch_tx[n]):
			if token_ix == 0:
				continue	# Ignore padding tokens

			tagged_sent.append([									\
				self.word_vocab.ix_to_token[token_ix],				\
				self.hierarchy.onehot2categories(batch_ty[n][i]),	\
				self.hierarchy.onehot2categories(token_preds[n][i])	\
			])
		tagged_sents.append(tagged_sent)

		# 2. Convert the tagged_sents to a string, which prints nicely
			
		s = ""		
		for tagged_sent in tagged_sents:
			inside_entity = False
			current_labels = []
			current_preds  = []
			current_words  = []
			for tagged_word in tagged_sent:

				is_entity = len(tagged_word[1]) > 0 or len(tagged_word[2]) > 0		
			
				if (not is_entity and inside_entity) or (is_entity and (len(current_preds) > 0 and tagged_word[1] != current_labels)):	
						s += " ".join(current_words)[:37].ljust(40)					
						s += "Predicted: "

						if len(current_preds) == 0:
							ps = "%s<No predictions>%s" % (Fore.YELLOW, Style.RESET_ALL)
						else:
							ps = ", ".join(["%s%s%s" % (Fore.GREEN if pred in current_labels else Fore.RED, pred, Style.RESET_ALL) for pred in current_preds])
						
						s += ps.ljust(40)
						s += "Actual: "
						if len(current_labels) == 0:
							s += "%s<No labels>%s" % (Fore.YELLOW, Style.RESET_ALL)
						else:
							s += ", ".join(current_labels)
						s += "\n"

						inside_entity = False
						current_labels = []
						current_preds  = []
						current_words  = []

				if is_entity:					
					if not inside_entity:
						inside_entity = True
						current_labels = tagged_word[1]
						current_preds = tagged_word[2]

					current_words.append(tagged_word[0])
			
		return s





	# Save the best model to the best model directory, and save a small json file with some details (epoch, f1 score).
	def save_best_model(self, f1_score, epoch):
		logger.info("Saving model to %s." % cf.BEST_MODEL_FILENAME)
		torch.save(self.model.state_dict(), cf.BEST_MODEL_FILENAME)

		logger.info("Saving model details to %s." % cf.BEST_MODEL_JSON_FILENAME)
		model_details = {
			"epoch": epoch,
			"f1_score": f1_score
		}
		with open(cf.BEST_MODEL_JSON_FILENAME, 'w') as f:
			json.dump(model_details, f)

	# Determine whether the given f1 is better than the best f1 score so far.
	def is_new_best_f1_score(self, f1):
		return f1 > self.best_f1_and_epoch[0]

	# Determine whether there has been no improvement to f1 over the past n epochs.
	def no_improvement_in_n_epochs(self, n, epoch):
		return epoch - self.best_f1_and_epoch[1] >= n

	# Evaluate the model every n epochs.
	def evaluate_every_n_epochs(self, n, epoch):		
		if epoch % n == 0 or epoch == cf.MAX_EPOCHS:
			f1 = self.evaluate_model(epoch)

			if self.is_new_best_f1_score(f1):
				self.best_f1_and_epoch = [f1, epoch]
				logger.info("New best average F1 score achieved!        (%s%.4f%s)" % (Fore.YELLOW, f1, Style.RESET_ALL))
				self.save_best_model(f1, epoch)
			elif self.no_improvement_in_n_epochs(cf.STOP_CONDITION, epoch):#:cf.STOP_CONDITION):
				logger.info("No improvement to F1 score in past %d epochs. Stopping early." % cf.STOP_CONDITION)
				logger.info("Best F1 Score: %.4f" % self.best_f1_and_epoch[0])
				sys.exit(0)
		