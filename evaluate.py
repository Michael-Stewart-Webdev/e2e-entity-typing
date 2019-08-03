import data_utils as dutils
from load_config import load_config, device
cf = load_config()
from logger import logger
import torch, json, sys
from data_utils import batch_to_wordpieces, wordpieces_to_bert_embs, build_token_to_wp_mapping

from sklearn.metrics import f1_score, classification_report, accuracy_score

from colorama import Fore, Back, Style

import random

import nfgec_evaluate

torch.manual_seed(123)
torch.backends.cudnn.deterministic=True

class ModelEvaluator():

	def __init__(self, model, data_loader, word_vocab, wordpiece_vocab, hierarchy, bc, mode="train"):
		self.model = model 
		self.data_loader = data_loader 
		self.word_vocab = word_vocab 
		self.wordpiece_vocab = wordpiece_vocab
		self.hierarchy = hierarchy
		self.bc = bc
		self.mode = mode

		self.best_f1_and_epoch = [0.0, -1]


	# Evaluate a given model via F1 score over the entire test corpus.
	def evaluate_model(self, epoch):

		self.model.zero_grad()
		self.model.eval()
		
	
		all_tys   = None
		all_preds = None

		
		true_and_prediction = []
		if cf.TASK == "end_to_end":
			for (i, (batch_x, batch_y, batch_z, _, batch_tx, batch_ty, batch_tm)) in enumerate(self.data_loader):

				# 1. Convert the batch_x from wordpiece ids into wordpieces
				wordpieces = batch_to_wordpieces(batch_x, self.wordpiece_vocab)
			
				# 2. Encode the wordpieces into Bert vectors
				bert_embs  = wordpieces_to_bert_embs(wordpieces, self.bc)

				bert_embs = bert_embs.to(device)
				batch_y = batch_y.float().to(device)

				# 3. Build the token to wordpiece mapping using batch_tm, built during the build_data stage.
				token_idxs_to_wp_idxs = build_token_to_wp_mapping(batch_tm)


				non_padding_indexes = torch.ByteTensor((batch_tx > 0))


				# 4. Retrieve the token predictions for this batch, from the model.
				token_preds = self.model.predict_token_labels(bert_embs, token_idxs_to_wp_idxs)

				#print token_preds, "<TP", len(token_preds)
				token_preds = token_preds[non_padding_indexes]
			
				batch_tx = batch_tx[non_padding_indexes]
				batch_ty = batch_ty[non_padding_indexes]
			

				if all_tys is None:
					all_tys = batch_ty
				else:
					all_tys    = torch.cat((all_tys, batch_ty))

				if all_preds is None:
					all_preds = token_preds
				else:
					all_preds = torch.cat((all_preds, token_preds))

				if i == 0:
					logger.info("\n" + self.get_tagged_sent_example(batch_tx, token_preds, batch_ty))

		elif cf.TASK == "mention_level":
			if self.model.attention_type == "scalar":
				logger.info("Component weights: " + str(self.model.component_weights))
			num_batches = len(self.data_loader)
			for (i, (batch_xl, batch_xr, batch_xa, batch_xm, batch_y)) in enumerate(self.data_loader):
				

				# 1. Convert the batch_x from wordpiece ids into wordpieces
				wordpieces_l = batch_to_wordpieces(batch_xl, self.wordpiece_vocab)
				wordpieces_r = batch_to_wordpieces(batch_xr, self.wordpiece_vocab)
				#wordpieces_a = batch_to_wordpieces(batch_xa, self.wordpiece_vocab)
				wordpieces_m = batch_to_wordpieces(batch_xm, self.wordpiece_vocab)

				# 2. Encode the wordpieces into Bert vectors
				bert_embs_l  = wordpieces_to_bert_embs(wordpieces_l, self.bc).to(device)
				bert_embs_r  = wordpieces_to_bert_embs(wordpieces_r, self.bc).to(device)				
				#bert_embs_a  = wordpieces_to_bert_embs(wordpieces_a, self.bc).to(device)
				bert_embs_m  = wordpieces_to_bert_embs(wordpieces_m, self.bc).to(device)
								
				mention_preds = self.model.evaluate(bert_embs_l, bert_embs_r, None, bert_embs_m)

				batch_y = batch_y.float().to(device)

				for j, row in enumerate(batch_y):

					labels = self.hierarchy.onehot2categories(batch_y[j])
					preds = self.hierarchy.onehot2categories(mention_preds[j])

					

					true_and_prediction.append((labels, preds))

				sys.stdout.write("\rEvaluating batch %d / %d" % (i, num_batches))

				#if all_tys is None:
				#	all_tys = batch_y
				#else:
				#	
				#	all_tys    = torch.cat((all_tys, batch_y))
				#
				#if all_preds is None:
				#	all_preds = mention_preds
				#else:
				
				#	all_preds = torch.cat((all_preds, mention_preds))
		
			

		
		# Convert all one-hot to categories

		def build_true_and_preds(tys, preds):
			true_and_prediction = []
			empty = 0
			for i, row in enumerate(tys):	
				true_cats = self.hierarchy.onehot2categories(tys[i])		
				pred_cats = self.hierarchy.onehot2categories(preds[i])
				#if pred_cats == []:
				#	empty += 1
				true_and_prediction.append((true_cats, pred_cats))	
			#if empty > 0:
			#	logger.warn("There were %d empty predictions." % empty)
			return true_and_prediction	
	


		#all_tys = all_tys.cpu()
		#all_preds = all_preds.cpu()

		#acc = accuracy_score(all_tys, all_preds)
		#micro_f1 = f1_score(all_tys, all_preds, average="micro")
		#macro_f1 = f1_score(all_tys, all_preds, average="macro")

		#logger.info("                  Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (micro_f1, macro_f1, acc))

		if cf.TASK == "end_to_end":

			print all_tys.size()
			# Filter out any completely-zero rows in batch_ty, i.e. the words that are not entities
			nonzeros = torch.nonzero(all_tys)
			indexes = torch.index_select(nonzeros, dim=1, index=torch.tensor([0])).view(-1)
			indexes = torch.unique(indexes)
			filtered_tys = all_tys[indexes]
			filtered_preds = all_preds[indexes]

			filtered_acc = accuracy_score(filtered_tys, filtered_preds)
			filtered_micro_f1 = f1_score(filtered_tys, filtered_preds, average="micro")
			filtered_macro_f1 = f1_score(filtered_tys, filtered_preds, average="macro")

			# Predictable: only considers labels that appear in the test hierarchy. A category is not 'predictable' if it only appears in the training hierarchy.
			overlapping_category_ids = self.hierarchy.get_overlapping_category_ids()

			predictable_tys = all_tys[:, overlapping_category_ids]
			predictable_preds = all_preds[:, overlapping_category_ids]

			predictable_acc = accuracy_score(predictable_tys, predictable_preds)
			predictable_micro_f1 = f1_score(predictable_tys, predictable_preds, average="micro")
			predictable_macro_f1 = f1_score(predictable_tys, predictable_preds, average="macro")

			# Filtered + Predictable: Combines Filter + Predictable, i.e. entities only, and categories that appear in the training hierarchy
			filtered_predictable_tys = filtered_tys[:, overlapping_category_ids]
			filtered_predictable_preds = filtered_preds[:, overlapping_category_ids]

			filtered_predictable_acc = accuracy_score(filtered_predictable_tys, filtered_predictable_preds)
			filtered_predictable_micro_f1 = f1_score(filtered_predictable_tys, filtered_predictable_preds, average="micro")
			filtered_predictable_macro_f1 = f1_score(filtered_predictable_tys, filtered_predictable_preds, average="macro")
		

			logger.info("Classification report (all):")
			logger.info("\n" + classification_report(all_tys, all_preds, target_names=self.hierarchy.categories))

			logger.info("Classification report (filtered, test categories only):")
			logger.info("\n" + classification_report(predictable_tys, predictable_preds, target_names=self.hierarchy.get_overlapping_categories()))
		
		
			logger.info("(Filtered)        Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (filtered_micro_f1, filtered_macro_f1, filtered_acc))
			logger.info("(Predictable)     Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (predictable_micro_f1, predictable_macro_f1, predictable_acc))
			logger.info("(F + Predictable) Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (filtered_predictable_micro_f1, filtered_predictable_macro_f1, filtered_predictable_acc))

		
		


			logger.info("\nUsing NFGEC:")
			nfgec_default  			= build_true_and_preds(all_tys, all_preds)
			nfgec_filtered 			= build_true_and_preds(filtered_tys, filtered_preds)	
			nfgec_predictable 		= build_true_and_preds(predictable_tys, predictable_preds)
			nfgec_filtered_predictable	 = build_true_and_preds(filtered_predictable_tys, filtered_predictable_preds)
	
			logger.info("                  Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (nfgec_evaluate.loose_micro(nfgec_default)[2], nfgec_evaluate.loose_macro(nfgec_default)[2], nfgec_evaluate.strict(nfgec_default)[2]))	
			logger.info("(Filtered)        Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (nfgec_evaluate.loose_micro(nfgec_filtered)[2], nfgec_evaluate.loose_macro(nfgec_filtered)[2], nfgec_evaluate.strict(nfgec_filtered)[2]))		
			logger.info("(Predictable)     Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (nfgec_evaluate.loose_micro(nfgec_predictable)[2], nfgec_evaluate.loose_macro(nfgec_predictable)[2], nfgec_evaluate.strict(nfgec_predictable)[2]))	
			logger.info("(F + Predictable) Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (nfgec_evaluate.loose_micro(nfgec_filtered_predictable)[2], nfgec_evaluate.loose_macro(nfgec_filtered_predictable)[2], nfgec_evaluate.strict(nfgec_filtered_predictable)[2]))	



			return (micro_f1 + macro_f1 + filtered_micro_f1 + filtered_macro_f1 + predictable_micro_f1 + predictable_macro_f1 + filtered_predictable_micro_f1 + filtered_predictable_macro_f1) / 8

		elif cf.TASK == "mention_level":
			print ""
			print len(true_and_prediction)
			#nfgec_default  			= build_true_and_preds(all_tys, all_preds)
			micro, macro, acc = nfgec_evaluate.loose_micro(true_and_prediction)[2], nfgec_evaluate.loose_macro(true_and_prediction)[2], nfgec_evaluate.strict(true_and_prediction)[2]
			logger.info("                  Micro F1: %.4f\tMacro F1: %.4f\tAcc: %.4f" % (micro, macro, acc))
			return (acc + macro + micro) / 3

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
		for i, token_ix in enumerate(batch_tx):
			if token_ix == 0:
				continue	# Ignore padding tokens

			tagged_sent.append([									\
				self.word_vocab.ix_to_token[token_ix],				\
				self.hierarchy.onehot2categories(batch_ty[i]),	\
				self.hierarchy.onehot2categories(token_preds[i])	\
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
			elif epoch > 5 and self.no_improvement_in_n_epochs(cf.STOP_CONDITION, epoch):#:cf.STOP_CONDITION):
				logger.info("No improvement to F1 score in past %d epochs. Stopping early." % cf.STOP_CONDITION)
				logger.info("Best F1 Score: %.4f" % self.best_f1_and_epoch[0])

				main()
				exit()
			if epoch == cf.MAX_EPOCHS:
				logger.info("Training complete.")
				main()
				exit()
				
def create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
	from model import E2EETModel, MentionLevelModel
	if cf.TASK == "end_to_end":
		model = E2EETModel(	embedding_dim = cf.EMBEDDING_DIM,
							hidden_dim = cf.HIDDEN_DIM,
							vocab_size = len(wordpiece_vocab),
							label_size = len(hierarchy),
							model_options = cf.MODEL_OPTIONS,
							total_wordpieces = total_wordpieces,
							category_counts = hierarchy.get_train_category_counts(),
							hierarchy_matrix = hierarchy.hierarchy_matrix)

	elif cf.TASK == "mention_level":
		model = MentionLevelModel(	embedding_dim = cf.EMBEDDING_DIM,
							hidden_dim = cf.HIDDEN_DIM,
							vocab_size = len(wordpiece_vocab),
							label_size = len(hierarchy),
							model_options = cf.MODEL_OPTIONS,
							total_wordpieces = total_wordpieces,
							category_counts = hierarchy.get_train_category_counts(),
							hierarchy_matrix = hierarchy.hierarchy_matrix,
							context_window = cf.MODEL_OPTIONS['context_window'],
							mention_window = cf.MODEL_OPTIONS['mention_window'],
							attention_type = cf.MODEL_OPTIONS['attention_type'],
							use_context_encoders = cf.MODEL_OPTIONS['use_context_encoders'])
	return model

def evaluate_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
	from model import E2EETModel, MentionLevelModel
	from bert_serving.client import BertClient
	import jsonlines
	
	bc = BertClient()

	logger.info("Loading files...")

	
	
	logger.info("Building model.")
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)	
	model.cuda()

	model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))

	modelEvaluator = ModelEvaluator(model, data_loaders['test'], word_vocab, wordpiece_vocab, hierarchy, bc, mode="test")
	
	with jsonlines.open(cf.BEST_MODEL_JSON_FILENAME, "r") as reader:
		for line in reader:
			f1_score, epoch = line['f1_score'], line['epoch']

	modelEvaluator.evaluate_model(epoch)	

def main():

	from model import E2EETModel, MentionLevelModel
	from bert_serving.client import BertClient
	import jsonlines
	
	bc = BertClient()

	logger.info("Loading files...")

	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')
	
	logger.info("Building model.")
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)	
	model.cuda()

	model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))

	modelEvaluator = ModelEvaluator(model, data_loaders['test'], word_vocab, wordpiece_vocab, hierarchy, bc, mode="test")
	
	with jsonlines.open(cf.BEST_MODEL_JSON_FILENAME, "r") as reader:
		for line in reader:
			f1_score, epoch = line['f1_score'], line['epoch']

	modelEvaluator.evaluate_model(epoch)
	

if __name__ == "__main__":
	main()
		
