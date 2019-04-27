from load_config import load_config, device
cf = load_config()
from logger import logger
import torch, json, sys
from data_utils import batch_to_wordpieces, wordpieces_to_bert_embs, build_token_to_wp_mapping












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


		
		for (i, (batch_x, batch_y, batch_z, _, _, batch_ty, batch_tm)) in enumerate(self.test_loader):

			# 1. Convert the batch_x from wordpiece ids into wordpieces
			wordpieces = batch_to_wordpieces(batch_x, self.wordpiece_vocab)
			
			# 2. Encode the wordpieces into Bert vectors
			bert_embs  = wordpieces_to_bert_embs(wordpieces, self.bc)

			batch_x = bert_embs.to(device)
			batch_y = batch_y.float().to(device)

			token_idxs_to_wp_idxs = build_token_to_wp_mapping(batch_tm)



			token_preds = self.model.predict_token_labels(batch_x, token_idxs_to_wp_idxs)


			if i == 0:
				print batch_ty.float()[0][1]
				print token_preds[0][1]
				print "==="



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
				logger.info("New best F1 score achieved!")
				self.save_best_model(f1, epoch)
			elif self.no_improvement_in_n_epochs(500, epoch):#:cf.STOP_CONDITION):
				logger.info("No improvement to F1 score in past %d epochs. Stopping early." % cf.STOP_CONDITION)
				logger.info("Best F1 Score: %.4f" % self.best_f1_and_epoch[0])
				sys.exit(0)
		