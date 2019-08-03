import data_utils as dutils
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset, batch_to_wordpieces, wordpieces_to_bert_embs
from bert_serving.client import BertClient
from logger import logger
from model import E2EETModel, MentionLevelModel
import torch.optim as optim
from progress_bar import ProgressBar
import time, json
import torch
from load_config import load_config, device
cf = load_config()
from evaluate import ModelEvaluator


torch.manual_seed(123)
torch.backends.cudnn.deterministic=True

# Train the model, evaluating it every 10 epochs.
def train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy, epoch_start = 1):

	logger.info("Training model.")
	
	# Set up a new Bert Client, for encoding the wordpieces
	bc = BertClient()

	modelEvaluator = ModelEvaluator(model, data_loaders['dev'], word_vocab, wordpiece_vocab, hierarchy, bc)
	
	#optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE, momentum=0.9)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE)#, eps=1e-4, amsgrad=True)#, momentum=0.9)
	model.cuda()


	num_batches = len(data_loaders["train"])
	print num_batches
	progress_bar = ProgressBar(num_batches = num_batches, max_epochs = cf.MAX_EPOCHS, logger = logger)
	avg_loss_list = []

	# Train the model

	for epoch in range(epoch_start, cf.MAX_EPOCHS + 1):
		epoch_start_time = time.time()
		epoch_losses = []

		if cf.TASK == "end_to_end":
			for (i, (batch_x, batch_y, batch_z, _, _, _, _)) in enumerate(data_loaders["train"]):
				#if i > 1:
				#	continue
				# 1. Convert the batch_x from wordpiece ids into wordpieces
				wordpieces = batch_to_wordpieces(batch_x, wordpiece_vocab)
			
				# 2. Encode the wordpieces into Bert vectors
				bert_embs  = wordpieces_to_bert_embs(wordpieces, bc)


				bert_embs = bert_embs.to(device)
				batch_y = batch_y.float().to(device)
				batch_z = batch_z.float().to(device)

				# 3. Feed these Bert vectors to our model
				model.zero_grad()
				model.train()

				y_hat = model(bert_embs)

				loss = model.calculate_loss(y_hat, batch_x, batch_y, batch_z)

				# 4. Backpropagate
				loss.backward()
				optimizer.step()
				epoch_losses.append(loss)

				# 5. Draw the progress bar
				progress_bar.draw_bar(i, epoch, epoch_start_time)

		elif cf.TASK == "mention_level":
			for (i, (batch_xl, batch_xr, batch_xa, batch_xm, batch_y)) in enumerate(data_loaders["train"]):

				#torch.cuda.empty_cache()
				#if i > 1:
				#	continue
				# 1. Convert the batch_x from wordpiece ids into wordpieces
				wordpieces_l = batch_to_wordpieces(batch_xl, wordpiece_vocab)
				wordpieces_r = batch_to_wordpieces(batch_xr, wordpiece_vocab)
				#wordpieces_a = batch_to_wordpieces(batch_xa, wordpiece_vocab)
				wordpieces_m = batch_to_wordpieces(batch_xm, wordpiece_vocab)
	
				
				#print len(wordpieces_l[0]), len(wordpieces_r[0]), len(wordpieces_m[0])
				

				#print len(wordpieces_l[0]),  len(wordpieces_r[0]),  len(wordpieces_a[0]) ,  len(wordpieces_m[0])
				

				# 2. Encode the wordpieces into Bert vectors
				bert_embs_l  = wordpieces_to_bert_embs(wordpieces_l, bc).to(device)
				bert_embs_r  = wordpieces_to_bert_embs(wordpieces_r, bc).to(device)				
				#bert_embs_a  = wordpieces_to_bert_embs(wordpieces_a, bc).to(device)
				bert_embs_m  = wordpieces_to_bert_embs(wordpieces_m, bc).to(device)
				
				batch_y = batch_y.float().to(device)				

				# 3. Feed these Bert vectors to our model
				model.zero_grad()
				model.train()

				y_hat = model(bert_embs_l, bert_embs_r, None, bert_embs_m)

				loss = model.calculate_loss(y_hat, batch_y)

				# 4. Backpropagate
				loss.backward()
				optimizer.step()
				epoch_losses.append(loss)

				# 5. Draw the progress bar
				progress_bar.draw_bar(i, epoch, epoch_start_time)

					

				

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)

		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, epoch_start_time)

		#logger.info(avg_loss)

		modelEvaluator.evaluate_every_n_epochs(1, epoch)




	

def create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
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


def train_without_loading(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces):
	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)
	model.cuda()

	logger.info("Loading model weights...")
	model.load_state_dict(torch.load(cf.BEST_MODEL_FILENAME))
	
	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy)

def main():

	logger.info("Loading files...")

	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	total_wordpieces = dutils.load_obj_from_pkl_file('total wordpieces', cf.ASSET_FOLDER + '/total_wordpieces.pkl')
	
	logger.info("Building model.")

	model = create_model(data_loaders, word_vocab, wordpiece_vocab, hierarchy, total_wordpieces)		
	model.cuda()

	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy)

if __name__ == "__main__":
	main()
