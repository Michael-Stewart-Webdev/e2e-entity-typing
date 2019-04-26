import data_utils as dutils
from data_utils import Vocab, CategoryHierarchy, EntityTypingDataset
from bert_serving.client import BertClient
from logger import logger
from model import E2EETModel
import torch.optim as optim
from progress_bar import ProgressBar
import time
import torch
from load_config import load_config
cf = load_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert an entire batch to wordpieces using the vocab object.
def batch_to_wordpieces(batch_x, vocab):
	wordpieces = []
	padding_idx = vocab.token_to_ix["[PAD]"]
	for sent in batch_x:
		wordpieces.append([vocab.ix_to_token[x] for x in sent if x != padding_idx])
	return wordpieces

def wordpieces_to_bert_embs(batch_x, bc):
	return torch.from_numpy(bc.encode(batch_x, is_tokenized=True))

def train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy, epoch_start = 1):

	
	logger.info("Training model.")
	

	bc = BertClient()
	
	

	optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cf.LEARNING_RATE, momentum=0.9)
	model.cuda()

	num_batches = len(data_loaders["train"])
	progress_bar = ProgressBar(num_batches = num_batches, max_epochs = cf.MAX_EPOCHS, logger = logger)
	avg_loss_list = []

	# Train the model

	for epoch in range(epoch_start, cf.MAX_EPOCHS + 1):
		epoch_start_time = time.time()
		epoch_losses = []	

		for (i, (batch_x, batch_y, batch_z, _, _, _, _)) in enumerate(data_loaders["train"]):

			# 1. Convert the batch_x from wordpiece ids into wordpieces
			wordpieces = batch_to_wordpieces(batch_x, wordpiece_vocab)
			
			# 2. Encode the wordpieces into Bert vectors
			bert_embs  = wordpieces_to_bert_embs(wordpieces, bc)

			batch_x = bert_embs.to(device)
			batch_y = batch_y.float().to(device)

			# 3. Feed these Bert vectors to our model
			model.zero_grad()
			model.train()

			loss = model(batch_x, batch_y)

			# 4. Backpropagate
			loss.backward()
			optimizer.step()
			epoch_losses.append(loss)

			# 5. Draw the progress bar
			progress_bar.draw_bar(i, epoch, epoch_start_time)

		avg_loss = sum(epoch_losses) / float(len(epoch_losses))
		avg_loss_list.append(avg_loss)


		progress_bar.draw_completed_epoch(avg_loss, avg_loss_list, epoch, epoch_start_time)

def main():

	data_loaders = dutils.load_obj_from_pkl_file('data loaders', cf.ASSET_FOLDER + '/data_loaders.pkl')
	word_vocab = dutils.load_obj_from_pkl_file('word vocab', cf.ASSET_FOLDER + '/word_vocab.pkl')
	wordpiece_vocab = dutils.load_obj_from_pkl_file('wordpiece vocab', cf.ASSET_FOLDER + '/wordpiece_vocab.pkl')
	hierarchy = dutils.load_obj_from_pkl_file('hierarchy', cf.ASSET_FOLDER + '/hierarchy.pkl')
	
	logger.info("Building model.")
	model = E2EETModel(	embedding_dim = cf.EMBEDDING_DIM,
						hidden_dim = cf.HIDDEN_DIM,
						vocab_size = len(wordpiece_vocab),
						label_size = len(hierarchy))
	model.cuda()

	train(model, data_loaders, word_vocab, wordpiece_vocab, hierarchy)

if __name__ == "__main__":
	main()