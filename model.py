import torch
import torch.nn as nn
import torch.nn.functional as F
from load_config import device

torch.manual_seed(123)

class E2EETModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size ):
		super(E2EETModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.layer_1 = nn.Linear(embedding_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, label_size)

	def forward(self, batch_x):


		batch_x = F.relu(self.layer_1(batch_x))
		batch_x = F.sigmoid(self.hidden2tag(batch_x))

		return batch_x

		# print batch_x.size(), batch_y.size()

		

	def calculate_loss(self, batch_x, batch_y):
		loss = nn.BCELoss()
		return loss(batch_x, batch_y)



	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):
		hits  = (preds >= 0.5).float()
		return hits

	# Evaluate a given batch_x, predicting the labels.
	def evaluate(self, batch_x):
		preds = self.forward(batch_x)
		return self.predict_labels(preds)



	# Evaluate a given batch_x, but convert the predictions for each wordpiece into the predictions of each token using
	# the token_idxs_to_wp_idxs map.
	def predict_token_labels(self, batch_x, token_idxs_to_wp_idxs):
		preds = self.forward(batch_x)

		avg_preds = torch.zeros(list(batch_x.shape)[0], list(batch_x.shape)[1], list(preds.shape)[2])
		#print avg_preds.size()
		#print batch_x.size()

		for i, batch in enumerate(batch_x):
			#if i > 0:
			#	continue

			for j, wp_idxs in enumerate(token_idxs_to_wp_idxs[i]):
				#print "preds:", j, wp_idxs, preds[i], preds[i].size()
				#print "mean preds:", preds[i][wp_idxs].mean(dim=0)
				avg_preds[i][j] = preds[i][wp_idxs].mean(dim=0)  			# -> preds[[1, 2]]
				#print "avg_preds:", avg_preds[0]
			


		#print "---+++"
		#print avg_preds, len(avg_preds)

		return self.predict_labels(avg_preds)