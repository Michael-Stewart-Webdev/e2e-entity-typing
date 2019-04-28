import torch
import torch.nn as nn
#import torch.nn.functional as F
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


		batch_x = torch.relu(self.layer_1(batch_x))
		y_hat = torch.sigmoid(self.hidden2tag(batch_x))

		return y_hat

		# print batch_x.size(), batch_y.size()

		

	def calculate_loss(self, y_hat, batch_y, batch_x):

		#print "===="
		#print batch_x

		#nonzeros = torch.nonzero(batch_x)
		#indexes = torch.index_select(nonzeros, dim=1, index=torch.tensor([1]))#.view(-1)
		#indexes = torch.unique(indexes)

		#non_padding_indexes = torch.sum((batch_x > 0), 1) - 1
		non_padding_indexes = torch.ByteTensor((batch_x > 0))


		#print batch_y

		#print non_padding_indexes

		#print batch_y[non_padding_indexes]

		#print batch_y[non_padding_indexes].size()

		#print mask
		
		#print torch.unique(nonzeros, False, False, 1)

		#print batch_x[nonzeros]
		#print y_hat.size()
		#exit()
		#print "===="


		loss = nn.BCELoss()
		return loss(y_hat[non_padding_indexes], batch_y[non_padding_indexes])



	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):
		#print preds
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

		for i, batch in enumerate(batch_x):
			for j, wp_idxs in enumerate(token_idxs_to_wp_idxs[i]):
				#print "preds:", j, wp_idxs, preds[i], preds[i].size()
				#print "mean preds:", preds[i][wp_idxs].mean(dim=0)
				avg_preds[i][j] = preds[i][wp_idxs].mean(dim=0)  			# -> preds[[1, 2]]
				#print "avg_preds:", avg_preds[0]

		return self.predict_labels(avg_preds)