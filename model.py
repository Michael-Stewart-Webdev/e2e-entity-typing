import torch
import torch.nn as nn
#import torch.nn.functional as F
from load_config import device

torch.manual_seed(123)



class E2EETModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, model_options ):
		super(E2EETModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.layer_1 = nn.Linear(embedding_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, label_size)

		self.use_mention_layer = model_options['use_mention_layer']
		self.use_hierarchy 	 = model_options['use_hierarchy']

		if self.use_mention_layer:
			self.layer_1_m = nn.Linear(embedding_dim, hidden_dim)
			self.hidden2tag_m = nn.Linear(hidden_dim, 1)

	def forward(self, batch_x, hierarchy_matrix):



		batch_x_out = torch.relu(self.layer_1(batch_x))
		y_hat = self.hidden2tag(batch_x_out)

		if self.use_mention_layer:
			batch_x_out_m = torch.relu(self.layer_1_m(batch_x))
			z_hat = torch.sigmoid(self.hidden2tag_m(batch_x_out_m))
			#z_hat = z_hat.view(batch_x.size()[0], batch_x.size()[1])

			y_hat = torch.mul(z_hat, y_hat)

		if self.use_hierarchy:
			y_hat_size = y_hat.size()
			y_hat_v = y_hat.view(-1, hierarchy_matrix.size()[0])
			y_hat =  torch.matmul(y_hat_v, hierarchy_matrix)
			y_hat = y_hat.view(y_hat_size)

		return y_hat

		# print batch_x.size(), batch_y.size()

		

	def calculate_loss(self, y_hat, batch_x, batch_y, batch_z):

		

		non_padding_indexes = torch.ByteTensor((batch_x > 0))

		loss_fn = nn.BCEWithLogitsLoss()

		loss = loss_fn(y_hat[non_padding_indexes], batch_y[non_padding_indexes])

		#if self.use_mention_layer:
		#	loss_fn = nn.BCELoss()
		#	z_hat = z_hat.view(batch_x.size()[0], batch_x.size()[1])
		#	loss += loss_fn(z_hat[non_padding_indexes], batch_z[non_padding_indexes])

		return loss


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
	def predict_token_labels(self, batch_x, hierarchy_matrix, token_idxs_to_wp_idxs):
		preds = self.forward(batch_x, hierarchy_matrix)

		avg_preds = torch.zeros(list(batch_x.shape)[0], list(batch_x.shape)[1], list(preds.shape)[2])

		for i, batch in enumerate(batch_x):
			for j, wp_idxs in enumerate(token_idxs_to_wp_idxs[i]):
				#print "preds:", j, wp_idxs, preds[i], preds[i].size()
				#print "mean preds:", preds[i][wp_idxs].mean(dim=0)
				avg_preds[i][j] = preds[i][wp_idxs].mean(dim=0)  			# -> preds[[1, 2]]
				#print "avg_preds:", avg_preds[0]

		return self.predict_labels(avg_preds)