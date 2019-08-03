import torch
import torch.nn as nn
#import torch.nn.functional as F
from load_config import device

from sklearn.cluster import KMeans

torch.manual_seed(256)
torch.backends.cudnn.deterministic=True



class MentionLevelModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, model_options, total_wordpieces, category_counts, hierarchy_matrix, context_window, mention_window, attention_type):
		super(MentionLevelModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.layer_1 = nn.Linear(hidden_dim + hidden_dim + hidden_dim, hidden_dim)

		self.dropout = nn.Dropout()

		self.hidden2tag = nn.Linear(hidden_dim, label_size)


		self.use_hierarchy 	 = model_options['use_hierarchy']

	

		self.dropout = nn.Dropout(p=0.5)

		self.hierarchy_matrix = hierarchy_matrix
		self.context_window = context_window
		self.mention_window = mention_window

		self.left_enc = nn.Linear(embedding_dim, hidden_dim)
		self.right_enc = nn.Linear(embedding_dim, hidden_dim)
		self.mention_enc = nn.Linear(embedding_dim, hidden_dim)
				

		self.attention_type = attention_type
		
		if self.attention_type == "dynamic":
			print "Using dynamic attention"
			self.attention_layer = nn.Linear(embedding_dim, 3)
		elif self.attention_type == "scalar":
			self.component_weights = nn.Parameter(torch.ones(3).float())

	
	def forward(self, batch_xl, batch_xr, batch_xa, batch_xm):		

		batch_xl = batch_xl[:, 0:self.context_window, :].mean(1)	# Truncate the weights to the appropriate window length, just in case BERT's max_seq_len exceeds it	
		batch_xr = batch_xr[:, 0:self.context_window, :].mean(1)
		batch_xm = batch_xm[:, 0:self.mention_window, :].mean(1)

		batch_xl = torch.relu(self.left_enc(batch_xl))
		batch_xr = torch.relu(self.right_enc(batch_xr))
		batch_xm = torch.relu(self.mention_enc(batch_xm))
	


		if self.attention_type == "dynamic":		
			# If using 'dynamic attention', multiply the concatenated weights of each component by each attention weight.
			# The attention weights should correspond to the weights from batch_xm, the mention context.
			# The idea is that the network will learn to determine the effectiveness of the left, right, or mention context depending
			# on the mention context (i.e. "Apple" requires the left and right context to predict accurately, whereas "Obama" only requires
			# the mention context.
			attn_weights = torch.softmax(self.attention_layer(batch_xm), dim=1)
			#print attn_weights[0]
			joined = torch.cat((batch_xl, batch_xr, batch_xm), dim=1).view(batch_xm.size()[0], 3, batch_xm.size()[1])
			joined = torch.einsum("ijk,ij->ijk", (joined, attn_weights))
			joined = joined.view(batch_xm.size()[0], batch_xm.size()[1] * 3)
		elif self.attention_type == "scalar":
			component_weights = torch.softmax(self.component_weights, dim=0)
			joined = torch.cat((batch_xl * component_weights[0], batch_xr * component_weights[1],  batch_xm * component_weights[2]), 1)
		elif self.attention_type == "none":
			joined = torch.cat((batch_xl, batch_xr,  batch_xm), 1)
		
		batch_x_out = self.dropout(torch.relu(self.layer_1(joined)))
		y_hat = self.hidden2tag(batch_x_out)	


		if self.use_hierarchy:
			y_hat_size = y_hat.size()
			y_hat_v = y_hat.view(-1, self.hierarchy_matrix.size()[0])
			y_hat =  torch.matmul(y_hat_v, self.hierarchy_matrix)
			y_hat = y_hat.view(y_hat_size)

	
		return y_hat

	def calculate_loss(self, y_hat, batch_y):
		loss_fn = nn.BCEWithLogitsLoss()
		
		return loss_fn(y_hat, batch_y)


	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):	
		hits  = (preds > 0.0).float()
		autohits = 0
		
		nonzeros = set(torch.index_select(hits.nonzero(), dim=1, index=torch.tensor([0]).to(device)).unique().tolist())
		#print hits.nonzero()
		#print "---"
		#print len(nonzeros), len(hits)
		# If any prediction rows are entirely zero, select the class with the highest probability instead.
		if len(nonzeros) != len(hits):		
			am = preds.max(1)[1]
			for i, col_id in enumerate(am):
				if i not in nonzeros:
					hits[i, col_id] = 1.0
					autohits += 1

		#print "Model predicted %d labels using argmax." % autohits
		return hits

	# Evaluate a given batch_x, predicting the labels.
	def evaluate(self, batch_xl, batch_xr, batch_xa, batch_xm):
		preds = self.forward(batch_xl, batch_xr, batch_xa, batch_xm)
		return self.predict_labels(preds)

	


	

class E2EETModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, model_options, total_wordpieces, category_counts, hierarchy_matrix):
		super(E2EETModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.layer_1 = nn.Linear(embedding_dim, hidden_dim)

		self.dropout = nn.Dropout()

		self.hidden2tag = nn.Linear(hidden_dim, label_size)



		self.use_mention_layer = model_options['use_mention_layer']
		self.use_hierarchy 	 = model_options['use_hierarchy']

		print "Use hierarchy:", self.use_hierarchy

		self.dropout = nn.Dropout(p=0.5)

		if self.use_mention_layer:
			self.layer_1_m = nn.Linear(embedding_dim, hidden_dim)
			self.hidden2tag_m = nn.Linear(hidden_dim, 1)

		category_weights = [((total_wordpieces - c * 1.0) / (c)) if c > 0 else 0  for c in category_counts]
		print category_counts
		print category_weights

		self.pos_weights = torch.tensor(category_weights).to(device)
		self.pos_weights.requires_grad = False

		#self.hierarchy_matrix = torch.nn.Parameter(hierarchy_matrix)
		self.hierarchy_matrix = hierarchy_matrix

	def forward(self, batch_x):




		batch_x_out = self.dropout(torch.relu(self.layer_1(batch_x)))
		#batch_x_out = self.layer_1(batch_x)
		y_hat = self.hidden2tag(batch_x_out)


		if self.use_mention_layer:
			batch_x_out_m = torch.relu(self.layer_1_m(batch_x))
			z_hat = torch.sigmoid(self.hidden2tag_m(batch_x_out_m))
			#z_hat = z_hat.view(batch_x.size()[0], batch_x.size()[1])

			y_hat = torch.mul(z_hat, y_hat)

		if self.use_hierarchy:
			y_hat_size = y_hat.size()
			y_hat_v = y_hat.view(-1, self.hierarchy_matrix.size()[0])
			y_hat =  torch.matmul(y_hat_v, self.hierarchy_matrix)
			y_hat = y_hat.view(y_hat_size)


		return y_hat
		#return torch.sigmoid(y_hat)

		# print batch_x.size(), batch_y.size()




		

	def calculate_loss(self, y_hat, batch_x, batch_y, batch_z):

		

		non_padding_indexes = torch.ByteTensor((batch_x > 0))

		loss_fn = nn.BCEWithLogitsLoss()#pos_weight=self.pos_weights)

		loss = loss_fn(y_hat[non_padding_indexes], batch_y[non_padding_indexes])

		#if self.use_mention_layer:
		#	loss_fn = nn.BCELoss()
		#	z_hat = z_hat.view(batch_x.size()[0], batch_x.size()[1])
		#	loss += loss_fn(z_hat[non_padding_indexes], batch_z[non_padding_indexes])

		return loss


	# Predict the labels of a batch of wordpieces using a threshold of 0.5.
	def predict_labels(self, preds):
		#print preds
		#print preds
		#hits = torch.zeros(preds.size())
		#for sent in preds:
		#	if len(sent[sent > 0.1]) > 0:
		#		kmeans = KMeans(n_clusters=2, random_state=0).fit([x for x in sent if x > 0])
		#		print kmeans.labels_, len(kmeans.labels)
				#hits[i] = kmeans.labels_
		#exit()
		hits  = (preds > 0.0).float()
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



