import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class E2EETModel(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size ):
		super(E2EETModel, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.label_size = label_size

		self.layer_1 = nn.Linear(embedding_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, label_size)

	def forward(self, batch_x, batch_y):


		batch_x = F.relu(self.layer_1(batch_x))
		batch_x = F.sigmoid(self.hidden2tag(batch_x))

		# print batch_x.size(), batch_y.size()

		loss = nn.BCELoss()

		return loss(batch_x, batch_y)