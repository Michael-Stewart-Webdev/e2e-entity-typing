import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(u'bert-base-uncased')

# Tokenized input
sents = [u"First do it", u"then do it right", u"then do it considerably better than before Trump"]
tokenized_text = [tokenizer.tokenize(text) for text in sents]

max_seq_len = max([len(x) for x in tokenized_text])

for sent in tokenized_text:
	if len(sent) < max_seq_len:
		for i in range(max_seq_len - len(sent)):
			sent.append(u"[PAD]")

# Mask a token that we will try to predict back with `BertForMaskedLM`
#tokenized_text[masked_index] = '[MASK]'
#assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = [tokenizer.convert_tokens_to_ids(text) for text in tokenized_text]
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

# Convert inputs to PyTorch tensors

print tokenized_text


model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda')

for sent in indexed_tokens:

	print sent

	tokens_tensor = torch.tensor([sent]).to('cuda')
	segments_tensors = torch.tensor([0 for x in sent]).to('cuda')

	# Predict hidden states features for each layer
	with torch.no_grad():
	    encoded_layers, _ = model(tokens_tensor, segments_tensors)
	# We have a hidden states for each of the 12 layers in model bert-base-uncased
	assert len(encoded_layers) == 12


	print encoded_layers[11], encoded_layers[11].size()