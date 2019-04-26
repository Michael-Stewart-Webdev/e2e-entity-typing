
#print vocab.word_to_ix
#print vocab.wordpiece_to_ix

'''

x = [["hello"], ["hi", "commisserations"], ["hi"]]
y = [[[0,0,0]], [[0, 0, 0], [1,1,1]], [[0, 0, 0]]]

sentences = []
for xi, yi in zip(x, y):
	sentences.append(Sentence(xi, yi))





'''







''' Build data needs to ...


["Michael", "was", "here"]

[
 ["Michael", ["/person"]],
 ["was", []],
 ["here", []]
]






x = ["[CLS]", Mich", "##ael", "was", "here", "[SEP]"]
y = [ [], [0], [0], [], [], [], [] ]
z = [ 0, 1, 1, 0, 0, 0, 0 ]

(z can be done automatically ? )


how to match wordpieces with mention indexes?

need function to convert x, y back to sentence



d.wordpieces 				= ["[CLS]", Mich", "##ael", "was", "here", "[SEP]"]

d.categories 				= [ [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0] ]
d.mentions   				= [ 0, 1, 1, 0, 0, 0, 0 ]	# [0 if all(x) == 0 for x in self.categories else 1]

d.tokens 			 		= ["Michael", "was", "here"]
d.categories_tokens 		= [ [1, 0, 0], [0, 0, 0], [0, 0, 0]]

d.token_idxs_to_wp_idxs		= [[1, 2], [3], [4]] 

# step 1 -> tokenize
"Michael was here".encode()[1] = ["[CLS]", "Mich", "##ael"...]

# step 2 -> map back to original indexes

self.token_idxs_to_wp_idxs = [ [] ]
current_token_idx = 0
for wp_idx in range(1, len(self.wordpieces)-1):
	wp = self.wordpieces[wp_idx]
	if self.tokens[current_token_idx].startswith(wp):
		self.token_idxs_to_wp_idxs.append( [] )
		self.token_idxs_to_wp_idxs[-1].append(wp_idx)
		current_token_idx += 1 
	if wp.startswith("##"):
		self.token_idxs_to_wp_idxs[-1].append(wp_idx)
		
		



x = d.wordpieces
y = d.categories
z = d.mentions

in eval, iterate over each batch
get model to predict Y given x, i.e. given d.wordpieces above, returns [[0, 0, 0], [1, 0, 0], [1, 0, 0], ... ]

what if pred for ##ael is different from pred for Mich ??

In model.predict_categories(x), it needs to predict per token, not per wordpiece. i.e. needs to return
[1, 0, 0], [0, 0, 0], [0, 0, 0]



def predict_labels(y_hat):
	# y_hat = get top k or sigmoid or whatever
	return y_hat

def evaluate(x):
	y_hat = self.forward(x)
	return self.predict_labels(y_hat)

def evaluate_wps_to_tokens(x, token_idxs_to_wp_idxs)

	y_hat = self.forward(x)

	avg_preds = [None] * len(token_ids_to_wp_ids)
	for wp_idxs in token_idxs_to_wp_idxs:
		avg_preds[-1] = y_hat[wp_idxs].mean(axis=0)  			# -> preds[[1, 2]]

	return self.predict_labels(avg_preds)





# eval.py

for x_wp, y_wp, _, x_tokens, y_tokens, token_idxs_to_wp_idxs in dataloader.get():
	...

	token_preds = model.evaluate_wps_to_tokens(x, token_idxs_to_wp_idxs)

	# compare to y_tokens



WordPieceSentence
	toTokenSentence() -> converts back to token sentence

TokenSentence







'''