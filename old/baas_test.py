from bert_serving.client import BertClient


import codecs, jsonlines

dataset = "data/datasets/ontonotes/train.json"

tagged_sents = []


i = 0

data = []

with jsonlines.open(dataset, "r") as reader:
	for line in reader:
		doc = []
		for i, w in enumerate(line['tokens']):
			doc.append([w, []])

		for m in line['mentions']:
			for i in range(m['start'], m['end']):
				doc[i][1] = m['labels']
		data.append(doc)

		i += 1

		if i == 200:
			break
			#tagged_sent[m['start']] = m['labels'][0]

for d in data[138]:
	print d
exit()


output_dataset = "train.txt"
with codecs.open(output_dataset, 'w', 'utf-8') as f:
	for sent in tagged_sents:
		for word, tag in sent:
			f.write("%s %s\n" % (word, tag))
		f.write("\n")







bc = BertClient()
embs = bc.encode([['First', 'do', 'it'],
               ['then', 'do', 'it', 'right'],
               ['then', 'do', 'it', 'considerably', 'better', 'than', 'before', 'Trump']], is_tokenized=True, show_tokens=True)


print embs[0][0][:6]
print embs[0].shape

print embs[1]

'''
x = [
		['Obama', 'did', 'it'],
		['then', 'do', 'it', 'right']
		...
	]

y = [
	[
		[1, 2, 3],
		[],
		[]
	],
	[
		[],
		[],
		[],
		[]
	]
	...
]
'''