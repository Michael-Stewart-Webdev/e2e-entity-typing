import jsonlines
import os

DEBUG = False

def parse(folder):
	print(folder)
	print("=" * 50)

	total = 0
	seg_issues = 0

	for f in os.listdir(folder):
		if not f.endswith('.json'):
			continue

		data = []
		data = [line for line in jsonlines.open(os.path.join(folder, f), 'r')]
		problematic_docs = []
		for d in data:
			#print(d)
			mentions = sorted(d['mentions'], key=lambda x: x['start'])

			prev_end = -1
			for i, m in enumerate(mentions):
				start = m['start']
				end = m['end']
				labels = m['labels']
				if i > 0:
					if start == (prev_end) and (set(labels) == set(prev_labels)):						
						problematic_docs.append(d)
						break

				prev_end = end
				prev_labels = labels
				


		print("%s" % f)
		print("\t%d docs total" % len(data))
		print("\t%d docs with segmentation issues" % len(problematic_docs))

		#print()
		if folder == "figer_50k" and DEBUG:		
			for d in problematic_docs:
				print(" ".join(d['tokens']))
				mentions = sorted(d['mentions'], key=lambda x: x['start'])
				for m in mentions:
					print("%s %s %s" % (m['start'], m['end'], m['labels']))
				print()


		print()
		total += len(data)
		seg_issues += len(problematic_docs)
	return total, seg_issues, problematic_docs

def main():
	
	total, seg_issues = 0, 0
	for folder in ["figer_50k", "bbn_modified", "ontonotes_modified"]:
		t, s, ps = parse(folder)
		total += t 
		seg_issues += s 

		print("Total docs: %s" % t)
		print("Total docs with segmentation issues: %s (%.2f)" % (t, s/t*100))

		with open('problematic_%s.txt' % folder, 'w') as f:
			for p in ps:
				#if len(p['tokens']) < 25:
				f.write(str(p['tokens']))
				f.write('\n')
				for m in sorted(p['mentions'], key = lambda x: x['start']):
					f.write(str(m) + '\n')
				f.write('\n')
				f.write(str(p))
				f.write('\n\n')



	print("Total docs: %s" % total)
	print("Total docs with segmentation issues: %s (%.2f)" % (seg_issues, seg_issues/total*100))


if __name__ == "__main__":
	main()
