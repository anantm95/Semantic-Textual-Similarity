import codecs
import random

pairs_file = codecs.open("../data/new_data/all-pairs.txt", "r")
gs_file = codecs.open("../data/new_data/all-gs.txt", "r")

outfile = codecs.open("../data/new_data/all-data.txt", "w")
shufflefile = codecs.open("../data/new_data/all-data-shuffled.txt", "w")

pair_lines = pairs_file.readlines()
gs_lines = gs_file.readlines()

for i in range(len(pair_lines)):
	sentences = pair_lines[i].split("\t")
	score = gs_lines[i].strip()
	if (len(sentences) < 2 or score == ""):
		continue
	else:
		sent1 = sentences[0].strip()
		sent2 = sentences[1].strip()
		outfile.write(sent1 + "\t" + sent2 + "\t" + score + "\n")


# randomly shuffle lines to remove any possible patterns

lines = open("../data/new_data/all-data.txt").readlines()
random.shuffle(lines)

for line in lines:
	shufflefile.write(line)

# ----------------------------------------------------------------------

# use shuffled data to divide into train, validation and test (a total of 15115 pairs)

infile = codecs.open("../data/new_data/all-data-shuffled.txt", "r")
trainfile = codecs.open("../data/new_data/en-train.txt", "w")
valfile = codecs.open("../data/new_data/en-val.txt", "w")

lines = infile.readlines()

for i in range(13365):
	trainfile.write(lines[i])

for i in range(13365, 14865):
	valfile.write(lines[i])