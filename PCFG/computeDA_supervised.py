import numpy as np
import math

# the supervised output files require more processing than the compound/neural PCFGs

def countNTs(line):
	count = 0
	for char in line:
		if char == "(":
			count += 1

	return count

def countWords(line):
	count = 0

	split = line.split()
	for s in split:
		if s[0] == "1" or s[0] == "0" or s[0] == "x":
			count += 1

	return count


sentenceCount = 0
totalDA = 0
totalLL = 0
count = 0

ntCount = 0
wordCount = 0

prob = 0

allLines = list()

def get_chunks(lst, n, folds):
    """Yield successive n-sized chunks from lst."""
    chunks = [lst[i * n:(i + 1) * n] for i in range((len(lst) + n - 1) // n )]  

    while(len(chunks) > folds):
        del chunks[-1]

    return chunks

NUMBER_OF_FOLDS = 10

with open("supervised_parsing.txt", "r") as f:
	for line in f:
		line = line.strip()
		allLines.append(line)

chunks = get_chunks(allLines, int(len(allLines)/NUMBER_OF_FOLDS) + 1, 10)

for i in range(NUMBER_OF_FOLDS):
	totalDA = 0
	totalLL = 0
	count = 0
	for line in chunks[i]:
		
		if "prob (not log!)" in line:
			split = line.split()
			prob = float(split[-1])

			totalLL += math.log(prob)

		else:
			ntCount = countNTs(line)
			wordCount = countWords(line)

			if (wordCount != 1):

				da = prob
				da = math.log(da, wordCount)
				da = 1 + (da / ntCount)

				if da < 0:
					numer = 1 - math.exp(-0.25 * da)
					denom = 1 + math.exp(-0.25 * da)

					da = numer / denom

				totalDA += da
				count += 1

	avgNLL = totalLL / count
	print(str(i) + "\tsupervised\t" + str(avgNLL) +"\t"+ str(totalDA/count))


