import numpy as np
import math

# count non ternimals
def countNTs(line):
	count = 0
	for char in line:
		if char == "(":
			count += 1

	return count

# count words
def countWords(line):
	count = 0

	split = line.split()
	for s in split:
		if s[0] == "1" or s[0] == "0" or s[0] == "x":
			count += 1

	return count

NUM_CROSSFOLDS = 10

types = ["compound", "neural"]

for t in types:
	
	for i in range(NUM_CROSSFOLDS):

		totalLL = 0
		
		totalDA = 0

		count = 0

		ntCount = 0
		wordCount = 0

		nll = 0

		with open("evalresults_" + t + "_" + str(i) + ".txt", "r") as f:
			for line in f:
				line = line.strip()

				if "loading" not in line and ":" not in line and "nll" not in line:
					ntCount = countNTs(line)
					wordCount = countWords(line)

					# to avoid overflow/underflow problems
					if nll > -700:
						da = math.exp(nll)
						da = math.log(da, wordCount)

						da = 1 + (da / ntCount)

						if da < 0:
							numer = 1 - math.exp(-0.25 * da)
							denom = 1 + math.exp(-0.25 * da)

							da = numer / denom

						totalDA += da
						count += 1

				if "nll" in line:
					split = line.split()
					nll = -1 * float(split[-1])
					totalLL += nll

			print(str(i) + "\t" + t + "\t" + str(totalLL/count) +"\t"+ str(totalDA/count))

					

