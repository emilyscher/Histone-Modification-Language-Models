import matplotlib.pyplot as plt
import numpy as np

types = ["neural", "compound"]

neuralStats = list()
compoundStats = list()

plt.style.use('grayscale')

class Stat:
  def __init__(self, epoch, nll, corpusF1, sentF1):
      self.epoch = epoch
      self.nll = float(nll)
      self.corpusF1 = float(corpusF1)
      self.sentF1 = float(sentF1)

def printFig(name, nlls, corpusF1s, sentF1s):

	# x axis is the # of steps
	x_axis = range(len(nlls))
	# y axis is the log likelihood of the training sequence at each step
	y_axis = nlls

	x_axis = np.asarray(x_axis)
	y_axis = np.asarray(y_axis)

	# plot
	plt.plot(x_axis, y_axis)
	plt.xlabel('Epoch Number')
	plt.ylabel('Validation Log Likelihood')


	# for running locally
	plt.savefig("./" + name + "_nll.eps", bbox_inches = "tight")

	# start new plot
	plt.clf()


	# x axis is the # of steps
	x_axis = range(len(nlls))
	# y axis is the log likelihood of the training sequence at each step
	y_axis = corpusF1s

	x_axis = np.asarray(x_axis)
	y_axis = np.asarray(y_axis)

	# plot
	plt.plot(x_axis, y_axis)
	plt.xlabel('Epoch Number')
	plt.ylabel('Validation Corpus F1')

	# for running locally
	plt.savefig("./" + name + "_corpusF1.eps",bbox_inches = "tight")

	# start new plot
	plt.clf()


	# x axis if the # of steps
	x_axis = range(len(nlls))
	# y axis is the log likelihood of the training sequence at each step
	y_axis = sentF1s

	x_axis = np.asarray(x_axis)
	y_axis = np.asarray(y_axis)

	# plot
	plt.plot(x_axis, y_axis)
	plt.xlabel('Epoch Number')
	plt.ylabel('Validation Sentence F1')

	# for running locally
	plt.savefig("./" + name + "_sentenceF1.eps",bbox_inches = "tight")

	# start new plot
	plt.clf()

for t in types:
	for i in range(10):
	
		print(t + " " + str(i))
		with open(t + "_crossfold"+str(i)+".txt") as f:

			epoch = ""

			ppl = ""
			kl = ""
			pplBound = ""
			corpusF1 =""
			sentF1 = ""
			nll = ""

			for line in f:
				line = line.strip()
				if "Starting epoch" in line:
					epoch = line[-1]
				if "ReconPPL" in line and "Epoch" not in line:
					split = line.split()

					ppl = split[1][:-1]
					kl = split[3][:-1]
					pplBound = split[-1]

				if "Corpus F1" in line and "Epoch" not in line:
					split = line.split()

					corpusF1 = split[2][:-1]
					sentF1 = split[-1]

				if "total_nll" in line:
					split = line.split()

					nll = split[-1]

					stat = Stat(epoch, nll, corpusF1, sentF1)

					if t == "neural":
						neuralStats.append(stat)
					else:
						compoundStats.append(stat)

		f.close()

		print("\n\n")


avgNlls = dict()
avgCorpusF1s = dict()
avgSentF1s = dict()

for s in neuralStats:
	epoch = s.epoch

	if epoch not in avgNlls:
		avgNlls[epoch] = 0
		avgCorpusF1s[epoch] = 0
		avgSentF1s[epoch] = 0


	avgNlls[epoch] = avgNlls[epoch] + -1 * s.nll
	avgCorpusF1s[epoch] = avgCorpusF1s[epoch] + s.corpusF1
	avgSentF1s[epoch] = avgSentF1s[epoch] + s.sentF1


nlls = list()
corpusF1s = list()
sentF1s = list()

for i in range(10):
	i = str(i)
	nlls.append(avgNlls[i]/10)
	corpusF1s.append(avgCorpusF1s[i]/1000)
	sentF1s.append(avgSentF1s[i]/1000)

printFig("neural", nlls, corpusF1s, sentF1s)

avgNlls = dict()
avgCorpusF1s = dict()
avgSentF1s = dict()

for s in compoundStats:
	epoch = s.epoch

	if epoch not in avgNlls:
		avgNlls[epoch] = 0
		avgCorpusF1s[epoch] = 0
		avgSentF1s[epoch] = 0


	avgNlls[epoch] = avgNlls[epoch] + -1 * s.nll
	avgCorpusF1s[epoch] = avgCorpusF1s[epoch] + s.corpusF1
	avgSentF1s[epoch] = avgSentF1s[epoch] + s.sentF1


nlls = list()
corpusF1s = list()
sentF1s = list()

for i in range(10):
	i = str(i)
	nlls.append(avgNlls[i]/10)
	corpusF1s.append(avgCorpusF1s[i]/1000)
	sentF1s.append(avgSentF1s[i]/1000)

printFig("compound", nlls, corpusF1s, sentF1s)
