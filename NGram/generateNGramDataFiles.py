from __future__ import division
import sys
import os
from os import listdir
import math

# only valid n sizes for this script are 2 and 3!
nSize = sys.argv[1]

# small class to hold nucleosome vector objects
class Nucleosome:
  def __init__(self, num, chromosome, center):
    self.num = num
    self.chromosome = chromosome
    self.center = center

  def setVector(self, vector):
    self.vector = vector

nucleosomes = dict()
centersToNucs = dict()
chromosomes = set()
chromsToSeqs = dict()

# gather chromosome sequences
with open("yeastgenome.fsa") as f:
	seq = ""
	chrom = ""
	for line in f:
		if ">" in line:
			if len(seq) != 0:
				chromsToSeqs[chrom] = seq
				seq = ""
			line = line.strip()
			chrom = line[4:]
		else:
			line = line.strip()
			seq = seq + line
f.close()

# get nucleosome centers
with open("mmc3.csv","r") as f:
	for line in f:
		if "nuc_id" not in line:
			data = line.split(",")

			n = Nucleosome(data[0], data[1], data[2])
			nucleosomes[data[0]] = n
			centersToNucs[data[2]] = n

			chromosomes.add(data[1])

f.close()

# create a histone modificaiton vector
def createVector(line):

	data = line.split(",")
	data.pop(0)

	vector = ""
	count = 0
	tot = 0

	histoneCount = 0
	index = 0

	# 156 because of the length of the file lines
	while index < 156:
		mod = data[index]
		index = index + 1

		if count < 5:
			if mod != "NaN":
				tot = tot + float(mod)
				
			count = count + 1
		else:
			if mod != "NaN":
				tot = tot + float(mod)

			count = 0
			average = tot / 6

			# threshold of 1
			if average >= 1:
				vector = vector + "1,"
			else:
				vector = vector + "0,"

			tot = 0

			histoneCount = histoneCount + 1

	return vector

headers = ""

with open("molcel_5341_mmc4.csv","r") as f:
	for line in f:
		if "H2AK5ac" not in line and "nuc_id" not in line:
			vector = createVector(line)
			nucid = line.split(",")[0]

			nucleosomes[nucid].setVector(vector)

		# this means its the header line
		elif "H2AK5ac" in line:
			titles = line.split(",")

			tempTitles = set()

			tempHeaders = []

			for t in titles:
				if t not in tempTitles:
					tempTitles.add(t)
					tempHeaders.append(t)

			tempHeaders = tempHeaders[1:27]

			headers = ",".join(tempHeaders)

			# generating headers

			for h in tempHeaders:
				headers = headers + ",prev" + h

			if(nSize == 3):
				for h in tempHeaders:
					headers = headers + ",prevPrev" + h

f.close()

chromsToNucs = dict()

# assign chroms to nucleosomes
for chrom in chromosomes:
	chromsToNucs[chrom] = []
	for key in nucleosomes:
		if nucleosomes[key].chromosome == chrom:
			chromsToNucs[chrom].append(int(nucleosomes[key].center))

	chromsToNucs[chrom].sort()

allVectors = set()
	
# get base counts for a nucleosome
def getNucStats(chrom, center):

	chromSeq = chromsToSeqs[chrom]

	start = center - 74
	end = center + 74

	seq = chromSeq[start:end]

	aCount = 0
	tCount = 0
	cCount = 0
	gCount = 0

	for c in seq:
		if c == "A" or c =="a":
			aCount = aCount + 1
		elif c == "T" or c == "t":
			tCount = tCount + 1
		elif c =="C" or c =="c":
			cCount = cCount + 1
		elif c == "G" or c == "g":
			gCount = gCount + 1

	return str(aCount) + "," + str(tCount) + "," + str(cCount) + "," + str(gCount)

# add nucleotide counts to headers
if(nSize == 3):
	headers = headers + ",aCount,tCount,cCount,gCount,prevACount,prevTCount,prevCCount,prevGCount,prevPrevACount,prevPrevTCount,prevPrevCCount,prevPrevGCount"
else:
	headers = headers + ",aCount,tCount,cCount,gCount,prevACount,prevTCount,prevCCount,prevGCount"

# generate data lines
for chrom in chromosomes:
	prevPrevMarks = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
	prevPrevNucs = "0,0,0,0"

	prevMarks = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
	prevNucs = "0,0,0,0"

	for center in chromsToNucs[chrom]:
		marks = centersToNucs[str(center)].vector
		nucs = getNucStats(chrom, center)

		if(nSize == 2):
			print(marks + prevMarks + nucs + "," + prevNucs)
		elif(nSize ==3):
			print(marks + prevMarks + prevPrevMarks + nucs + "," + prevNucs + "," + prevPrevNucs)

		prevPrevMarks = prevMarks
		prevPrevNucs = prevNucs

		prevMarks = marks
		prevNucs = nucs

		allVectors.add(centersToNucs[str(center)].vector)

