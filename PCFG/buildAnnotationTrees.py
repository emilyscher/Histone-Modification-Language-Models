from graphviz import Digraph

chromToDNASeq = dict()

startFlag = False
seq = ""
chrom = ""

# gather each chromosome's sequence
with open("/Users/emilyscher/workspace/yeast_annotation/GCF_000001405.39_GRCh38.p13_genomic.fna","r") as f:
    for line in f:
        line = line.strip()

        if ">" in line:
            if startFlag and len(chrom) > 0:
                chromToDNASeq[chrom] = seq
            seq = ""
            startFlag = False
            chrom = ""

        if "scaffold" not in line and "Primary Assembly" in line:
            startFlag = True
            split = line.split("chromosome ")
            chromNumber = split[1].split()[0]
            chrom = "chr" + chromNumber[:-1]
            seq = ""

        elif startFlag:
            seq = seq + line.upper()

    if startFlag and len(chrom) > 0:
        chromToDNASeq[chrom] = seq


f.close()

# to make sure that the chromosome names match between all files
def getCorrectChromName(chrom):
	if chrom == "chrI":
		return "chr1"
	elif chrom == "chrII":
		return "chr2"
	elif chrom == "chrIII":
		return "chr3"
	elif chrom == "chrIV":
		return "chr4"
	elif chrom == "chrV":
		return "chr5"
	elif chrom == "chrVI":
		return "chr6"
	elif chrom == "chrVII":
		return "chr7"
	elif chrom == "chrVIII":
		return "chr8"
	elif chrom == "chrIX":
		return "chr9"
	elif chrom == "chrX":
		return "chr10"
	elif chrom == "chrXI":
		return "chr11"
	elif chrom == "chrXII":
		return "chr12"
	elif chrom == "chrXIII":
		return "chr13"
	elif chrom == "chrXIV":
		return "chr14"
	elif chrom == "chrXV":
		return "chr15"
	elif chrom == "chrXVI":
		return "chr16"
	else:
		return chrom

# get the gc content of a sequence
def getGCContent(chrom, start, stop):
		seq = chromToDNASeq[chrom][int(start)-1:int(stop)-1]

		if len(seq) == 0:
			return 0

		a = 0
		t = 0
		c = 0
		g = 0

		for char in seq:
			if char == "A":
				a = a + 1
			if char == "T":
				t = t + 1
			if char == "C":
				c = c + 1
			if char == "G":
				g = g + 1


		return round((g+c) * 100 / len(seq))

# small class for storing nucleosome vector objects
class Nucleosome:
	def __init__(self, data, chrom, start, stop):
		self.data = data
		self.start = start
		self.stop = stop
		self.chrom = getCorrectChromName(chrom)

	def getGCContent(self):
		return getGCContent(self.chrom, self.start, self.stop)

# small class for storing annotation objects
class Annotation:
	def __init__(self, label, chrom, start, stop):
		self.label = label
		self.start = start
		self.stop = stop

		self.parents = []
		self.children = []
		self.nucleosomes = []

		self.chrom = getCorrectChromName(chrom)

	def getId(self):
		return self.label + "_" + self.chrom + "_" + str(self.start) + "_" + str(self.stop)

	def name(self):
		return self.label + "_" + str(self.start) + ":" + str(self.stop)

	def getGCContent(self):
		return getGCContent(self.chrom, self.start, self.stop)

annotations = dict()

# create annotation objects for each annotation in the gff file
with open("gencode.v33.annotation.gff3") as f:
	for line in f:
		if "##FASTA" in line:
			break
		if line[0] != "#" and line[0] != ">":
			line = line.strip()

			split = line.split()

			# only interested in chromosomes 1, 2, and 3 for this experiment, just to limit the amount of data
			if split[0] == "chr1" or split[0] == "chrI" or split[0] == "chr2" or split[0] == "chrII" or split[0] == "chr3" or split[0] == "chrIII":
				annot = Annotation(split[2], split[0], int(split[3]), int(split[4]))
				
				if annot.chrom not in annotations:
					annotations[annot.chrom] = list()

				if annot.label == "gene":
					annot.start = annot.start - 500
					annot.stop = annot.stop + 500
					if annot.start < 0:
						annot.start = 0

				if annot.label != "chromosome":
					annotations[annot.chrom].append(annot)


f.close()

labels = dict()

# counting to see how many of each annotation type there are in the data set
for chrom in annotations.keys():
	for i in annotations[chrom]:
		if i.label not in labels:
			labels[i.label] = 0
		labels[i.label] = labels[i.label] + 1

nucleosomes = dict()
histoneCounts = dict()

# get nucleosome vectors, assign them to chromosome dict
with open("allHumanNucVectors.txt") as f:
	for line in f:
		# checking to make sure its not the header line or end line
		if "H2AK5ac" not in line and len(line) > 1:
			line = line.strip()

			split = line.split("\t")

			if len(split) == 4:
				center = int(split[2])
				start = center - 74
				stop = center + 74

				chrom = split[1]

				nuc = Nucleosome(split[-1], chrom, start, stop)

				for index in range(len(split[-1])):
					if index not in histoneCounts:
						histoneCounts[index] = 0

					if split[-1][index] == "1":
						histoneCounts[index] = histoneCounts[index] + 1
				
				if chrom not in nucleosomes:
					nucleosomes[chrom] = list()
				nucleosomes[chrom].append(nuc)

f.close()

# utility method to see if an annotation overlaps with another based on coords
def overlapping(nuc, annot):
	if annot.start <= nuc.start and annot.stop >= nuc.stop:
		return True
	if nuc.start <= annot.start and nuc.stop >= annot.stop:
		return True
	if nuc.start <= annot.start and nuc.stop >= annot.start:
		return True
	if annot.start <= nuc.start and annot.stop >= nuc.start:
		return True


	return False

# utility method to see if an annotation is contained in another based on coords
def contains(child, parent):
	if parent.start <= child.start and parent.stop >= child.stop:
		return True

	return False

geneParts = dict()
count = 0


# get each annotations parent, if it has one
for chrom in annotations.keys():

	geneParts[chrom] = []

	for child in annotations[chrom]:
		count = count + 1
		for parent in annotations[chrom]:
			if child != parent and contains(child, parent):
				child.parents.append(parent)
				parent.children.append(child)

		flag = True

		if flag:
			geneParts[child.chrom].append(child)

again = True

# pruning trees to get rid of excess connections
while again:
	for chrom in geneParts.keys():
		for annot in geneParts[chrom]:
			for child in annot.children:
				for c in annot.children:
					if c != child:
						if contains(c, child):
							annot.children.remove(c)
							c.parents.remove(annot)

	again = False
	for chrom in geneParts.keys():
		for annot in geneParts[chrom]:
			for child in annot.children:
				for c in annot.children:
					if c != child:
						if contains(c, child):
							again = True



# get tops of the trees
startNodes = []

# find annotations which have no parents and are genes, these are the root 
# nodes of our eventual sentence trees
for chrom in geneParts.keys():
	for annot in geneParts[chrom]:
		if len(annot.parents) == 0 and annot.label == "gene":
			startNodes.append(annot)

# returns if a particular node is contained by another
def nodeContainedInSubTree(parent, node):
	children = []

	if len(parent.children) == 0 and parent != node:
		return False

	for child in parent.children:
		if child == node:
			return True
		else:
			return nodeContainedInSubTree(child, node)

	return False

treeBottom = dict()

# link nodes together with endges to build a graph of annotations
def add_nodes_and_edges(graph, annot):
	if len(annot.children) == 0:
		if annot.chrom not in treeBottom:
			treeBottom[annot.chrom] = []

		treeBottom[annot.chrom].append(annot)
	else:
		annot.children.sort(key=lambda x: x.start, reverse=False)

		for child in annot.children:
			if child.getId() not in addedNodes:
				graph.node(child.getId(), child.name())
				graph.edge(annot.getId(), child.getId())
				addedNodes.append(child.getId())
				add_nodes_and_edges(graph, child)

addedNodes = []
nucAlphabet = set()
ruleList = list()

sentences = open("sentences.txt", "w")
rules = open("rules.txt", "w")
ptbfile = open("trees.ptb", "w")

# recursive method for generating ptb formatted lines
def generate_ptb(annot):

	flag = False

	for child in annot.children:
		if child.getId() not in addedNodes:
			flag = True

	if not flag:
		nucs = " "

		for nuc in nucleosomes[annot.chrom]:
			if overlapping(nuc, annot):
				annot.nucleosomes.append(nuc)
				nucAlphabet.add(nuc.data)

		annot.nucleosomes.sort(key=lambda x: x.data, reverse=False)

		if len(annot.nucleosomes) == 0:
			nucs = "(NTUNKNOWN x)"

		for n in annot.nucleosomes:
			nucs = nucs + "(NT" + n.data + "-" + str(n.getGCContent()) + " " + n.data + ")"
		return "(" + annot.label.upper()  + "-" + str(annot.getGCContent()) +" "+ nucs + ")"

	retString = "(" + annot.label.upper() + "-" + str(annot.getGCContent()) + " "

	annot.children.sort(key=lambda x: x.start, reverse=False)

	for child in annot.children:
		if child.getId() not in addedNodes:
			addedNodes.append(child.getId())
			retString = retString + generate_ptb(child)

	retString = retString + ")"

	return retString


# recursive method for generating sentences
def generate_sentence(annot):
	flag = False

	for child in annot.children:
		if child.getId() not in addedNodes:
			flag = True

	if not flag:
		nucs = ""

		for nuc in nucleosomes[annot.chrom]:
			if overlapping(nuc, annot):
				annot.nucleosomes.append(nuc)
				nucAlphabet.add(nuc.data)

		annot.nucleosomes.sort(key=lambda x: x.data, reverse=False)

		if len(annot.nucleosomes) == 0:
			nucs = " x"

		for n in annot.nucleosomes:
			nucs = nucs + " " + n.data
		return nucs

	annot.children.sort(key=lambda x: x.start, reverse=False)

	retString = ""

	for child in annot.children:
		if child.getId() not in addedNodes:
			addedNodes.append(child.getId())
			retString = retString + generate_sentence(child)

	return retString

# recursive method for generating rules
def generate_rules(annot):

	flag = False

	for child in annot.children:
		if child.getId() not in addedNodes:
			flag = True

	if not flag:
		nucs = ""

		for nuc in nucleosomes[annot.chrom]:
			if overlapping(nuc, annot):
				annot.nucleosomes.append(nuc)
				nucAlphabet.add(nuc.data)

		annot.nucleosomes.sort(key=lambda x: x.data, reverse=False)

		if len(annot.nucleosomes) == 0:
			ruleList.append("NTUNKNOWN --> x")
			nucs = "NTUNKNOWN"

		for n in annot.nucleosomes:
			ruleList.append("NT" + n.data + " --> " + n.data)
			nucs = nucs + " NT" + n.data


		ruleList.append(annot.label.upper() + " --> " + nucs)
		return

	annot.children.sort(key=lambda x: x.start, reverse=False)

	allChildren = ""

	for child in annot.children:

		if child.getId() not in addedNodes:
			allChildren = allChildren + " " + child.label.upper()
			addedNodes.append(child.getId())
			generate_rules(child)

	ruleList.append(annot.label.upper() + " -->" + allChildren)

	return 


# generate list of rules and sentences
for n in startNodes:
	sentence = ""

	if len(n.children) == 0:
		nucs = ""
		for nuc in nucleosomes[n.chrom]:
			if overlapping(nuc, annot):
				n.nucleosomes.append(nuc)
				nucAlphabet.add(nuc.data)

		n.nucleosomes.sort(key=lambda x: x.data, reverse=False)

		nucList = ""

		if len(n.nucleosomes) == 0:
			nucs = " x"
			ruleList.append("NTUNKNOWN --> x")
			nucList = "NTUNKNOWN"

		for nuc in n.nucleosomes:
			nucs = nucs + " " + nuc.data
			ruleList.append("NT" + n.data + " --> " + n.data)
			nucList = nucList + " NT" + n.data

		sentence = sentence + nucs

		ruleList.append(n.label.upper() + " --> " + nucList)

	allChildren = ""
	for child in n.children:
		if child.getId() not in addedNodes:
			allChildren = allChildren + " " + child.label.upper()

			addedNodes.append(child.getId())
			sentence = sentence + generate_sentence(child)
			generate_rules(child)

	ruleList.append(n.label.upper() + " -->" + allChildren)

	if len(sentence) > 0:
		sentences.write(sentence[1:] + "\n")


sentences.close()

ruleDict = dict()

# count occureence of rules
for r in ruleList:
	if r not in ruleDict:
		ruleDict[r] = 0

	ruleDict[r] = ruleDict[r] + 1

# write to rules file
for key in ruleDict:
	rules.write(str(ruleDict[key]) + " " + key + "\n")

rules.close()

# clear addedNodes from its previous life generating rule/sentence files
addedNodes = []

# generate ptb file of trees
for n in startNodes:
	ptb = "(" + n.label.upper() + "-" + str(n.getGCContent()) + " "

	if len(n.children) == 0:
		nucs = " "
		for nuc in nucleosomes[n.chrom]:
			if overlapping(nuc, annot):
				n.nucleosomes.append(nuc)
				nucAlphabet.add(nuc.data)

		n.nucleosomes.sort(key=lambda x: x.data, reverse=False)

		if len(n.nucleosomes) == 0:
			nucs = "(NTUNKNOWN x)"

		for nuc in n.nucleosomes:
			nucs = nucs + "(NT" + nuc.data + "-" + str(nuc.getGCContent()) + " " + nuc.data + ")"

		ptb = ptb + nucs


	for child in n.children:
		if child.getId() not in addedNodes:
			addedNodes.append(child.getId())
			ptb = ptb + generate_ptb(child)

	ptb = ptb + ")"
	ptbfile.write(ptb + "\n")

