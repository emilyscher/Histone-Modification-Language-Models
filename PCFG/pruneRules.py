
# this scripts ensures that all rules in rules.txt are in the correct format 
# (chompsky normal form) for the supervised PCFG

newRuleCounts = dict()

with open("rules.txt") as f:
	for line in f:
		split = line.split('-->')
		num = int(line.split()[0])
		label = split[0].split()[1]
		split = split[1].split()

		flag = True

		# checking to see if this rule only points to terminals
		for i in split:
			if i[0] != "N" or i[1] != "T":
				flag = False

		if len(split) == 1:
			flag = False

		if flag:
			for x in split[:-1]:
				newRule = label + " --> " + x + " " + label

				if newRule not in newRuleCounts:
					newRuleCounts[newRule] = 0

				newRuleCounts[newRule] = newRuleCounts[newRule] + num

			if len(split) != 0:
				x = split[-1]
				newRule = label + " --> " + x

				if newRule not in newRuleCounts:
					newRuleCounts[newRule] = 0

				newRuleCounts[newRule] = newRuleCounts[newRule] + num
		else:
			print(line.strip())

for key in newRuleCounts:
	print(str(newRuleCounts[key]) + " " + key)