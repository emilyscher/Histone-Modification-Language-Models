#import pandas
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import sys

nSize = sys.argv[1]
dataFile = sys.argv[2]

col_names = ["H2AK5ac","H2AS129ph","H3K14ac","H3K18ac","H3K23ac","H3K27ac",
			 "H3K36me","H3K36me2","H3K36me3","H3K4ac","H3K4me","H3K4me2",
			 "H3K4me3","H3K56ac","H3K79me","H3K79me3","H3K9ac","H3S10ph",
			 "H4K12ac","H4K16ac","H4K20me","H4K5ac","H4K8ac","H4R3me",
			 "H4R3me2s","Htz1","prevH2AK5ac","prevH2AS129ph","prevH3K14ac",
			 "prevH3K18ac","prevH3K23ac","prevH3K27ac","prevH3K36me",
			 "prevH3K36me2","prevH3K36me3","prevH3K4ac","prevH3K4me",
			 "prevH3K4me2","prevH3K4me3","prevH3K56ac","prevH3K79me",
			 "prevH3K79me3","prevH3K9ac","prevH3S10ph","prevH4K12ac",
			 "prevH4K16ac","prevH4K20me","prevH4K5ac","prevH4K8ac",
			 "prevH4R3me","prevH4R3me2s","prevHtz1","aCount","tCount","cCount",
			 "gCount","prevACount","prevTCount","prevCCount","prevGCount"]

if(nSize == 3):
	col_names = ["H2AK5ac","H2AS129ph","H3K14ac","H3K18ac","H3K23ac","H3K27ac",
				 "H3K36me","H3K36me2","H3K36me3","H3K4ac","H3K4me","H3K4me2",
				 "H3K4me3","H3K56ac","H3K79me","H3K79me3","H3K9ac","H3S10ph",
				 "H4K12ac","H4K16ac","H4K20me","H4K5ac","H4K8ac","H4R3me",
				 "H4R3me2s","Htz1","prevH2AK5ac","prevH2AS129ph","prevH3K14ac",
				 "prevH3K18ac","prevH3K23ac","prevH3K27ac","prevH3K36me",
				 "prevH3K36me2","prevH3K36me3","prevH3K4ac","prevH3K4me",
				 "prevH3K4me2","prevH3K4me3","prevH3K56ac","prevH3K79me",
				 "prevH3K79me3","prevH3K9ac","prevH3S10ph","prevH4K12ac",
				 "prevH4K16ac","prevH4K20me","prevH4K5ac","prevH4K8ac",
				 "prevH4R3me","prevH4R3me2s","prevHtz1","prevPrevH2AK5ac",
				 "prevPrevH2AS129ph","prevPrevH3K14ac","prevPrevH3K18ac",
				 "prevPrevH3K23ac","prevPrevH3K27ac","prevPrevH3K36me",
				 "prevPrevH3K36me2","prevPrevH3K36me3","prevPrevH3K4ac",
				 "prevPrevH3K4me","prevPrevH3K4me2","prevPrevH3K4me3",
				 "prevPrevH3K56ac","prevPrevH3K79me","prevPrevH3K79me3",
				 "prevPrevH3K9ac","prevPrevH3S10ph","prevPrevH4K12ac",
				 "prevPrevH4K16ac","prevPrevH4K20me","prevPrevH4K5ac",
				 "prevPrevH4K8ac","prevPrevH4R3me","prevPrevH4R3me2s",
				 "prevPrevHtz1","aCount","tCount","cCount","gCount",
				 "prevACount","prevTCount","prevCCount","prevGCount",
				 "prevPrevACount","prevPrevTCount","prevPrevCCount",
				 "prevPrevGCount"]

# load dataset
pima = pd.read_csv(dataFile, header=None, names=col_names)
pima.head()

# train a logistic regression for each histone modification, and test
for i in range(26):

	feature_cols = list(col_names)

	h = feature_cols.pop(i)
	X = pima[feature_cols]

	y = pima[h]

	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

	logreg = LogisticRegression()
	logreg.fit(X_train,y_train)

	y_pred=logreg.predict(X_test)
	y_scores = logreg.predict_proba(X_test)

	cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

	fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
	roc_auc = metrics.auc(fpr, tpr)

	precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_scores[:, 1])

	auc = metrics.auc(recall, precision)

	line = h + "\t" + str(accuracy_score(y_test, y_pred)) + "\t" + str(f1_score(y_test, y_pred, average="macro")) +"\t"+ str(precision_score(y_test, y_pred, average="macro")) + "\t" + str(recall_score(y_test, y_pred, average="macro")) + "\t" + str(roc_auc) + "\t" + str(auc)

	print(line)
