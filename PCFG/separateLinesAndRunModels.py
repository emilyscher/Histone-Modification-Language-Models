import os

'''
This script separates the ptb file into testing/training/validation sequences, 
and then runs the compound and neural pcfg training.
'''

# Yield successive n-sized chunks from lst.
def get_chunks(lst, n, folds):
    chunks = [lst[i * n:(i + 1) * n] for i in range((len(lst) + n - 1) // n )]  

    while(len(chunks) > folds):
        del chunks[-1]

    return chunks

allLines = []

# open the file, get a list of lines
with open("lines.ptb") as f:
    for line in f:
        line = line.strip()
        split = line.split()

        # get rid of junk data
        if len(line) > 10 and len(line) < 5000 and "(GENE-0 )" not in line:
            allLines.append(line)

f.close()

# get NUM_FOLDS number of chunks of lines
NUM_FOLDS = 10
chunks = get_chunks(allLines, round(len(allLines)/NUM_FOLDS) + 1, NUM_FOLDS)

# to make sure they're all similarly sized
for c in chunks:
    print(len(c))

# create the testing/validation/training set files for each crossfold, then train
for test_index in range(NUM_FOLDS):
    valid_index = test_index - 1

    valid_seq = chunks[valid_index]
    test_seq = chunks[test_index]

    training_seq = []
    for i in range(len(chunks)):
        if i != valid_index and i != test_index:
            training_seq.extend(chunks[i])

    f = open("valid.ptb", "w")
    for l in validLines:
        f.write(l+"\n")
    f.close()

    f = open("test.ptb", "w")
    for l in testLines:
        f.write(l+"\n")
    f.close()

    f = open("train.ptb", "w")
    for l in trainLines:
        f.write(l+"\n")
    f.close()

    # the following lines run the preprocessing and training for both the neural and compound PCFGs. 
    # Only run this on a system where you have the proper hardware resources.

    os.system('cp valid.ptb test.ptb train.ptb ./compound-pcfg/parsed/mrg/ws/')
    os.system('python3 preprocess.py --trainfile parsed/mrg/ws/train.ptb --valfile parsed/mrg/ws/valid.ptb --testfile parsed/mrg/ws/test.ptb --outputfile data/ptb --vocabsize 300 --lowercase 1 --replace_num 0')

    model_name = "compound-pcfg-test"+str(test_index)+"-valid"+str(valid_index)+".pt" 
    os.system('python3 train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path '+model_name+' --gpu 0')
    
    model_name = "neural-pcfg-test"+str(test_index)+"-valid"+str(valid_index)+".pt" 
    os.system('python3 train.py --z_dim 0 --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path '+model_name+' --gpu 0')







