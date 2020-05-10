# (python shebang)
"""
	Path to list.
"""
from os import listdir
from os.path import isfile, join

TESTING_PATH = ["images/test_set"]
TESTING_FILE = "cat-dog-test.txt"

TRAINING_PATH = ["images/training_set"]
TRAINING_FILE = "cat-dog-train.txt"

if __name__ == "__main__":
	# Training
	ftest = open(TRAINING_FILE, 'w')
	for path in TRAINING_PATH:
		for file in listdir(path):
			if isfile(join(path, file)):
				ftest.write("training/" + join(path, file) + '\n')
	print("Training done")

	# Testing
	ftest = open(TESTING_FILE, 'w')
	for path in TESTING_PATH:
		for file in listdir(path):
			if isfile(join(path, file)):
				ftest.write("training/" + join(path, file) + '\n')
	print("Testing done")

