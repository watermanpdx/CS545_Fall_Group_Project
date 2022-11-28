"""
======================================================================
Description:  Data Management - Handle input data and files
Purpose:      Manage input training data (breast cancer data) for
              use in sklearn libraries and matplotlib
              Uses sklearn TSNE library to project data down to
              2 dimensions (PCA) for plotting
Python:       Version 3.9
Authors:      Tayte Waterman
Course:       CS445/545 - Machine Learning
Assignment:   Group project
Date:         11/27/22
======================================================================
"""
#Imports =============================================================
import math
import numpy as np
import random as rd
from sklearn.manifold import TSNE

#Constants ===========================================================
IN_PATH = "input/"
INPUT_FILE = "breast-cancer.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

#Functions/Classes====================================================
class Dataset:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.N = [1.0]*len(X)
        return

    def update(self):
        tX = TSNE(n_components=2, learning_rate="auto",init="pca",
                       perplexity=10).fit_transform(self.X)
        
        self.tX = [[],[]]
        self.tXm = [[],[]]
        self.tXb = [[],[]]
        for i in range(len(self.X)):
            self.tX[0].append(tX[i][0])
            self.tX[1].append(tX[i][1])

            if self.Y[i] == 1:
                self.tXm[0].append(tX[i][0])
                self.tXm[1].append(tX[i][1])
            else:
                self.tXb[0].append(tX[i][0])
                self.tXb[1].append(tX[i][1])
        
        return True

    def normalize(self,N):
        for i in range(len(self.X)):
            for j in range(len(self.X[0])):
                self.X[i,j] = self.X[i,j]/N[j]
        self.update()

        return True
    
class Data:
    def __init__(self,in_file=IN_PATH+INPUT_FILE,
                 train_file=IN_PATH+TRAIN_FILE,
                 test_file=IN_PATH+TEST_FILE,
                 ratio=0.75,restore=True):
        #Constructor
        self.train = None
        self.test = None

        #If restore, restore training data from file
        if restore:
            print("Restoring training and test data from file:")
            self.train = self.load(train_file)
            self.test = self.load(test_file)
            if self.train != None and self.test != None:
                self.update()

        #Else, regenerate data
        if self.train == None or self.test == None:
            print("Generating training and test data from <" + str(in_file) + ">:")
            data = self.load(in_file,raw=True)
            if data != None:
                self.generate(train_file,test_file,data,ratio)

        return

    def update(self,normalize=True):
        if normalize:
            self.normalize()
        else:
            self.train.update()
            self.test.update()
        return True

    def normalize(self):
        N = [None]*len(self.train.X[0])
        for i in range(len(self.train.X)):
            for j in range(len(self.train.X[0])):
                val = abs(self.train.X[i,j])
                if N[j] == None or val > N[j]:
                    N[j] = val
        for i in range(len(self.test.X)):
            for j in range(len(self.test.X[0])):
                val = abs(self.test.X[i,j])
                if N[j] == None or val > N[j]:
                    N[j] = val

        self.train.normalize(N)
        self.test.normalize(N)

        return True

    def generate(self,train_file,test_file,data,ratio=0.75,normalize=True):
        #Split data in to Malignant and Benign samples
        m = [x for x in data if x[1] == "M"]
        b = [x for x in data if x[1] == "B"]

        #Shuffle samples randomly
        rd.shuffle(m)
        rd.shuffle(b)

        #Split malignant samples at ratio threshold
        i = math.floor(len(m)*ratio)
        train = m[:i]
        test = m[i:]

        #Split benign samples at ratio threshold, add
        #  to training and test datasets
        i = math.floor(len(b)*ratio)
        train += b[:i]
        test += b[i:]

        #Randomly shuffle training and test datasets
        rd.shuffle(train)
        rd.shuffle(test)

        #Save results to file
        self.save(train_file,train)
        self.save(test_file,test)

        #Update member data
        self.train = self.load(train_file)
        self.test = self.load(test_file)

        #Update imported data
        self.update(normalize)

        return True

    def save(self,filename,data):
        print("Saving data to file: <" + filename + "> ...")
        try:
            with open(filename,"w") as file:
                file.write(self.header)
                for line in data:
                    line = ",".join(line) +"\n"
                    file.write(line)
                    
                #Close file
                file.close()

        except IOError:
            print(">> Unable to open/write to file: <" + filename + ">")
            return False

        print("\tSuccessfully wrote data to <" + filename + ">")
        return True

    def load(self,filename,raw=False):
        print("Loading data from file: <" + filename + "> ...")
        try:
            with open(filename,"r") as file:
                #Extract header
                self.header = file.readline()

                X = []
                Y = []
                R = []
                line = file.readline()
                while line:
                    line = line[:-1].split(",")
                    x = [float(x) for x in line[2:]]
                    y = 1 if line[1] == "M" else 0
                    R.append(line)
                    X.append(x)
                    Y.append(y)

                    line = file.readline()

                #Close file
                file.close()

                #Transform data and save to object
                data = Dataset(np.array(X),np.array(Y))

        except IOError:
            print(">> Unable to read file: <" + filename + ">")
            return None

        print("\tSuccessfully loaded data from <" + filename + ">")
        if raw:
            return R
        else:
            return data
