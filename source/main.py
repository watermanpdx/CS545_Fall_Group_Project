"""
======================================================================
Description:  Main - main script (tbd)
Purpose:      (tbd) under construction
Python:       Version 3.9
Authors:      Tayte Waterman
Course:       CS445/545 - Machine Learning
Assignment:   Group project
Date:         11/27/22
======================================================================
"""
#Imports =============================================================
from data_management import Data
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#Constants ===========================================================

#Functions/Classes====================================================

#Main ----------------------------------------------------------------
def main():
    #Load training/test data
    data = Data()

    #Train and compute results - Random Forest Classifier
    model = RandomForestClassifier().fit(data.train.X,data.train.Y)
    Y = model.predict(data.test.X)
    accuracy = 100*model.score(data.test.X,data.test.Y)

    #Separate data into Malignent and Benign predicted datasets
    M = [[],[]]
    B = [[],[]]
    for i in range(len(Y)):
        if Y[i] == 1:
            M[0].append(data.test.tX[0][i])
            M[1].append(data.test.tX[1][i])
        else:
            B[0].append(data.test.tX[0][i])
            B[1].append(data.test.tX[1][i])

    #Plot data (projected to 2D)
    plt.figure("Breast Cancer Classification",figsize=(8,5))
    plt.title("Breast Cancer - Random Forest Classifier")

    #Plot training data
    plt.scatter(data.train.tXb[0],data.train.tXb[1],
                c="#cccccc",s=10,label="Training - Benign")
    plt.scatter(data.train.tXm[0],data.train.tXm[1],
                c="#777777",marker="X",s=30,
                label="Training - Malignent")

    #Plot prediction results over test data
    plt.scatter(B[0],B[1],
                c="blue",s=15,label="Predicted - Benign")    
    plt.scatter(M[0],M[1],
                c="red",marker="X",s=40,
                label="Predicted - Malignent")

    #Append accuracy
    plt.gcf().text(0.74,0.70,
                   "Accuracy = " + str(round(accuracy,2)) + "%")

    #Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()

    #Show
    plt.show()
    
#Execute Main ========================================================
if __name__ == "__main__":
    main()
