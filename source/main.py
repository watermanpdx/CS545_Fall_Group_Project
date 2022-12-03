#Imports =============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#Constants ===========================================================

#Functions/Classes====================================================

#Main ----------------------------------------------------------------
def main():
    print('Importing data ...')
    data = pd.read_csv('emails.csv')
    features = list(data.head())[1:-2]
    y = np.array(data)[:,-1].astype(int)
    x = np.array(data)[:,1:-2]

    print('Parsing training and test data ...')
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    """
    #Test performance over tree depth
    e_train = []
    e_test = []
    nodes = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500]

    for node in nodes:
        model = DecisionTreeClassifier(max_leaf_nodes=node)
        model.fit(x_train,y_train)
        e_train.append(model.score(x_train,y_train))
        e_test.append(model.score(x_test,y_test))

    plt.plot(nodes,e_train,c='red')
    plt.plot(nodes,e_test,c='blue')
    plt.show()
    """

    #Initial alphas calculation
    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)

    ccp = model.cost_complexity_pruning_path(x_train,y_train)
    alphas = ccp.ccp_alphas

    e_train = []
    e_test = []
    i = 0
    for alpha in alphas:
        print(str(i)+' of '+str(len(alphas)))
        i += 1
        
        model = DecisionTreeClassifier(ccp_alpha=alpha)
        model.fit(x_train,y_train)
        e_train.append(model.score(x_train,y_train))
        e_test.append(model.score(x_test,y_test))

    plt.plot(alphas,e_train,c='red')
    plt.plot(alphas,e_test,c='gray')
    plt.show()

    """
    #Train and plot model
    print('Training model ...')
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    print(acc)

    plt.figure()
    tree.plot_tree(model,feature_names=features,filled=True)
    plt.show()
    """
    
#Execute Main ========================================================
if __name__ == "__main__":
    main()
