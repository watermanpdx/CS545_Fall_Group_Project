'''
======================================================================
Description:    Decision Tree Analysis
Purpose:        Implementation of Decision Tree and Random Forest
                learning algorithms as part of model implementation
                and analysis. Explores behavior over multiple model
                hyper-parameters and benchmarks against alternative
                learning models.
Python:         Version 3.9
Authors:        Tayte Waterman
                Manisha Yadav
                Mary Muhly
                Brandon Gatewood
Course:         CS445/CS545 - Machine Learning
Assignment:     CS445/CS545 - Group Project
Date:           12/03/2022
======================================================================
'''

#Imports =============================================================
#General support packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Supporting Sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

#Decision tree packages
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Benchmark models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#Constants ===========================================================

#Functions/Classes====================================================
def benchmark(models,metrics,X,Y,filename=None):
    #Benchmark analysis of multiple classifier models
    #  Builds comparison table of different models over test data
    #Inputs:    models - (list) list of model objects to test
    #           metrics - (list) string arguments for cross_validate
    #                     to include in performance metrics results
    #           X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           filename - (string) filename to save to (if provided)
    #Outputs:   (filename) - saves to file if filename not None
    #           return - (pandas DataFrame) results of benchmark

    #Initilize variables and initial user prompt
    print('Benchmarking models ...')
    results = pd.DataFrame()
    header = None

    #Validate across provided models
    for model in models:
        name = model.__repr__()
        print('\tValidating ' + name + ' ...')

        #Extract performance results
        performance = cross_validate(model,X,Y,scoring=metrics)

        #Define DataFrame headers if not yet set
        if header == None:
            header = ['model'] + [key for key in performance]

        #Append results to DataFrame
        data = [name] + [performance[key].mean() for key in performance]
        results = pd.concat([results, pd.DataFrame([data],columns=header)])

    #Reset and remove leading DataFrame indices
    results = results.reset_index(drop=True)

    #Save to file if filename provided
    if filename != None:
        print('\tWriting to file <' + filename + '> ...')
        results.to_csv(filename,index=False)

    return results

def plot_tree_size(nodes,X,Y,filename=None):
    #Assess decision tree performance as function of tree size
    #  Assesses tree performance with increasing number of max nodes
    #Inputs:    nodes - (list) list of number of max nodes to test
    #           X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None

    #Create training and test datasets from X and Y
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Initialize variables
    print('Test trees of increasing size:')
    e_train = []
    e_test = []

    #Test tree against multiple max node values
    for node in nodes:
        print('\tTesting max nodes = ' + str(node) + ' ...')
        model = DecisionTreeClassifier(max_leaf_nodes=node)
        model.fit(x_train,y_train)
        e_train.append(100*model.score(x_train,y_train))
        e_test.append(100*model.score(x_test,y_test))

    #Construct plot
    plt.figure('Perfomance vs Tree Size')

    plt.title('Decision Tree Accuracy vs Tree Size')
    plt.xlabel('Tree Size (Max Nodes)')
    plt.ylabel('Accuracy (%)')
    
    plt.plot(nodes,e_train,linestyle='solid',label='training')
    plt.plot(nodes,e_test,linestyle='dashed',label='test')
    
    plt.legend()

    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_greedy_criteria(nodes,X,Y,filename=None):
    #Assess decision tree performance as function of split criteria
    #  Assesses tree performance based on different splitting algorithms
    #Inputs:    nodes - (list) list of number of max nodes to test
    #           X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None

    #Create training and test datasets from X and Y
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Initialize variables
    print('Test trees of increasing size across splitting criteria:')
    e_gini = []
    e_entropy = []
    e_log = []

    #Test tree against multiple max node values. For each test and store
    #  results of split criteria algorithms: gini, cross-entropy, and
    #  log-loss
    for node in nodes:
        print('\tTesting max nodes = ' + str(node) + ' ...')

        print('\t\tgini ...')
        gini = DecisionTreeClassifier(max_leaf_nodes=node,criterion='gini')
        gini.fit(x_train,y_train)
        e_gini.append(100*gini.score(x_train,y_train))
        
        print('\t\tentropy ...')
        entropy = DecisionTreeClassifier(max_leaf_nodes=node,criterion='entropy')
        entropy.fit(x_train,y_train)
        e_entropy.append(100*entropy.score(x_train,y_train))

        print('\t\tlog loss ...')
        log = DecisionTreeClassifier(max_leaf_nodes=node,criterion='log_loss')
        log.fit(x_train,y_train)
        e_log.append(100*log.score(x_train,y_train))

    #Construct plot
    plt.figure('Split Criteria Performance')

    plt.title('Decision Tree Accuracy vs Tree Size Across Split Criteria')
    plt.xlabel('Tree Size (Max Nodes)')
    plt.ylabel('Accuracy (%)')
    
    plt.plot(nodes,e_gini,linestyle='solid',label='gini')
    plt.plot(nodes,e_entropy,linestyle='dashed',label='cross-entropy')
    plt.plot(nodes,e_log,linestyle='dotted',label='log-loss')

    plt.legend()

    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_pruning(X,Y,filename=None):
    #Assess decision tree performance across pruning
    #  Assesses tree performance across multiple levels of pruning
    #Inputs:    X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None

    #Create training and test datasets from X and Y
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)
    
    #Initial alphas calculation
    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)
    ccp = model.cost_complexity_pruning_path(x_train,y_train)
    alphas = ccp.ccp_alphas

    #Initialize variables
    e_train = []
    e_test = []
    i = 1

    #Assess tree performance over alpha thresholds
    print('Test trees of increasing alpha thresholds:')
    for alpha in alphas:
        print('\tAlpha threshold = ' + str(alpha) + ', '
              + str(i)+' of '+str(len(alphas)))
        i += 1
        
        model = DecisionTreeClassifier(ccp_alpha=alpha)
        model.fit(x_train,y_train)
        e_train.append(100*model.score(x_train,y_train))
        e_test.append(100*model.score(x_test,y_test))

    #Construct plot
    plt.figure('Tree Performance vs Alpha Threshold')

    plt.title('Decision Tree Accuracy vs Pruning Alpha Threshold')
    plt.xlabel('Alpha Threshold')
    plt.ylabel('Accuracy (%)')

    plt.plot(alphas,e_train,linestyle='solid',label='training')
    plt.plot(alphas,e_test,linestyle='dashed',label='test')

    plt.legend()

    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_tree(X,Y,max_depth=4,features=None,filename=None):
    #Plot tree flowchart
    #  Plot resultant trained tree as human-readable diagram
    #Inputs:    X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           params - (list) DecisionTreeClassifier() arguments
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None

    #Create training and test datasets from X and Y
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Train and plot model
    print('Training model ...')
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(x_train,y_train)
    acc = 100*model.score(x_test,y_test)
    print('Resultant (test) tree accuracy: ' + str(round(acc,2)) + '%')

    #Construct plot
    plt.figure('Decision Tree Diagram')
    
    tree.plot_tree(model,feature_names=features,filled=True)
    
    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

#Main ----------------------------------------------------------------
def main():
    #Import dataset from .csv; not yet split into training/test sets
    print('Importing data ...')
    data = pd.read_csv('emails.csv')
    features = list(data.head())[1:-2]
    Y = np.array(data)[:,-1].astype(int)
    X = np.array(data)[:,1:-2]

    #Benchmark models
    models = [DecisionTreeClassifier(),
              RandomForestClassifier(),
              SVC(),
              GaussianNB(),
              LogisticRegression()]
    metrics = ['accuracy','precision','recall']
    benchmark(models,metrics,X,Y,filename='results.csv')

    #Test performance over greedy optimization methods
    nodes = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,
             100,110,120,130,140,150,160,170,180,190,200]
    plot_greedy_criteria(nodes,X,Y,'greedy_criteria_analysis.png')

    #Test performance over tree size
    nodes = [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,
             100,110,120,130,140,150,160,170,180,190,200]
    plot_tree_size(nodes,X,Y,'tree_size_analysis.png') 

    #Test performance over pruning values
    plot_pruning(X,Y,filename='pruning_analysis.png')

    #Train and plot model
    plot_tree(X,Y,features=features,filename='tree_diagram.png')

#Execute Main ========================================================
if __name__ == "__main__":
    main()
