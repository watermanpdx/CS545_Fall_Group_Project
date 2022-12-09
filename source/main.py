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
Date:           12/05/2022
======================================================================
'''

#Imports =============================================================
#General support packages
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Supporting Sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

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
def print_features(features,per_line=10):
    #Print feature set to terminal
    #Inputs:    features - (list) features in dataset
    #           per_line - number of features to print per line
    #Outputs:   None - prints to terminal directly

    #Print data to terminal
    print("Input data features:")
    line = []
    for feature in features:
        if len(line) < per_line:
            line.append(feature)
        else:
            print('\t' + str(line)[1:-1])
            line = []
    print('\t' + str(line)[1:-1])
    
    return

def benchmark(models,X,Y,filename=None):
    #Benchmark analysis of multiple classifier models
    #  Builds comparison table of different models over test data
    #Inputs:    models - (list) list of model objects to test
    #           X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           filename - (string) filename to save to (if provided)
    #Outputs:   (filename) - saves to file if filename not None
    #           return - (pandas DataFrame) results of benchmark

    #Initilize variables and initial user prompt
    print('Benchmarking models ...')
    results = pd.DataFrame()

    #Create training and test datasets from X and Y
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Validate across provided models
    for model in models:
        name = model.__repr__()
        print('\tValidating ' + name + ' ...')

        #Train model
        start = time.time()
        model.fit(x_train,y_train)
        fit_time = time.time() - start

        #Assess over test data and compute metrics
        start = time.time()
        y_predict = model.predict(x_test)
        score_time = time.time() - start
        
        tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        #Append results to DataFrame
        header = ['model','fit time(s)', 'score time(s)', 'accuracy', 'precision',
              'recall', 'TP', 'TN', 'FP', 'FN']
        data = [name, fit_time, score_time, accuracy, precision,
                recall, tp, tn, fp, fn]
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
    plt.close()
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

    #Construct plot
    plt.close()
    plt.figure('Split Criteria Performance')

    plt.title('Decision Tree Accuracy vs Tree Size Across Split Criteria')
    plt.xlabel('Tree Size (Max Nodes)')
    plt.ylabel('Accuracy (%)')
    
    plt.plot(nodes,e_gini,linestyle='solid',label='gini')
    plt.plot(nodes,e_entropy,linestyle='dashed',label='cross-entropy')

    plt.legend()

    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_info_functions(filename=None):
    #Plots info metric values across proportions of binary classes
    #Inputs:    filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None
    
    #Construct X values
    resolution = 0.01
    X = [x*resolution for x in range(int(1/resolution)+1)]

    #Compute gini and cross-entropy values
    gini = []
    entropy = []
    for x in X:
        p0 = x
        p1 = 1-x
        gini.append(p0*(1-p0) + p1*(1-p1))
        if p0 == 0 or p1 == 0:
            entropy.append(0)
        else:
            entropy.append(-p0*math.log2(p0) - p1*math.log2(p1))

    #Construct plot
    plt.close()
    fig,ax1 = plt.subplots()
    ax2 = plt.twinx()
    if filename == None:
        plt.gcf().canvas.set_window_title('Entropy vs Gini')

    plt.title('Information Metrics: Cross-Entropy vs Gini')
    plt.xlabel('Proportion of Class A vs B')
    ax1.set_ylabel('Impurity - Gini')
    ax2.set_ylabel('Impurity - Cross-Entropy')

    ax1.plot(X,gini,linestyle='solid',label='gini')
    ax2.plot(X,entropy,linestyle='dashed',label='cross-entropy')

    ax1.legend(loc=2)
    ax2.legend(loc=0)
    
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
    plt.close()
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

def plot_tree(X,Y,max_depth=4,ccp_alpha=0.0,features=None,filename=None):
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
    model = DecisionTreeClassifier(max_depth=max_depth,ccp_alpha=ccp_alpha)
    model.fit(x_train,y_train)
    acc = 100*model.score(x_test,y_test)
    print('Resultant (test) tree accuracy: ' + str(round(acc,2)) + '%')

    #Construct plot
    plt.close()
    plt.figure('Decision Tree Diagram')
    
    tree.plot_tree(model,feature_names=features,filled=True)
    
    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_feature_importance(X,Y,features,max_features=10,filename=None):
    #Plot feature significance
    #  Extract feature significance from forest impurity metrics
    #Inputs:    X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           features - (list) model feature names (strings)
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None

    #Train model
    print('Training model ...')
    model = RandomForestClassifier()
    model.fit(X,Y)

    #Compute feature importances via impurity reduction
    print('Computing feature importances ...')
    importances = model.feature_importances_

    #Sort features by importance and separate into plotable lists
    data = [(features[i],importances[i]) for i in range(len(features))]
    data.sort(reverse=True,key=lambda x:x[1])
    p_features = []
    p_importances = []
    for i in range(min(len(data), max_features)):
        p_features.append(data[i][0])
        p_importances.append(data[i][1])

    #Construct plot
    plt.close()
    plt.figure('Feature Importance')

    plt.title('Impurity-Based Variable Importance (Top ' + str(max_features) + ')')
    plt.xlabel('Features')
    plt.ylabel('Mean Gini Impurity Decrease')
    plt.xticks(rotation=90)
    
    plt.bar(p_features,p_importances)

    plt.tight_layout()

    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_forest_feature_size(X,Y,sizes,features,filename=None):
    #Plot forest accuracy as a function of feature size
    #  Forest accuracy against feature size during bootstrapping
    #Inputs:    X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           sizes - (list) of max features sizes to test
    #           features - (list) model feature names (strings)
    #                      (used for computing sqrt(features))
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None

    #Build training and testing sets
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Assess model performance over max feature sizes
    print('Assessing model performance ...')
    e_train = []
    e_test = []
    for i in range(len(sizes)):
        print('\t' + str(i+1) + ' of ' + str(len(sizes)) + ' : '
              + str(sizes[i]) + ' max features')
        model = RandomForestClassifier(max_samples=500,max_features=sizes[i])
        model.fit(x_train,y_train)
        
        e_train.append(100*model.score(x_train,y_train))
        e_test.append(100*model.score(x_test,y_test))
        
    #Construct plot
    plt.close()
    plt.figure('Bootstrapping Feature Size')

    plt.title('Forest Accuracy vs Max Bootstrapping Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy (%)')

    pos = math.sqrt(len(features))
    plt.axvline(x=pos,linestyle='dashed',c='black',label='sqrt(total features)')
    
    plt.plot(sizes,e_train,linestyle='solid',label='training')
    plt.plot(sizes,e_test,linestyle='dashed',label='test')

    plt.legend()
    
    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_forest_sample_size(X,Y,sizes,filename=None):
    #Plot forest accuracy as a function of sample size
    #  Forest accuracy against bootstrapping sample size
    #Inputs:    X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           sizes - (list) of max sample size to test
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None
    
    #Build training and testing sets
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Assess model performance over max feature sizes
    print('Assessing model performance ...')
    e_train = []
    e_test = []
    for i in range(len(sizes)):
        print('\t' + str(i+1) + ' of ' + str(len(sizes)) + ' : '
              + str(sizes[i]) + ' max features')
        model = RandomForestClassifier(max_samples=sizes[i])
        model.fit(x_train,y_train)
        
        e_train.append(100*model.score(x_train,y_train))
        e_test.append(100*model.score(x_test,y_test))
        
    #Construct plot
    plt.close()
    plt.figure('Forest Sample Size')

    plt.title('Forest Accuracy vs Size of Bootstrapping Samples')
    plt.xlabel('Sample Size')
    plt.ylabel('Accuracy (%)')

    plt.plot(sizes,e_train,linestyle='solid',label='training')
    plt.plot(sizes,e_test,linestyle='dashed',label='test')

    plt.legend()
    
    #Save plot to file or present to user directly
    if filename != None:
        print('Writing plot to file <' + filename + '> ...')
        plt.savefig(filename)
    else:
        plt.show()

    return

def plot_forest_estimator_size(X,Y,sizes,filename=None,oob=False):
    #Plot forest accuracy as a function of number of trees
    #  Forest accuracy against contained trees in forest
    #Inputs:    X - (numpy array) input data
    #           Y - (numpy array) classification data
    #           sizes - (list) of number of trees to test
    #           filename - (string) filename to save to. Plots to
    #                      window directly if not provided
    #Outputs:   (filename) - saves to file if filename not None
    #           return - None
    
    #Build training and testing sets
    print('Splitting data into training and test sets ...')
    x_train, x_test, y_train, y_test = train_test_split(X,Y)

    #Assess model performance over max feature sizes
    print('Assessing model performance ...')
    e_train = []
    e_test = []
    e_oob = []
    for i in range(len(sizes)):
        print('\t' + str(i+1) + ' of ' + str(len(sizes)) + ' : '
              + str(sizes[i]) + ' max features')
        model = RandomForestClassifier(oob_score=oob,max_samples=500,n_estimators=sizes[i])
        model.fit(x_train,y_train)
        
        e_train.append(100*model.score(x_train,y_train))
        e_test.append(100*model.score(x_test,y_test))

        if oob: e_oob.append(100*model.oob_score_)
        
    #Construct plot
    plt.close()
    plt.figure('Forest Estimator Size')

    plt.title('Forest Accuracy vs Number of Estimators (Trees)')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy (%)')

    plt.plot(sizes,e_train,linestyle='solid',label='training')
    plt.plot(sizes,e_test,linestyle='dashed',label='test')
    if oob: plt.plot(sizes,e_oob,c='gray',
             linestyle='dotted',label='OOB (training)')

    plt.legend()
    
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
    #print_features(features)
    Y = np.array(data)[:,-1].astype(int)
    X = np.array(data)[:,1:-2]
    print('\t(Percent positives in data: ' + str(round(100*Y.mean(),2)) + '%)')

    #Benchmark models
    models = [DecisionTreeClassifier(),
              RandomForestClassifier(),
              SVC(C=100,kernel='rbf'),
              GaussianNB(),
              LogisticRegression()]
    benchmark(models,X,Y,filename='results.csv')
    
    '''
    #Forest accuracy vs sample size
    sizes = [1,10,20,30,40,50,60,70,80,90,100,125,150,175,
             200,225,250,275,300,325,350,375,
             400,425,450,475,500,550,600,650,700,750,
             800,850,900,1000,1100,1200,1300,1400,1500,
             1600,1700,1800,1900,2000]
    plot_forest_sample_size(X,Y,sizes,'forest_max_samples.png')
    
    #Forest accuracy vs number of estimators
    sizes = [5,10,20,30,40,50,60,70,80,90,100,
             110,120,130,140,150,160,170,180,190,200,
             225,250,275,300,325,350,375,400,
             425,450,475,500]
    plot_forest_estimator_size(X,Y,sizes,'forest_n_estimators.png')
    plot_forest_estimator_size(X,Y,sizes,'forest_n_estimators_oob.png',oob=True)
    
    #Forest accuracy vs feature size in bagging
    sizes = [1,5,10,25,50,75,100,150,200,250,300,350,400,450,
             500,550,600,650,700,750,800,850,900,950,
             1000,1250,1500,1750,2000]
    plot_forest_feature_size(X,Y,sizes,features,'bootstrapping_features.png')
    
    #Random Forest feature importance
    plot_feature_importance(X,Y,features,25,'forest_feature_importance.png')

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

    plot_tree(X,Y,max_depth=None,ccp_alpha=0.005,
              features=features,filename='tree_diagram_pruning.png')

    #Train and plot model
    plot_tree(X,Y,features=features,filename='tree_diagram.png')
    
    #Plot entropy vs gini values
    plot_info_functions('entropy_vs_gini.png')
    '''

#Execute Main ========================================================
if __name__ == "__main__":
    main()
