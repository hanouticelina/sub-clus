from random import sample
import pandas as pd
import numpy as np
import graphviz as gv
from . import Classifiers as cl
from . import OneVsAll as ova
from . import LabeledSet as ls
from . import utils as ut
from . import DecisionTrees_ as dt

def shuffle_rows(df):
    """
    Effectue un shuffle des elements présents dans le DataFrame.
    
    Parameters
    -------------
    df : DataFrame
        dataframe à considérer.
    Returns
    ------------
    Un dataframe
    """
    new_df = df.iloc[sample(range(df.shape[0]),df.shape[0]),:]
    return new_df


def K_subsets(df, k):
    """
    Renvoie K folds à partir d'un Dataframe
    Parameters
    ------------
    df : DataFrame
        dataframe à considérer
    k : int
        taille des folds
    Returns
    -----------
    K Folds
    """
    k_datasets = list(range(k))
    start = 0
    end = round(df.shape[0]/k)

    shuffle_data = shuffle_rows(df)
    for i in range(k):
        if i == (k-1):
            k_datasets[i] = shuffle_data.iloc[start:, :]
        else:
            k_datasets[i] = shuffle_data.iloc[start:end, :]
            start = end
            end += round(df.shape[0]/k)

    return k_datasets

def K_cross(K_data,algorithm, parameters,attributs):
    """
    Permet d'effectuer la cross validation.
    Parameters
    -----------
    K_data : list of dataframe.
        les K folds.
    algorithm :
        L'algorithme à utiliser.
    parameters : list
        les paramètres que prend l'algorithme.
    attributs : list of str
        l'attribut qui représente la classe qu'on souhaite prédire.
    Returns
    -----------
    list of float
        l'accuracy pour chaque fold
    """
    k = len(K_data)
    accs = []
    for i in range(k):
        #remove one of the k data sets
        k_data_temp = [K_data[j] for j in range(k) if j != i]
        df_train_data = pd.concat(k_data_temp)
        #create all the training and test sets
        X,Y = ut.getXY(df_train_data,attributs)
        train_set = ut.loadSet(X,Y)
        test_data = pd.DataFrame(K_data[i])
        X_test,Y_test = ut.getXY(test_data,attributs)
        test_set = ut.loadSet(X_test,Y_test)
        #run the algorithm and get the accuracy and predicted values
        classifier = algorithm(*parameters)
        classifier.train(train_set)
        acc = classifier.accuracy(test_set)
        accs.append(acc)

    return accs


def K_cross_validation(df, k,algorithm,parameters,attributs):
    Kfolds_data = K_subsets(df, k)
    accuracy = K_cross(Kfolds_data,algorithm,parameters,attributs)
    return accuracy

def Print_Cross_Validation(name, accuracy):
    print(name+': ')
    for i in range(len(accuracy)):
        print('-------------------------------------')
        print('fold: ',i)
        print('Accuracy: ',accuracy[i])
