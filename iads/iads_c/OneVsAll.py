import numpy as np
import pandas as pd
import copy

def one_vs_all(algorithm,parameters,df_train):
    """
    Methode One vs All

    Parameters
    ----------
    algorithme : 
        L'algorithme à utiliser.

    parameters : liste
        liste des parametres de l'algorithme.
    df_train : LabeledSet
        L'ensemble d'apprentissage.

    Returns
    -------
    liste
        liste des classifieurs.
    """
    classifiers = []
    nb_classes = df_train.y.shape[1]-1
    for i in range(nb_classes):
        df_train_c = copy.copy(df_train)
        df_train_c.y = df_train.y[:,i]
        df_train_c.y = df_train_c.y.reshape(-1,1)
        classifier = algorithm(*parameters)
        classifier.train(df_train_c)
        classifiers.append(classifier)
    return classifiers

def predict_one_vs_all(classifiers,df_test):
    """
    Methode de prédiction avec l'approche One vs All.

    Parameters
    ----------
    classifiers : Liste
        La liste des classifieurs renvoyé par la méthode one_vs_all.

    df_test : LabeledSet
        Ensemble de test.
    
    Returns
    -------
    float
        L'accuracy du modèle.

    """
    scores = np.zeros((df_test.y.shape[1] - 1, df_test.size()))
    expected = np.zeros((df_test.size()))
    num_classes = df_test.y.shape[1] - 1
    for i in range(num_classes):
        classifier = classifiers[i]
        for j in range(df_test.size()):
            scores[i, j] = classifier.predict(df_test.getX(j))
            expected[j] = df_test.getY(j)[-1]
    prediction = np.argmax(scores, axis=0)
    accuracy = np.mean(expected == prediction)
    return accuracy
