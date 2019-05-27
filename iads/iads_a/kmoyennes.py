# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import math
import random

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(df):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon
             la méthode vue en cours 8.
    """
    return (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.sqrt(((v1 - v2) ** 2).sum())

def dist_vect_df(df, series):
    return df.apply(lambda x: dist_vect(x, series), axis=1)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(df):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return df.aggregate(['mean'], axis=0)

def centroide_ser(df): # renvoie un Series
    return df.mean(axis=0)

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    return (dist_vect_df(df, centroide_ser(df)) ** 2).sum()


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(k, df):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    return df.sample(n=k, axis=0)


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(example, centres):
    """ Series * DataFrame -> int
        example : Series contenant un exemple
        centres : DataFrame contenant les K centres
    """
    dists = dist_vect_df(centres, example)
    return np.argwhere(dists == dists.min())[0][0]

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(base, centres):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        base: DataFrame contenant la base d'apprentissage
        centres : DataFrame contenant des centroides
    """
    matrix = {k: [] for k in range(len(centres))}
    for ind, example in base.iterrows():
        matrix[plus_proche(example, centres)].append(ind)
    return matrix

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(base, matrix):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        base : DataFrame contenant la base d'apprentissage
        matrix : Dictionnaire d'affectation
    """
    keys = [i for i in matrix.keys()]
    centres = base.iloc[keys].copy()
    for ind, examples in matrix.items():
        centres.iloc[ind] = centroide_ser(base.iloc[examples])
    return centres

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(base, matrix):
    """ DataFrame * dict[int,list[int]] -> float
        base : DataFrame pour la base d'apprentissage
        matrix : Dictionnaire d'affectation
    """
    inertia = [inertie_cluster(base.iloc[examples]) for examples in matrix.values()]
    return sum(inertia)
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(k, base, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        k : entier > 1 (nombre de clusters)
        base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    # partition de départ
    current_centres = initialisation(k, base)
    current_matrix = affecte_cluster(base, current_centres)
    current_inertia = inertie_globale(base, current_matrix)

    curr_iter = 0

    # détermination de la partition suivante
    while curr_iter < iter_max:
        curr_iter += 1
        current_centres = nouveaux_centroides(base, current_matrix)
        current_matrix = affecte_cluster(base, current_centres)
        new_inertia = inertie_globale(base, current_matrix)

        # arrêt si convergence
        if abs(current_inertia - new_inertia) < epsilon:
            break

        current_inertia = new_inertia
    return current_centres, current_matrix
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(base, centres, matrix):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    _colors = np.random.choice(np.linspace(0.,1.,10), size=len(centres), replace=False)
    colors = [0] * len(base)
    for ind in sorted(matrix.keys()):
        for ex in matrix[ind]:
            colors[ex] = _colors[ind]
    plt.scatter(base['X'], base['Y'], c=colors)
    plt.scatter(centres['X'], centres['Y'], c=_colors, marker='+')

def affiche_resultat_sans_centres(base, matrix, k):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """
    _colors = np.random.choice(np.linspace(0.,1.,100), size=k, replace=False)
    colors = [0] * len(base)
    for ind in sorted(matrix.keys()):
        for ex in matrix[ind]:
            colors[ex] = _colors[ind]
    plt.scatter(base['X'], base['Y'], c=colors)
    # plt.scatter(centres['X'], centres['Y'], c=_colors, marker='+')

def dist_intracluster(base):
    # pour chaque ligne, la distance max avec une autre ligne
    dists = base.apply(lambda x: dist_vect_df(base, x).max(), axis=1)
    return dists.max()

def global_intraclusters(base, matrix):
    dists = np.zeros(len(matrix), dtype=np.float)
    for ind, examples in enumerate(matrix.values()):
        dists[ind] = dist_intracluster(base.iloc[examples])
    return dists.max()

def sep_clusters(centres):
    def distance(c0):
        # dropna() pour négliger le cas dist(x, x)
        return dist_vect_df(centres[centres != c0].dropna(), c0).min()
    dists = centres.apply(lambda x: distance(x), axis=1)
    return dists.min()

def dunn_index(base, centres, matrix):
    return global_intraclusters(base, matrix) / sep_clusters(centres)


def optimise_k_means(base, max_k, eps, max_iter):
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = [10,10]

    k_values = np.arange(2, max_k+1)

    dunn = np.empty_like(k_values, dtype=np.float)
    # xb = np.zeros_like(k_values, dtype=np.float)

    for i, k in enumerate(k_values):
        centres, matrix = kmoyennes(k, base, eps, max_iter)
        dunn[i] = dunn_index(base, centres, matrix)
        # xb[i] += xb_index(base, centres, matrix)

    dunn_pairs = [(k, ind) for k, ind in zip(k_values, dunn)]
    # xb_pairs = [(k, ind) for k, ind in zip(k_values, xb)]
    dunn_pairs.sort(key=lambda x: x[1])
    # xb_pairs.sort(key=lambda x: x[1])
    print("Nombre de clusters optimal selon l'index de Dunn :", dunn_pairs[0][0])
    # print("Optimal number of clusters according to the Xie-Beni index:", xb_pairs[0][0])

    plt.title("Variation de l'index de Dunn selon le nombre de clusters utilisés")
    # legends = ['Dunn index', 'Xie-Beni index']
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Index de Dunn')
    plt.plot(k_values, dunn)
    # plt.plot(k_values, xb)
    # plt.legend(legends)
    plt.show()

    return dunn_pairs
