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
def normalisation(DF):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    mins = np.min(DF,axis=0)
    sigma = np.max(DF,axis=0) - mins
    return (DF-mins)/sigma

# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.sqrt(np.sum((v1-v2)**2))

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(DF):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return DF.mean().to_frame().transpose()

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(DF):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    c=centroide(DF)
    I=0
    for i in range(len(DF)):
        I+=dist_vect(DF.iloc[i],c.iloc[0])**2
    return I


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(K,DF):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    return DF.sample(K)


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(Exe,Centres):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    min_d=99999999
    for i in range(len(Centres)):
        d=dist_vect(Centres.iloc[i],Exe)
        if(d<min_d):
            min_d=d
            ind=i
    return ind

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(Base,Centres):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    d=dict()
    for i in range(len(Base)):
        k=plus_proche(Base.iloc[i],Centres)
        if(k in d.keys()):
            d[k].append(i)
        else:
            d[k]=[i]
    return d

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(Base,U):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    new=[]
    for i,k in enumerate(U):
        tab=U[k]
        c=centroide(Base.iloc[tab])
        new.append(c)
        c.index={i}
    return pd.concat(new)

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(Base, U):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    I=0
    for k in U.keys():
        I+=inertie_cluster(Base.iloc[U[k]])
    return I
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(K, Base, epsilon, iter_max):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    centroide_df=initialisation(K,Base)
    J=dict()
    J[0]=0
    for i in range(iter_max-1):
        mat=affecte_cluster(Base,centroide_df)
        J[1]=inertie_globale(Base,mat)
        if(abs(J[1]-J[0])<epsilon):
            return centroide_df,mat
        else:
            J[0]=J[1]
            centroide_df=nouveaux_centroides(Base,mat)
    return centroide_df,affecte_cluster(Base,centroide_df)
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(Base,centroide_df,mat):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    plt.scatter(centroide_df['X'],centroide_df['Y'],color='red',marker='x')
    coulours=['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    for i ,k in enumerate(mat):
        data=Base.iloc[mat[k]]
        plt.scatter(data['X'],data['Y'],color=coulours[i])
# -------
def dist_intracluster(df):
    d=list()
    for i in range(len(df)):
        for j in range(len(df)):
            if(i!=j):
                d.append(dist_vect(df.iloc[i],df.iloc[j]))
    return max(d)

def global_intraclusters(df,mat):
    dg=list()
    for k in mat.keys():
        dg.append(dist_intracluster(df.iloc[mat[k]]))
    return max(dg)

def sep_clusters(centroides_df):
    s=list()
    for i in range(len(centroides_df)):
        for j in range(len(centroides_df)):
            if(i!=j):
                s.append(dist_vect(centroides_df.iloc[i],centroides_df.iloc[j]))
    return min(s)

def evaluation(nom,df,centroides_df,mat):
    if(nom=="Dunn"):
        compacite=global_intraclusters(df,mat)
    else:
        compacite=inertie_globale(df,mat)
    return compacite/sep_clusters(centroides_df)
