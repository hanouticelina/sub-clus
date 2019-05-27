# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: decision_trees.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les arbres de décision

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import math
import random

def log_k(base, number):
    return np.log(number) / np.log(base)

def shannon(prob_distro):
    if len(prob_distro) <= 1: 
        return 0
    tmp = [-p * log_k(len(prob_distro), p) if p > 0. else 0. for p in prob_distro]
    return sum(tmp)

def get_classes(LSet):
    return np.sort(np.unique(LSet.y))[::-1]

def entropy(values, classes):
    vfunc = np.vectorize(lambda c: len(np.where(values == c)[0]) / len(values))
    prob_distro = vfunc(classes)
    return shannon(prob_distro)

def classe_majoritaire(LSet):
    classes = get_classes(LSet)
    vfunc = np.vectorize(lambda c: len(np.where(LSet.y == c)[0]))
    nb_examples_per_class = vfunc(classes)
    max_index = np.argmax(nb_examples_per_class)
    return classes[max_index]

def entropie(LSet, classes):
    try:
        return entropy(LSet.y, classes)
    except:
        return shannon([])

def discretise(LSet, col):
    """ LabelledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)

def divise(LSet, att, seuil):
    linf = ls.LabeledSet(LSet.getInputDimension())
    lsup = ls.LabeledSet(LSet.getInputDimension())
    size = LSet.size()
    
    ind = np.argsort(LSet.x, axis=0)[:, att]
    ind_of_last = np.where(LSet.x[ind, att] <= seuil)[0][-1]
    
    for i in ind[:ind_of_last+1]:
        linf.addExample(LSet.getX(i), LSet.getY(i))
    for i in ind[ind_of_last+1:]:
        lsup.addExample(LSet.getX(i), LSet.getY(i))
    return linf, lsup

def construit_AD(current, epsilon, classes=None):
    if classes is None: classes = get_classes(current)
    arbre = ArbreBinaire()
    if entropie(current, classes) <= epsilon:
        arbre.ajoute_feuille(classe_majoritaire(current))
    else:
        h_min = 1.1
        seuil_min = 0.
        col_min = -1
        for col in range(current.getInputDimension()):
            seuil, h = discretise(current, col)
            if h < h_min:
                h_min = h
                seuil_min = seuil
                col_min = col
        linf, lsup = divise(current, col_min, seuil_min)
        ABinf = construit_AD(linf, epsilon, classes)
        ABsup = construit_AD(lsup, epsilon, classes)
        arbre.ajoute_fils(ABinf, ABsup, col_min, seuil_min)
    return arbre

class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g

class ArbreGeneral:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None # seulement pour les attributs numériques
        self.labels = [] # labels des arêtes
        self.fils = [] # liste de fils
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.fils == []
    
    def ajoute_fils(self,AGfils,att,labels,seuil=None):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.labels = labels # labels sur les arcs
        self.fils = AGfils
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if self.seuil is not None: # cas d'attribut numérique
            if exemple[self.attribut] <= self.seuil:
                return self.fils[0].classifie(exemple)
            return self.fils[1].classifie(exemple)
        else:  # cas d'attribut catégoriel
            for child, label in zip(self.fils, self.labels):
                if label == str(exemple[self.attribut]):
                    return child.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            for i in range(len(self.fils)):
                self.fils[i].to_graph(g,prefixe+str(i))
            for i in range(len(self.fils)):
                g.edge(prefixe,prefixe+str(i), self.labels[i])
        
        return g

def informationn_gain(LSet, entropy, partition, classes):
    h_subsets = [ls.size()*entropie(ls, classes) for ls in partition]
    return entropy - sum(h_subsets) / LSet.size()

def construit_AD_GI(current, epsilon, classes=None):
    if classes is None: classes = get_classes(current)
    arbre = ArbreBinaire()
    current_h = entropie(current, classes)
    if current_h <= epsilon:
        arbre.ajoute_feuille(classe_majoritaire(current))
    else:
        h_min = 1.1
        seuil_min = 0.
        col_min = -1
        for col in range(current.getInputDimension()):
            seuil, h = discretise(current, col)
            if h < h_min:
                h_min = h
                seuil_min = seuil
                col_min = col
        linf, lsup = divise(current, col_min, seuil_min)
        if informationn_gain(current, current_h, [linf, lsup], classes) < epsilon:
            arbre.ajoute_feuille(classe_majoritaire(current))
        else:
            ABinf = construit_AD_GI(linf, epsilon, classes)
            ABsup = construit_AD_GI(lsup, epsilon, classes)
            arbre.ajoute_fils(ABinf, ABsup, col_min, seuil_min)
    return arbre

def tirage(indices, m, r):
    return np.random.choice(indices, size=m, replace=r)

def sampleLS(X, indices):
    lset = ls.LabeledSet(X.getInputDimension())
    for ind in indices:
        lset.addExample(X.getX(ind), X.getY(ind))
    return lset

def echantillonLS(X, m, r):
    indices = tirage(np.arange(X.size()), m, r)
    return sampleLS(X, indices)

def complement(X, indices):
    indices = np.unique(indices)
    return np.array([i for i in range(X.size()) if i not in indices])

def construit_AD_aleatoire(current, epsilon, nbatt, classes=None):
    if classes is None: classes = get_classes(current)
    arbre = ArbreBinaire()
    current_h = entropie(current, classes)
    if current_h <= epsilon:
        arbre.ajoute_feuille(classe_majoritaire(current))
    else:
        h_min = 1.1
        seuil_min = 0.
        col_min = -1
        cols = np.random.choice(np.arange(current.getInputDimension()),
                                size=nbatt, replace=False)
        for col in cols:
            seuil, h = discretise(current, col)
            if h < h_min:
                h_min = h
                seuil_min = seuil
                col_min = col
        linf, lsup = divise(current, col_min, seuil_min)
        if linf.size() == 0 or lsup.size() == 0:
            arbre.ajoute_feuille(classe_majoritaire(current))
        if informationn_gain(current, current_h, [linf, lsup], classes) < epsilon:
            arbre.ajoute_feuille(classe_majoritaire(current))
        else:
            ABinf = construit_AD_aleatoire(linf, epsilon, nbatt, classes)
            ABsup = construit_AD_aleatoire(lsup, epsilon, nbatt, classes)
            arbre.ajoute_fils(ABinf, ABsup, col_min, seuil_min)
    return arbre

def get_categorical_values(the_set, att):
    return np.unique(the_set.x[:, att])

def divise_gen(the_set, att, seuil):
    if seuil is not None: # données numériques
        linf, lsup = divise(the_set, att, seuil)
        return [linf, lsup]
    #données catégorielles
    categories = get_categorical_values(the_set, att)
    lsets = []
    for i in range(len(categories)):
        lsets.append(ls.LabeledSet(the_set.getInputDimension()))
    size = the_set.size()
    
    for c in range(len(categories)):
        indices = np.where(the_set.x[:, att] == categories[c])[0]
        for i in indices:
            lsets[c].addExample(the_set.getX(i), the_set.getY(i))
            
    return lsets

def is_categorical(the_set, att):
    import numbers
    return not isinstance(the_set.x[0, att], numbers.Number)

def construit_ADGen(current, epsilon, classes=None):
    if classes is None: classes = get_classes(current)
    arbre = ArbreGeneral()
    current_h = entropie(current, classes)
    if current_h <= epsilon:
        arbre.ajoute_feuille(classe_majoritaire(current))
    else:
        h_min = 1.1
        seuil_min = None
        att_min = -1
        for att in range(current.getInputDimension()):
            if is_categorical(current, att):
                seuil = None
                h = entropy(current.x[:, att], get_categorical_values(current, att))
            else:
                seuil, h = discretise(current, att)
            if h < h_min:
                h_min = h
                seuil_min = seuil
                att_min = att
        lsets = divise_gen(current, att_min, seuil_min)
        if informationn_gain(current, current_h, lsets, classes) < epsilon:
            arbre.ajoute_feuille(classe_majoritaire(current))
        else:
            if seuil_min is None:
                labels = [str(cat) for cat in get_categorical_values(current, att_min)]
            else:
                labels = ['<=' + str(seuil_min), '>' + str(seuil_min)]
            AGfils = [construit_ADGen(ls, epsilon, classes) for ls in lsets]
            arbre.ajoute_fils(AGfils, att_min, labels, seuil_min)
    return arbre
