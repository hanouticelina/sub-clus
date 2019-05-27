import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# La ligne suivante permet de préciser le chemin d'accès à la librairie iads


# Importation de la librairie iads


# importation de LabeledSet
from . import LabeledSet as ls

# importation de Classifiers
from . import Classifiers as cl

# importation de utils

from . import utils as ut
from math import log2


def classe_majoritaire(labeledSet,classes):
    long=list()
    for i in range(len(classes)):
        long.append(len(labeledSet.x[np.where(labeledSet.y == i),:][0]))
    return np.argmax(np.asarray(long))
def sampleLS(X, indices):
    lset = ls.LabeledSet(X.getInputDimension())
    for ind in indices:
        lset.addExample(X.getX(ind), X.getY(ind))
    return lset


def complement(X, indices):
    indices = np.unique(indices)
    return np.array([i for i in range(X.size()) if i not in indices])

def shannon(listeP):
    somme=0
    for p in listeP:
        if p != 0:
            somme+= p*log2(p)/log2(len(listeP))
    return - somme
def entropie(labeledSet,classes):
    C=classes
    P=[]
    for c in C:
            #print(labeledSet.size())
            S= labeledSet.x[np.where(labeledSet.y == c),:]
            P.append(len(S[0])/labeledSet.size())

    return shannon(P)
def discretise(LSet, col,classes):

    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)

    # calcul des distributions des classes pour E1 et E2:
    #inf_plus  = 0 [0,0]               # nombre de +1 dans E1
    #inf_moins = 0               # nombre de -1 dans E1
    #sup_plus  = 0 [0,0]              # nombre de +1 dans E2
    #sup_moins = 0
    inf=[0]*len(classes)
    sup=[0]*len(classes) # nombre de -1 dans E2
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E.
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins
    # dans E.
    for j in range(0,LSet.size()):
        sup[int(LSet.getY(j)[0])]+=1
    nb_total = sum(sup) # nombre d'exemples total dans E

    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        inf[int(LSet.getY(ind[i][col])[0])]+=1
        sup[int(LSet.getY(ind[i][col])[0])]-=1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = sum(inf)*1.0     # rem: on en fait un float pour éviter
        nb_sup = sum(sup)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        if(False):
            val_entropie_inf=0
        else:
            val_entropie_inf = shannon(np.asarray(inf)/nb_inf)
        if(False):
            val_entropie_sup=0
        else:
            val_entropie_sup = shannon(np.asarray(sup)/nb_sup)
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)

def divise(LSet,att,seuil):
    sbinf,sbsup = ls.LabeledSet(LSet.getInputDimension()),ls.LabeledSet(LSet.getInputDimension())
    for i in range(LSet.size()):
        if (LSet.getX(i)[att]<=seuil):
            sbinf.addExample(LSet.getX(i),LSet.getY(i))
        else:
            sbsup.addExample(LSet.getX(i),LSet.getY(i))
    return sbinf,sbsup

import graphviz as gv
# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz


class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1

    def est_feuille(self):
        return self.seuil == None

    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup

    def ajoute_feuille(self,classe):

        self.classe = classe

    def classifie(self,exemple,dico):

        if self.est_feuille():
            return self.classe
        if exemple[dico[self.attribut]] <= self.seuil:
            return self.inferieur.classifie(exemple,dico)
        return self.superieur.classifie(exemple,dico)

    def to_graph(self, g, prefixe='A'):

        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")

            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))

        return g
def construit_ad(LSet,epsilon,nom_col,classes):
    arbre=ArbreBinaire()
    if((entropie(LSet,classes)<=epsilon) or(LSet.size()==0)):
        if(LSet.size()==0):
            return

        arbre.ajoute_feuille(classe_majoritaire(LSet,classes))
        return arbre
    else:

        liste_Qmin,liste_Seuilmin=[],[]
        for i in range(len(LSet.getX(0))):
            couple=discretise(LSet,i,classes)
            liste_Seuilmin.append(couple[0])
            liste_Qmin.append(couple[1])
        attchoisit=liste_Qmin.index(min(liste_Qmin))
        probaplus= len( LSet.x[np.where(LSet.y == 1),:])/LSet.size()
        probamoins= len(LSet.x[np.where(LSet.y == -1),:])/LSet.size()

        if(entropie(LSet,classes) - min(liste_Qmin) <=epsilon or LSet.size()==0 ):
             arbre.ajoute_feuille(classe_majoritaire(LSet,classes))
        else:
            Linf, Lsup = divise(LSet,attchoisit,liste_Seuilmin[attchoisit])
            if(Lsup.size()==0):
                arbresup,arbreinf=ArbreBinaire(),ArbreBinaire()
                maj=classe_majoritaire(Linf,classes)
                z=random.choice(classes)
                while(z==maj):
                    z=random.choice(classes)
                arbresup.ajoute_feuille(z)


                arbreinf.ajoute_feuille(maj)



            elif(Linf.size()==0):
                arbreinf=ArbreBinaire()
                arbreinf.ajoute_feuille(-1)

                arbresup=construit_ad(Lsup,epsilon,nom_col,classes)
            else:
                arbreinf,arbresup= construit_ad(Linf,epsilon,nom_col,classes),construit_ad(Lsup,epsilon,nom_col,classes)
            arbre.ajoute_fils(arbreinf,arbresup,nom_col[attchoisit],liste_Seuilmin[attchoisit])

        return arbre
class ArbreDecision(cl.Classifier):
    # Constructeur
    def __init__(self,epsilon,nom_col,nom_num, classes):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
        self.nom_col=nom_col
        self.nom_num=nom_num
        self.classes = classes

    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend 0 (classe -1) ou 1 (classe 1)
        return self.racine.classifie(x,self.nom_num)
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        self.racine = construit_ad(set,self.epsilon,self.nom_col,self.classes)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)


def tirage(vx,m,r):
    if (r==False):
        
        return random.sample(vx,m)
    return [random.choice(vx) for i in range(m)]
 
def echantillonLS(LS_X, m, r):
    index = tirage([i for i in range (LS_X.size())], m, r)
    res = ls.LabeledSet(LS_X.getInputDimension())
    for ind in index:
        res.addExample(LS_X.getX(ind), LS_X.getY(ind))
    return res

class ClassifierBaggingTree(cl.Classifier):
    # Constructeur
    def __init__(self, B, m_percent, epsilon, replacement,nom_col,nom_num,classes):
        self.B = B
        self.m_percent = m_percent
        self.replacement = replacement
        self.epsilon= epsilon # valeur seuil d'entropie pour arrêter la construction
        self.forest = [] # fôrét d'arbres de décision
        self.nom_col = nom_col
        self.nom_num = nom_num
        self.classes = classes
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend -1 (classe -1) ou 1 (classe 1)
        votes = sum([tree.predict(x) for tree in self.forest])
        return np.sign(votes)

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        m = round(self.m_percent * set.size())
        # construction des arbres de décision
        for _ in range(self.B):
            tree = ArbreDecision(self.epsilon,self.nom_col,self.nom_num,self.classes)
            sample = echantillonLS(set, m, self.replacement)
            tree.train(sample)
            self.forest.append(tree)

# ---------------------------
class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    # Constructeur
    def __init__(self, B, m_percent, epsilon, replacement,nom_col, nom_num, classes):
        ClassifierBaggingTree.__init__(self, B, m_percent, epsilon, replacement,nom_col,nom_num,classes)
        self.oob = []

    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        m = round(self.m_percent * set.size())
        # construction des arbres de décision
        for _ in range(self.B):
            tree = ArbreDecision(self.epsilon,self.nom_col,self.nom_num,self.classes)
            sample_indices = tirage(np.arange(set.size()), m, self.replacement)
            sample = sampleLS(set, sample_indices)
            oob_indices = complement(set, sample_indices)
            oob = sampleLS(set, oob_indices)
            self.oob.append(oob)
            tree.train(sample)
            self.forest.append(tree)

    def oob_accuracy(self):
        ts = [self.forest[i].accuracy(self.oob[i]) for i in range(self.B)]
        return sum(ts) / self.B


def construit_ad_Alea(LSet,epsilon,nb_att,nom_col,classes):
    arbre=ArbreBinaire()
    if((entropie(LSet,classes)<=epsilon) or(LSet.size()==0)):
        if(LSet.size()==0):
            return

        arbre.ajoute_feuille(classe_majoritaire(LSet,classes))
        return arbre
    else:
        L = [i for i in range(LSet.getInputDimension())]
        liste_Qmin,liste_Seuilmin=[],[]
        cols = tirage(L, nb_att, False)
        for col in cols:
            couple=discretise(LSet,col,classes)
            liste_Seuilmin.append(couple[0])
            liste_Qmin.append(couple[1])
        attchoisit=liste_Qmin.index(min(liste_Qmin))
        probaplus= len( LSet.x[np.where(LSet.y == 1),:])/LSet.size()
        probamoins= len(LSet.x[np.where(LSet.y == -1),:])/LSet.size()

        if(entropie(LSet,classes) - min(liste_Qmin) <=epsilon or LSet.size()==0 ):
             arbre.ajoute_feuille(classe_majoritaire(LSet,classes))
        else:
            Linf, Lsup = divise(LSet,attchoisit,liste_Seuilmin[attchoisit])
            if(Lsup.size()==0):
                arbresup,arbreinf=ArbreBinaire(),ArbreBinaire()

                arbresup.ajoute_feuille(-1*classe_majoritaire(Linf,classes))


                arbreinf.ajoute_feuille(classe_majoritaire(Linf,classes))



            elif(Linf.size()==0):
                arbreinf=ArbreBinaire()
                arbreinf.ajoute_feuille(-1)

                arbresup=construit_ad_Alea(Lsup,epsilon,nb_att,nom_col,classes)
            else:
                arbreinf,arbresup=construit_ad_Alea(Linf,epsilon,nb_att,nom_col,classes),construit_ad_Alea(Lsup,epsilon,nb_att,nom_col,classes)


            arbre.ajoute_fils(arbreinf,arbresup,nom_col[attchoisit],liste_Seuilmin[attchoisit])

        return arbre

class ArbreDecisionAleatoire(ArbreDecision):
    # Constructeur
    def __init__(self,epsilon,nbatt,nom_col,nom_num,classes):
        super(ArbreDecisionAleatoire, self).__init__(epsilon,nom_col,nom_num,classes)
        self.nbatt = nbatt
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,Set):
        # construction de l'arbre de décision
        self.set=Set
        self.racine = construit_ad_Alea(Set,self.epsilon, self.nbatt,self.nom_col,self.classes)

class ClassifierRandomForest(ClassifierBaggingTree):
    """Arguments:
        - Le nombre B d'arbres à construire
        - Le pourcentage d'exemples de la base d'apprentissage utilisés pour constituer un échantillon
        - La valeur de seuil d'entropie pour arrêter la construction de chaque arbre
        - Un booléen qui précise si un échantillon est tiré avec ou sans remise
        - Le nombre de colonnes nbatt à utiliser à chaque niveau de l'arbre.
    """
    def __init__(self, B, pourc, seuil, r,nom_col,nom_num,nbatt,classes):
        super(ClassifierRandomForest, self).__init__(B, pourc, seuil, r,nom_col,nom_num,classes)
        self.nbatt = nbatt
    
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.arbres = set()
        taille = int(labeledSet.size() * self.m_percent)
        for i in range(self.B):
            temp_ls = echantillonLS(labeledSet, taille, self.replacement)
            temp_ad= ArbreDecisionAleatoire(self.epsilon, self.nbatt,self.nom_col,self.nom_num,self.classes)
            temp_ad.train(temp_ls)
            self.arbres.add(temp_ad)

class ClassifierRandomForestOOB(ClassifierRandomForest):
    """Arguments:
        - Le nombre B d'arbres à construire
        - Le pourcentage d'exemples de la base d'apprentissage utilisés pour constituer un échantillon
        - La valeur de seuil d'entropie pour arrêter la construction de chaque arbre
        - Un booléen qui précise si un échantillon est tiré avec ou sans remise
        - Le nombre de colonnes nbatt à utiliser à chaque niveau de l'arbre.
    """

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.arbres = dict()
        taille = int(labeledSet.size() * self.m_percent)
        for i in range(self.B):
            index = tirage([i for i in range (labeledSet.size())], taille, self.replacement)
            temp_ls = echantillonDepuisIndices(labeledSet, index)
            temp_ad= ArbreDecisionAleatoire(self.seuil, self.nbatt,self.classes)
            temp_ad.train(temp_ls)
            self.arbres[temp_ad] = index

    def accuracyOOB(self, labeledSet):
        """
        Accuracy par la méthode OOB. Il faut que labeledSet soit le même utilisé dans train.
        """
        tBar = 0
        for arbre in self.arbres:
            index = self.arbres[arbre]
            ti = 0
            Ti = np.setdiff1d(np.arange(labeledSet.size()), index)
            for i in Ti:
                if self.predict(labeledSet.getX(i)) == labeledSet.getY(i):
                    ti += 1
            ti /= Ti.size
            tBar += ti
        tBar /= self.nb_arbres
        return tBar*100
