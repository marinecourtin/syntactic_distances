# Fichiers annexe de mon mémoire de M2 "Mesures de distances syntaxiques entre langues à partir de treebanks"

## Scripts


##### Python

+ grew_extract.py : automatise l'extraction de motifs syntaxiques avec grew
+ grew_analysis.py : exploite les données précédemment extraites
+ plotting.py : visualisation des données sous formes de graphiques

##### R

+ grew_analysis.Rmd : autres types d'analyse des données en R

## Graphiques

##### Ordre des unités (sujet, verbe, objet)

+ heatmaps de similarité cosinus entre :
    + toutes les langues
    + tous les treebanks
    + les langues romanes
    + les treebanks de langue romane

+ visualisation 2D après Analyse en Composantes Principales (ACP)
    + toutes les langues
    + les langues romanes
    + un échantillon de langues

+ Barplot des préférences d'ordre de mot pour les langues romanes

+ Corrélation entre ordres de mots

##### Proportion des linéarisation dépendant-gouverneur pour les couples (objet/verbe, verbe/auxiliaire, adjectif/nom, nom/adposition)


+ heatmaps de similarité cosinus entre :
    + toutes les langues
    + tous les treebanks
    + les langues romanes
    + les treebanks de langue romane

+ Barplot des ratios de ces linéarisation pour les 4 motifs dans les langues romanes

+ Corrélation entre les ratios pour les 4 motifs

+ visualisation 2D après Analyse en Composantes Principales (ACP)
    + toutes les langues

##### Distribution des trigrammes d'étiquettes morpho-syntaxiques

+ heatmaps de divergence de Jensen-Shannon entre :
    + toutes les langues
    + tous les treebanks
    + les langues romanes


## Données

+ comptage pour chaque paramètre étudié
    + par langue
    + par treebank

+ matrices de distances pour chaque paramètre étudié
    + par langue
    + par treebank
