import get_data
import glob
import os
from collections import Counter
import pandas
import itertools
import sys
from scipy.spatial.distance import cosine
import numpy as np
import dit # JSD
from math import sqrt



pos = ["ADP", "ADJ", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB"]
all_3 = list(itertools.product(pos, repeat=3))

def language_3pos(dirpath, ignore_pos, byLang=False):

    if not byLang:
        files = [fichier for fichier in glob.glob(os.path.join(dirpath, '*.conllu'))]
        fi_name = files[0].split("/")[-1]
        lang = fi_name.split("-")[0]
        if "_" in lang:
            lang, treebank = lang.split("_")
        else:
            treebank = "original"
    else:
        files = dirpath
        fi_name = files[0].split("/")[-2]

        lang = fi_name.split("-")[0]



    trigrammes = []
    trigrammes.extend(all_3) # adding 1 fake example of each possible trigramme

    for fichier in files:
        with open(fichier, "r") as inf:
            sentence = []
            for line in inf:
                try:
                    num, _, _, pos, _, _, idgov, func, _, _ = line.split("\t")
                    if pos in ignore_pos: continue
                    sentence.append(pos)
                except ValueError:
                    if line[0] == "#": # on recommence
                        # sentence.extend(["@@", "@"])
                        if len(sentence) > 4:
                            tri = list(zip(sentence, sentence[1:], sentence[2:]))
                            trigrammes.extend(tri)
                        sentence = []

    freq_3pos = Counter(trigrammes)



    # for key in freq_3pos:
    #     print(key, freq_3pos[key])

    nb_3pos = sum([freq_3pos[key] for key in freq_3pos])
    if not byLang:
        return [(lang+"_"+treebank, k1+"_"+k2+"_"+k3, freq_3pos[(k1, k2, k3)]/nb_3pos) for (k1, k2, k3) in sorted(freq_3pos, key=lambda x:x[2], reverse=True)]
    else:
        return [(lang, k1+"_"+k2+"_"+k3, freq_3pos[(k1, k2, k3)]/nb_3pos) for (k1, k2, k3) in sorted(freq_3pos, key=lambda x:x[2], reverse=True)]

def write_all_3pos(ud_dir, ignore_pos, byLang=False):
    """
    Computes a given stat for all ud treebanks.

    @args:
    - name of the stat function
    - dict of @args : ud_dir, ignore_rel, ignore_pos, with_pos
    """
    tri_pos = list()

    if not byLang:
        for folder in os.listdir(ud_dir):
            treebank_dir = os.path.join(ud_dir, folder)
            print("=== dir", treebank_dir)

            result = language_3pos(treebank_dir, ignore_pos)

            tri_pos.extend(result)
            print("add tri_pos for", folder)

        return pandas.DataFrame(tri_pos, columns=["treebank", "3pos", "relative_frequency"])

    else:
        folders = os.listdir(ud_dir)
        langues = set([f.split("-")[0] for f in folders])
        # print(folders, langues)
        for lang in langues:
            the_files = []
            the_folders = [f for f in folders if f.startswith(lang)]
            for f in the_folders:
                the_files.extend([fichier for fichier in glob.glob(os.path.join(ud_dir, f, '*.conllu'))])
            result = language_3pos(the_files, ignore_pos, True)

            tri_pos.extend(result)
            print("add tri_pos for", lang)

        return pandas.DataFrame(tri_pos, columns=["langue", "3pos", "relative_frequency"])


### Writing the data

byLang = True

# if byLang:
#     df = write_all_3pos("/home/marine/Dropbox/TAL/M2/cours/memoire/ressources/UDD-subtypeless", ["PUNCT", "X", "SYM", "_"], True)
#     df.to_csv("../memoire_outfiles/trigrammes-pos/frequency_tripos_oldfrench.csv", sep='\t', encoding='utf-8')
#
# if not byLang:
#     df = write_all_3pos("/home/marine/Dropbox/TAL/M2/cours/memoire/ressources/UDD-subtypeless", ["PUNCT", "X", "SYM", "_"])
#     df.to_csv("../memoire_outfiles/trigrammes-pos/frequency_tripos_oldfrench_tb.csv", sep='\t', encoding='utf-8')
#

### Reading the data
byLang=False
if byLang:
    data = pandas.read_csv("../memoire_outfiles/trigrammes-pos/frequency_tripos_bylang.csv", sep="\t")
    data = data.set_index("langue")
else:
    data = pandas.read_csv("../memoire_outfiles/trigrammes-pos/frequency_tripos_bytb.csv", sep="\t")
    data = data.set_index("treebank")



### Pivot table (long to wide)
data = data.pivot_table(values='relative_frequency', index='treebank', columns='3pos')




def pairwise_distance(data, distance):
    """
    Computes a mesure of distance (cosine, euclidean...) for all combinations of treebanks in a dataframe

    @args :
    - the dataframe containing the information for each tb on a row (here -dep-> freq)

    @output:
    - a dataframe with the euclidean distances
    """
    dist = dict()

    for id_1, id_2 in itertools.product(data.index, repeat=2): # product AA AB BA BB
    # for id_1, id_2 in itertools.combinations(data.index, 2): # AB

        l1, *x = list(data.iloc[id_1])
        l2, *y = list(data.iloc[id_2])


        x = np.array(x)
        y = np.array(y)
        result = distance(x, y)

        if l1 in dist:
            dist[l1][l2] = result
        else:
            dist[l1] = dict()
            dist[l1][l2] = result

    df = pandas.DataFrame(dist)


    return df


def JSD_pos_dataframe(matrix):
    """
    Computes the Jensen Shannon Divergence from the pos distributions

    @args :
    - the statfile with pos distribution

    @output :
    - a dataframe with distance for every permutations (including fr/fr)
    """
    test = dict()

    langues = matrix.index.tolist()
    pos = matrix.columns.tolist()


    for l1, l2 in itertools.product(sorted(langues), repeat=2): # AA AB BA BB
        distrib_1 = matrix.loc[l1].tolist()
        distrib_2 = matrix.loc[l2].tolist()
        X = dit.ScalarDistribution(pos, distrib_1)
        Y = dit.ScalarDistribution(pos, distrib_2)
        JS = dit.divergences.jensen_shannon_divergence([X, Y])
        # print("l1 {}\tl2 {}\t Jensen-Shannon Divergence {}\n".format(l1, l2, JS))

        if l1 in test:
            test[l1][l2] = JS
        else:
            test[l1] = dict()
            test[l1][l2] = JS
    #
    df = pandas.DataFrame(test)
    return df

## JSD distance
dist_matrix_jsd = JSD_pos_dataframe(data)
# print(dist_matrix_jsd)
dist_matrix_jsd.to_csv("../memoire_outfiles/trigrammes-pos/jsd_distance_3pos_bytb.csv", sep="\t", encoding="utf-8")
sys.exit()

## Cosine similarity

# Store labelled index
langues = data.index

# Reindex to numeric
data = data.reset_index()
dist_matrix = pairwise_distance(data, cosine)
