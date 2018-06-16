import pandas
import itertools
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import math
import seaborn as sns
from adjustText import adjust_text
import sys
np.random.seed(5)

# Parameters
byLang = False
patterns = ["verb-aux", "noun-adj", "verb-obj", "noun-adp"]

# Creating the table
count = 0
total_data = []
for pattern in patterns :

    if byLang:
        data = pandas.read_csv("../memoire_outfiles/"+pattern+"_bylang_subtypless.csv", sep="\t")
        # with open("../memoire_outfiles/word_order/wordorder_bylang_latex.txt", "w") as outf:
        #     outf.write(data.to_latex())
    else:
        data = pandas.read_csv("../memoire_outfiles/"+pattern+"_bytb_subtypeless.csv", sep="\t")

    if count == 0:
        items = data.ix[:,0].tolist()
        total_data.append(items)

    # DEP-GOV (GOV-DEP + DEP-GOV)
    ratio_gov_dep = list(data["dep-gov"] / (data["dep-gov"] +data["gov-dep"]))

    total_data.append(ratio_gov_dep)
    count +=1

my_data = pandas.DataFrame(total_data)




my_data = my_data.transpose()
# my_data = my_data.replace(['Old', 'North', 'Upper', 'Ancient'],["OldFrench", "NorthSami", "UpperSorbian", 'AncientGreek'])
my_data.columns = ["treebank"]+patterns
my_data = my_data.set_index("treebank")


# my_data.to_csv("../memoire_outfiles/coherence/coherence_bylang_subtypeless_witholdfrench.csv", sep="\t")
# sys.exit()


# If one pattern isn't here, drop the row
my_data = my_data.dropna()
# print(my_data)

# Get back the filtered languages
langues = my_data.index


# Corrections


# Reindex to numeric
my_data = my_data.reset_index()


# write to file
# with open("../memoire_outfiles/coherence/coherence_directionhead_bylang_latex_subtypeless_witholdfrench.txt", "w") as outf:
#     outf.write(my_data.set_index("treebank").to_latex())

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


cosine_filtered = pairwise_distance(my_data, cosine)

# languages = ["French", "Italian", "Spanish", "Portuguese", "Catalan", "Latin", "Romanian", "Galician", "Italian", "OldFrench"]
def filterLanguages(languages, data, keep=True):
    """
    Returns a partial dataFrame with rows filtered on languages

    @args
    - the
    """
    all_languages = list(data.index)

    if keep:
        to_drop = [lang for lang in all_languages if lang not in languages]
    else:
        to_drop = languages

    for l in to_drop:
        try:
            data = data.drop(l) #rows
            data = data.drop(l, 1) #columns
        except ValueError:
            continue
            # the lang was already filtered or not present
    return data

# print(cosine_filtered)
# cosine_filtered = filterLanguages(languages, cosine_filtered, True)
# print(cosine_filtered)
# cosine_filtered.to_csv("../memoire_outfiles/coherence/coherence_cosinedist_bytb_subtypeless_witholdfrench.csv", sep="\t", encoding="utf-8")
# sys.exit()




lang_w_treebankS = ["Ancient_Greek", "Arabic", "Chinese", "Czech", "Dutch", "English", "Portuguese", "Romanian", "Russian", "Slovenian", "Spanish", "Swedish", "Turkish", "French", "Finnish", "Galician", "German", "Hindi", "Italian", "Japanese", "Latin", "Norwegian"]
#
tmp = dict()
for lang in lang_w_treebankS:
    treebanks = []
    sum_dist = []
    for row in my_data.iterrows():

        langue = row[1].treebank.split("-")[0][3:]
        if lang == langue:
            treebanks.append(row[0])
    # print(treebanks)
    for id_1, id_2 in itertools.combinations(treebanks, 2):
        if id_1== 1 : continue
        l1, *x = list(my_data.iloc[id_1])
        l2, *y = list(my_data.iloc[id_2])
        x = np.array(x)
        y = np.array(y)

        result = cosine(x, y)
        # print(lang, id_1, id_2, result)
        sum_dist.append(result)
    if sum_dist:
        moyenne = sum(sum_dist)/len(sum_dist)
        print(lang, sum_dist, moyenne)
        tmp["langue"] = tmp.get("langue", [])+[lang]
        tmp["distance-moyenne"] = tmp.get("distance-moyenne", [])+[moyenne]

table = pandas.DataFrame(data=tmp)
table = table.set_index("langue")

with open("../memoire_outfiles/coherence/coherence_distance_treebanksSameLang_subtypeless_witholdfrench_latex.txt", "w") as outf:
    outf.write(table.to_latex())


# Pour regarder à quel point les elts s'écartent de la moyenne
# ecart = list()
# for idx, row in my_data.iterrows():
#     vector = [elt for elt in row[1:] if not np.isnan(elt)]
#     nb = len(vector)
#     moyenne = sum(vector)/nb
#     somme_dist_moyenne_norm = sum([abs(elt-moyenne) for elt in vector])/nb
#     if nb == 4:
#         # print(vector, "\t", moyenne, "\t", somme_dist_moyenne_norm)
#         ecart.append(vector)
#     else:
#         print(idx, row)



# Clustering


from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage


mapping = dict([(idx, lang) for (idx, lang) in enumerate(cosine_filtered.index.tolist())])

def llf(idx):
    return mapping[idx]

def fancy_dendrogram(*args, **kwargs):
    """
    @source : https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    """
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

# The whole matrix is redundant, triangular is faster
condensed_dist = squareform(cosine_filtered)
linkage_matrix = linkage(condensed_dist, "ward")
#
dend = dendrogram(linkage_matrix, orientation="right", distance_sort="descending", leaf_label_func=llf)
#
plt.title("Clustering hiérarchique des langues d'après la proportion des linéarisations dépendant-gouverneur dans les catenas verb-aux, verb-obj, noun-adp, noun-adj")
plt.show()

sys.exit()
# print(len(ecart))
# #     ecart.append(somme_dist_moyenne_norm)
# my_data["deviation"] = pandas.Series(ecart)

# for x, item in zip(my_data["deviation"], my_data.ix[:,0]):
#     plt.scatter(x, 0)
# plt.xlim(0,1)
# plt.show()
import itertools

for pat1,pat2 in itertools.combinations(sorted(patterns), 2):
    items = list()
    for x,y,item in zip(my_data[pat1].tolist(),my_data[pat2].tolist(),my_data.ix[:,0].tolist()):
        if np.isnan(x) or np.isnan(y):continue
        plt.scatter(x,y)
        items.append(plt.text(x,y,item))
    adjust_text(items, only_move={'text':'xy'},arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
    plt.title("Ratio of governor final {} on Ratio of governor final {}".format(pat1, pat2))
    plt.savefig("../memoire_outfiles/plots/{}_on_{}.png".format(pat1,pat2), bbox_inches='tight')
#
# # items = list()
# for x,y,item in zip(my_data["verb-aux"].astype('float64').tolist(),my_data["noun-adj"].astype('float64').tolist(),my_data.ix[:,0].tolist()):
#     if np.isnan(x) or np.isnan(y): continue
#     plt.scatter(x,y)
    # print(x, y)
#     items.append(plt.text(x,y,item))
# adjust_text(items, only_move={'text':'xy'},arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
#
# plt.title("Ratio of governor final {} on Ratio of governor final {}".format("-aux", "verb-obj"))
# plt.show()
#
# print(my_data["noun-adj"])
# a = (sns.jointplot(x=my_data["noun-adj"].astype('float64'), y=my_data["verb-obj"].astype('float64'), color="skyblue", marginal_kws=dict(bins=15, rug=True), s=40).plot_joint(sns.kdeplot, zorder=0, n_levels=7))
# plt.show()
