import sys
import itertools
import numpy as np
import pandas
import math
# from sklearn.preprocessing import scale
from scipy.spatial.distance import cosine
# from numpy.linalg import norm
from sklearn import preprocessing
import matplotlib.pyplot as plt



def euclidean_distance(x, y):
    """
    Computes the euclidean distance between two vectors
    Not currently used

    @args:
    - two lists (probabilities of -func-> for ex)

    @output:
    - euclidean distance between 2 vectors

    @beware
    - vectors aren't normalized
    """
    dist = 0
    for i in range(len(x)):
        add = (x[i]-y[i])**2
        dist += add
    dist = math.sqrt(dist)
    return dist

def pairwise_distance(data, distance):
    """
    Computes a mesure of distance (cosine, euclidean...) for all combinations of languages/treebanks in a dataframe

    @args :
    - the dataframe containing the information for each language/treebank on a row

    @output:
    - a distance matrix (dataframe type)
    """
    dist = dict()

    count = 0
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

def filterLanguages(languages, data, keep=True):
    """
    Filters a distance matrix based on languages

    @args
    - a list of languages
    - the DataFrame
    - whether we want to keep these lang (True) or get rid of them (False)

    @output
    - filtered distance matrix (useful for plotting small heatmaps)
    """
    all_languages = list(data.index)

    if keep:
        to_drop = [lang for lang in all_languages if lang not in languages]
        print("to drop", to_drop)
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


langNames={'el': 'Greek', 'en': 'English', 'zh': 'Chinese', 'ug': 'Uyghur', 'vi': 'Vietnamese', 'ca': 'Catalan', 'it': 'Italian', 'eu': 'Basque', 'ar': 'Arabic', 'ga': 'Irish', 'cs': 'Czech', 'et': 'Estonian', 'gl': 'Galician', 'id': 'Indonesian', 'es': 'Spanish', 'ru': 'Russian', 'nl': 'Dutch', 'pt': 'Portuguese', 'no': 'Norwegian', 'tr': 'Turkish', 'lv': 'Latvian', 'lt': 'Lithuanian', 'grc': 'AncientGreek', 'got': 'Gothic', 'ro': 'Romanian', 'pl': 'Polish', 'ta': 'Tamil', 'be': 'Belarusian', 'fr': 'French', 'bg': 'Bulgarian', 'hr': 'Croatian', 'de': 'German', 'hu': 'Hungarian', 'fa': 'Persian', 'hi': 'Hindi', 'fi': 'Finnish', 'da': 'Danish', 'ja': 'Japanese', 'he': 'Hebrew', 'cop': 'Coptic', 'kk': 'Kazakh', 'la': 'Latin', 'ko': 'Korean', 'sv': 'Swedish', 'ur': 'Urdu', 'sk': 'Slovak', 'cu': 'OldChurchSlavonic', 'uk': 'Ukrainian', 'sl': 'Slovenian', 'sa': 'Sanskrit', 'yue':'Cantonese',
'af':'Afrikaans', 'swl':'SwedishSign', 'kmr':'Kurmanji', 'sme':'NorthSami', 'hsb':'UpperSorbian', 'bxr':'Buryat', 'te':'Telugu', 'sr':'Serbian', 'mr':'Marathi',  'pcm':'Naija', 'mt':'Maltese', 'fro':'OldFrench', 'th':'Thai',
'am':'Amharic', 'myv': 'Erzya', 'fo':'Faroese', 'tl': 'Tagalog', 'bm': 'Bambara', 'br': 'Breton', 'hy':'Armenian', 'kpv':'Komi'
	   }



np.random.seed(5)

# Parameters
byLang = True



# Read data
if byLang:
    data = pandas.read_csv("../memoire_outfiles/word_order/wordorder_bylang_subtypeless_witholdfrench.csv", sep="\t")
    # data = pandas.read_csv("../memoire_outfiles/word_order/wordorder_bylang.csv", sep="\t")

    data = data.set_index("langue")
    # with open("../memoire_outfiles/word_order/wordorder_bylang_latex.txt", "w") as outf:
    #     outf.write(data.to_latex())
else:
    data = pandas.read_csv("../memoire_outfiles/word_order/wordorder_bytb_subtypeless_witholdfrench.csv", sep="\t")

    # data = pandas.read_csv("../memoire_outfiles/word_order/wordorder_bytb.csv", sep="\t")
    data = data.set_index("treebank")
    # with open("../memoire_outfiles/word_order/wordorder_bytb_latex.txt", "w") as outf:
    #     outf.write(data.to_latex())


# Convert csv with metadata to laTEX
# metadata = pandas.read_csv("../memoire_outfiles/ud_general_table.csv", sep="\t")
# metadata = metadata.set_index("treebank")
# metadata = metadata.drop(["nb_tokens", "nb_words", "nb_sentences", "lemmas", "features", "origin", "license"], axis=1)
# for i, row in metadata.iterrows():
#     row[""]
# with open("../memoire_outfiles/word_order/metadata_bytb_latex.txt", "w") as outf:
#     outf.write(metadata.to_latex())
# sys.exit()


# Normalize data (relative frequency of type rather than count)
columns = ["svo_rel", "sov_rel", "ovs_rel", "osv_rel", "vos_rel", "vso_rel"]
normalized_data = []

for idx, vector in data.iterrows():
    total = vector.sum()
    vector_normalized = [value/total for value in vector.tolist()]
    row = pandas.Series([idx]+vector_normalized)
    normalized_data.append(row)



# normalized_data = preprocessing.normalize(normalized_data, norm='l2') # if we want to normalize to norm = 1
normalized_data = pandas.DataFrame(normalized_data)

if byLang:
    indexName = "langue"
else:
    indexName = "treebank"


normalized_data.columns = [indexName]+columns


# write results to files
# normalized_data.to_csv("../memoire_outfiles/word_order/wordorder_proportions_subtypeless_bytb_witholdfrench.csv", sep="\t", encoding="utf-8")
# normalized_data.to_csv("../memoire_outfiles/word_order/wordorder_proportions_bytb.csv", sep="\t", encoding="utf-8")
# sys.exit()

# correct names
# normalized_data = normalized_data.replace(['Old', 'North', 'Upper', 'Ancient'],["OldFrench", "NorthSami", "UpperSorbian", 'AncientGreek'])

# Distance matrix
cosine_filtered = pairwise_distance(normalized_data, cosine)


# filter based on these lists
languages = ["French", "Italian", "Spanish", "Portuguese", "Catalan", "Latin", "Romanian", "Galician", "Italian", "OldFrench"]
# languages = ["Hebrew", "Arabic", "Coptic", "Japanese", "Korean", "Buryat", "Urdu", "Sanskrit", "Marathi", "Kurmanji", "Persian", "AncientGreek", "Greek", "English", "Naija", "Chinese", "French", "OldFrench", "Latin", "Cantonese", "Vietnamese", "Turkish", "Kazakh", "Uyghur", "Telugu", "Tamil", "Irish", "NorthSami", "Finnish"]
# languages = ["Japanese", "Telugu", "Korean", "Kazakh", "Uyghur", "Turkish", "Marathi"]
# cosine_filtered = filterLanguages(languages, cosine_filtered)
# print(cosine_filtered)
# cosine_filtered.to_csv("../memoire_outfiles/word_order/wordorder_cosinedist_bytb_subtypeless_witholdfrench_rom.csv", sep="\t", encoding="utf-8")



# print(cosine_filtered.idxmin(axis=1))

### Distance moyenne entre treebanks d'une même langue
normalized_data = pandas.read_csv("../memoire_outfiles/trigrammes-pos/jsd_distance_3pos_bytb.csv", sep="\t", index_col=0)
# normalized_data = normalized_data.set_index(normalized_data.ix[:,0])
print(normalized_data)
lang_w_treebankS = ["ar", "zh", "cs", "en", "pt", "ro", "ru", "sl", "sv", "nl", "la", "it", "hi", 'grc', "fr", "gl", "es"]
# lang_w_treebankS = ["Ancient_Greek", "Arabic", "Chinese", "Czech", "Dutch", "English", "Portuguese", "Romanian", "Russian", "Slovenian", "Spanish", "Swedish", "Turkish", "French", "Finnish", "Galician", "German", "Hindi", "Italian", "Japanese", "Latin", "Norwegian"]
# #
tmp = dict()
for lang in lang_w_treebankS:
    treebanks = []
    sum_dist = []
    for row in normalized_data.iterrows():
        # print(row[1][0])
        treebank = row[0]
        lang = row[0].split("_")[0]
        if lang == langue:
            treebanks.append(row[1][0])
            print(lang, treebanks)
    for id_1, id_2 in itertools.combinations(treebanks, 2):
        print(id_1, id_2)
        print(data.index)
        l1, *x = list(data.loc[langNames.get(id_1)])
        l2, *y = list(data.loc[langNames.get(id_2)])
        x = np.array(x)
        y = np.array(y)
        result = cosine(x, y)
        sum_dist.append(result)
        print(id_1, id_2, result)

    moyenne = sum(sum_dist)/len(sum_dist)
    print(lang, sum_dist, moyenne)
    # if lang == "Ancient_Greek":
    #     moyenne = 0.13259270744202833
    tmp["langue"] = tmp.get("langue", [])+[lang]
    tmp["distance-moyenne"] = tmp.get("distance-moyenne", [])+[moyenne]
#
# table = pandas.DataFrame(data=tmp)
# table = table.set_index("langue")
# print(table)
with open("../memoire_outfiles/trigrammes-pos/distance_treebanksSameLang_2_latex.txt", "w") as outf:
    outf.write(table.to_latex())
sys.exit()

# print(normalized_data.iterrows)

# cosine_filtered = filterLanguages(languages, cosine_distances, True)
#
#
# cosine_filtered.to_csv("../memoire_outfiles/word_order/wordorder_cosinedist_bylang_2.csv", sep="\t", encoding="utf-8")
# sys.exit()
# cosine_filtered.to_csv("../memoire_outfiles/word_order/wordorder_cosinedist_bytb.csv", sep="\t", encoding="utf-8")


## Clustering

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage


mapping = dict([(idx, lang) for (idx, lang) in enumerate(normalized_data.langue.tolist())])

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
# condensed_dist = squareform(cosine_filtered)
# linkage_matrix = linkage(condensed_dist, "ward")
#
# # dend = dendrogram(linkage_matrix, orientation="right", distance_sort="descending", truncate_mode = 'lastp', p=9)
#
# # plt.title("Clustering hiérarchique des langues d'après une repréntation vectorielle de l'ordre des unités <s,v,o>")
# # plt.show()
#
# print(h)
# print(cosine_filtered)
#
# fig, ax = plt.subplots()
# dend = dendrogram(linkage_matrix, orientation="right", distance_sort="descending", leaf_label_func=llf, )
# threshold = 0.005 #for hline
# ax.axvline(threshold, c='k')
# plt.show()
# sys.exit()
#
# # plt.savefig("../memoire_outfiles/plots/word_order/wordordercluster_bytb.png")
# # from scipy.cluster.hierarchy import cut_tree
# # membership = list(cut_tree(linkage_matrix, n_clusters = 7))
# #
# # with open("../memoire_outfiles/word_order/h_clustering_ward_cosine_7.csv", "w") as outf:
# #     outf.write("langue\tcluster\n")
# #     for idx, lang in enumerate(normalized_data.langue.tolist()):
#         # print(lang, membership[idx][0])
#         outf.write(lang+"\t"+str(membership[idx][0])+"\n")


# Dendro + heatmap
import seaborn as sns

# OK now we can compare our individuals. But how do you determine the similarity between 2 cars?
# Several way to calculate that. the 2 most common ways are: correlation and euclidean distance?
# sns.clustermap(cosine_distances, metric="cosine", standard_scale=1)
# plt.show()


# corr = normalized_data.corr()

# Pattern correlation
# sns.heatmap(corr,
#         xticklabels=corr.columns,
#         yticklabels=corr.columns)
# plt.savefig('../memoire_outfiles/plots/word_order/correlation_between_patterns.png', bbox_inches='tight')

# print(normalized_data["sov_rel"])

# sns.jointplot(x=normalized_data["sov_rel"], y=normalized_data["ovs_rel"], kind='scatter', marginal_kws=dict(bins=30, rug=True))
# sns.jointplot(x=normalized_data["sov_rel"], y=normalized_data["ovs_rel"], kind='scatter', marginal_kws=dict(rug=True, bins=0))
# plt.show()

cosine_filtered = pandas.read_csv("../memoire_outfiles/coherence/coherence_cosinedist_bylang_subtypeless_witholdfrench.csv", sep="\t", index_col=0)
# print(cosine_filtered)
# cosine_filtered = cosine_filtered.set_index([[0]])
# cosine_filtered = pandas.read_csv("../memoire_outfiles/word_order/wordorder_cosinedist_bylang_subtypeless_witholdfrench.csv", sep="\t")

print(cosine_filtered)

# Multi-dimensional scaling
# print(cosine_filtered)
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt, mpld3

from adjustText import adjust_text

langs = list(cosine_filtered.index)

rev_mapping = dict([(lang, idx) for (idx, lang) in enumerate(langs)])

# mds = MDS(n_components=2, dissimilarity="precomputed", metric=True, verbose=3)
pca = PCA(n_components=2)
pca.fit(cosine_filtered)

variance = sum(pca.explained_variance_ratio_[:2])

X_r = pca.transform(cosine_filtered)

# Eigenvectors
print(pca.components_)
# result = mds.fit(cosine_filtered)
# X_r = result.embedding_
# stress = result.stress_
# normalized_stress = np.sqrt(stress / ((cosine_filtered.as_matrix().ravel() ** 2).sum() / 2))

figure = plt.figure()
xs, ys = X_r[:,0], X_r[:, 1]
# texts =[]
# for x,y,lang in zip(xs,ys,langs):
#     plt.scatter(x,y)
#     texts.append(plt.text(x,y,lang))
#
#
#
# # print(texts)
# # Adjust text labels so they don't overlap
# adjust_text(texts, only_move={'text':'xy'},arrowprops=dict(arrowstyle="->", color='black', lw=0.5))
#
# # Plot original variables in this vector space
# # ex = (1,0,0,0,0,0)
# import json
# plt.title("Positionnement bidimensionnel des langues romanes d'après une représentation vectorielle de l'ordre des unités <s,v,o>. Quantity of variance explained {}".format(variance))
# # figure.set_dpi(100)
# # figure.set_size_inches(14, 13)
# json.dumps(figure.tolist())
# mpld3.show()
# # figure.savefig("../memoire_outfiles/plots/word_order/pca_romancelanguages_subtypeless_witholdfrench.svg", bbox_inches='tight')
# plt.show()
# # plt.savefig('../memoire_outfiles/plots/word_order/wordorder_romance_mds.png', bbox_inches='tight')
# sys.exit()
# #
# # 3D mds
# from mpl_toolkits.mplot3d import Axes3D # 3D MDS
# import plotly.plotly as py
# import plotly.graph_objs as go
#
#
# mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
# pos = mds.fit_transform(cosine_distances)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# getRidOf =["Irish", "Arabic"]
#
#
# for l in getRidOf:
#     index = rev_mapping[l]
#     np.delete(pos, index)
# ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
#
# langs = [lang for lang in langs if lang not in getRidOf]
# for x, y, z, lang in zip(pos[:, 0], pos[:, 1], pos[:, 2], langs):
#     ax.text(x, y, z, lang)
#
#
# plt.show()

from matplotlib import gridspec
langnameGroup={"AncientGreek":"Indo-European", "Arabic":"Semitic", "Basque":"isolate", "Belarusian":"Indo-European-Baltoslavic", "Bulgarian":"Indo-European-Baltoslavic", "Cantonese":"Sino-Austronesian", "Catalan":"Indo-European-Romance", "Chinese":"Sino-Austronesian", "Coptic":"Afroasiatic", "Croatian":"Indo-European-Baltoslavic", "Czech":"Indo-European-Baltoslavic", "Danish":"Indo-European-Germanic", "Dutch":"Indo-European-Germanic", "English":"Indo-European-Germanic", "Estonian":"Agglutinating", "Finnish":"Agglutinating", "French":"Indo-European-Romance", "Galician":"Indo-European-Romance", "German":"Indo-European-Germanic", "Gothic":"Indo-European-Germanic", "Greek":"Indo-European", "Hebrew":"Semitic", "Hindi":"Indo-European", "Hungarian":"Agglutinating", "Indonesian":"Sino-Austronesian", "Irish":"Indo-European", "Italian":"Indo-European-Romance", "Japanese":"Agglutinating", "Kazakh":"Agglutinating", "Korean":"Agglutinating", "Latin":"Indo-European-Romance", "Latvian":"Indo-European-Baltoslavic", "Lithuanian":"Indo-European-Baltoslavic", "Norwegian":"Indo-European-Germanic", "OldChurchSlavonic":"Indo-European-Baltoslavic", "Persian":"Indo-European", "Polish":"Indo-European-Baltoslavic", "Portuguese":"Indo-European-Romance", "Romanian":"Indo-European-Romance", "Russian":"Indo-European-Baltoslavic", "Sanskrit":"Indo-European", "Slovak":"Indo-European-Baltoslavic", "Slovenian":"Indo-European-Baltoslavic", "Spanish":"Indo-European-Romance", "Swedish":"Indo-European-Germanic", "Tamil":"Dravidian", "Turkish":"Agglutinating", "Ukrainian":"Indo-European-Baltoslavic", "Urdu":"Indo-European", "Uyghur":"Agglutinating", "Vietnamese":"Sino-Austronesian",'Afrikaans':'Indo-European-Germanic', 'SwedishSign':'Indo-European-Germanic', 'Kurmanji':'Indo-European', 'NorthSami':'Agglutinating', 'UpperSorbian':"Indo-European-Baltoslavic", 'Buryat':'Agglutinating', 'Telugu':'Dravidian', 'Serbian':"Indo-European-Baltoslavic", 'Marathi':'Indo-European','Naija':"Indo-European-Germanic", "OldFrench":"Indo-European-Romance", "Maltese":"Semitic", "Thai":"Sino-Austronesian","Amharic":"Afroasiatic", 'Erzya': 'Agglutinating', 'Faroese':"Indo-European-Germanic", 'Tagalog':"Sino-Austronesian", 'Bambara':'Niger-Congo', 'Breton':"Indo-European", 'Armenian':"Indo-European", 'Komi': 'Agglutinating'}
groupColors={"Indo-European-Romance":'brown',"Indo-European-Baltoslavic":'purple',"Indo-European-Germanic":'olive',"Indo-European":'royalBlue',"Sino-Austronesian":'limeGreen', "Agglutinating":'red'}
groupMarkers={"Indo-European-Romance":'<',"Indo-European-Baltoslavic":'^',"Indo-European-Germanic":'v',"Indo-European":'>',"Sino-Austronesian":'s', "Agglutinating":'+'}

col1 = pandas.Series(xs)
col2 = pandas.Series(ys)

c=[groupColors.get(langnameGroup[label],'k') for label in langs]
m=[groupMarkers.get(langnameGroup[label],'o') for label in langs]

fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 25], height_ratios=[25, 1])
ax = plt.subplot(gs[1])


for xx, yy, cc, mm in zip(col1, col2, c, m):
    # if cc not in ["purple", "limeGreen", "olive"]:continue
    ax.scatter(xx, yy, marker=mm, c=cc)
# aa.scatter([0.5 for _ in col1], col2, c=c, alpha=0.5)
# bb.scatter(col1, [0.5 for _ in col2], c=c, alpha=0.5)

texts=[]
for label, x, y in zip(langs, col1, col2):
    # if groupColors.get(langnameGroup[label],'k') not in  ["purple", "limeGreen", "olive"]:continue
    texts+=[ax.text(x, y, label, color=groupColors.get(langnameGroup[label],'k'), fontsize=8)] # for adjustText

adjust_text(texts, col1, col2, ax=ax, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))

# plt.save
plt.title("Positionnement bidimensionnel des langues selon la proportion des linéarisations dépendant-gouverneur dans les couples verbe-objet, auxiliaire-verbe, nom-adjectif et adposition-nom. Quantité de variation expliquée {}".format(round(variance, 4)))
plt.savefig("../memoire_outfiles/plots/coherence/pca_languages_subtypeless_witholdfrench.svg", bbox_inches='tight')
