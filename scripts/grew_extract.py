""" This script is used to find occurrences of syntactic patterns in treebanks.\n
It can be used on the whole arborescence at once (beware it might take a while).\n
Results are stored by treebanks and language."""

import subprocess
import json
import glob
import os


def grew1file(fichier, pattern, constraint):
    """
    Extracts number of hits for a pattern for 1 conllu file

    @args
    - file to query
    - pattern to look for
    - constraint (used to filter the results of the query):
        no constraint : []
        only one "name_of_node" per sentence allowed : "name_of_node"

    @output
    - number of hits for this pattern

    @description :
    no constraint is used for the operator/operande types of patterns
    constraint is used for <s,v,o> patterns
    """
    outf = open("output_2.json", "w")
    print(fichier)
    subprocess.call(["grew", "grep", "-pattern", pattern, "-i", fichier], stdout=outf)
    results = json.loads(open("output_2.json").read())


    if not constraint:
        gov_dep, dep_gov = 0, 0
        for result in results:
            match = result["matching"]
            if int(match["operateur"]) < int(match["operande"]):
                dep_gov += 1
            elif int(match["operande"]) < int(match["operateur"]):
                gov_dep += 1
            else:
                raise Exception("Could not recognize this pattern : no constraint but no of the operateur-operande type...")
        return {"gov_dep" :gov_dep, "dep_gov":dep_gov}

    elif constraint == "OO":
        return len(results)

    # Each instance of a sent_id / verb_id combo can only appear once
    else:
        filtered_results = []
        for r in results:
            sent = r["sent_id"]
            v_id = r["matching"][constraint]
            unique = tuple([sent, v_id])
            if unique in filtered_results: continue
            filtered_results.append(unique)
        return len(filtered_results)


def grew1tb(UD_dir, treebank_dir, pattern, constraint):
    """
    Adds up the results for every connll in 1 treebank.

    @args
    - Directory where all UD_Treebanks are stored
    - Directory of the treebank to query
    - pattern
    - constraint (see grew1filefunc for a description)

    @output
    - number of hits for this query in 1 whole treebank
    """
    if not constraint:
        total_hits = {"dep_gov":0, "gov_dep":0}
    else:
        total_hits = 0
    for fichier in glob.glob(UD_dir+treebank_dir+"/*.conllu"):
        hits = grew1file(fichier, pattern, constraint)
        print("file\t{}\t{}".format(fichier, hits))
        if not constraint:
            total_hits["dep_gov"] += hits["dep_gov"]
            total_hits["gov_dep"] += hits["gov_dep"]
        else:
            total_hits+=hits
    print("\ntreebank\t{}\n".format(total_hits))
    return total_hits
#
# grew1tb("../../M2/cours/memoire/ressources/ud-treebanks-v2.1-subtypeless/", "UD_Ancient_Greek", "./Patterns/sov.pat", "V")
grew1tb("../../M2/cours/memoire/ressources/ud-treebanks-v2.1-subtypeless/", "UD_Naija", "./Patterns/OO.pat", "OO")

def grewUD(UD_dir, pattern, constraint):
    """
    Query the whole arborescence and look for a pattern in each treebank.
    Stores results based on language & treebank

    @args
    - Directory where all UD_Treebanks are stored
    - pattern
    - constraint (see grew1filefunc for a description)

    @output
    - write files summarizing the hits/treebank and hits/language

    """
    results = dict()
    directories = os.listdir(UD_dir)
    languages = [lang.split("_")[1].split("-")[0] for lang in directories]
    languages = sorted(set(languages))

    for lang in languages:
        lang_results = dict()
        treebanks = [folder for folder in directories if lang in folder]

        for treebank in treebanks:
            hits = grew1tb(UD_dir, treebank, pattern, constraint)
            lang_results[treebank]=hits
        results[lang]=lang_results

    print("global results\t{}\n".format(results))
    pattern_name = pattern.split("/")[-1].split('.')[0]

    # write normal
    # with open("../../M2/cours/memoire/memoire_outfiles/"+pattern_name+"_bylang.csv", "w") as outf, open("../../M2/cours/memoire/memoire_outfiles/"+pattern_name+"_bytb.csv", "w") as outff:
    #     outf.write("langue\toccurences\n")
    #     outff.write("treebank\toccurences\n")
    #     for lang in results:
    #         result = results[lang]
    #         total_hits = 0
    #         for tb in result:
    #             outff.write(tb+"\t"+str(result[tb])+"\n")
    #             total_hits +=result[tb]
    #         outf.write(lang+"\t"+str(total_hits)+"\n")

    # write without subtypes
    # with open("../../M2/cours/memoire/memoire_outfiles/"+pattern_name+"_bylang_subtypless.csv", "w") as outf, open("../../M2/cours/memoire/memoire_outfiles/"+pattern_name+"_bytb_subtypeless.csv", "w") as outff:
    #     outf.write("langue\tdep-gov\tgov-dep\n")
    #     outff.write("treebank\tdep-gov\tgov-dep\n")
    #     for lang in results:
    #         result = results[lang]
    #         dep_gov, gov_dep = 0, 0
    #         for tb in result:
    #             outff.write("\t".join([tb, str(result[tb]["dep_gov"]), str(result[tb]["gov_dep"])])+"\n")
    #             dep_gov += result[tb]["dep_gov"]
    #             gov_dep += result[tb]["gov_dep"]
    #         outf.write("\t".join([lang, str(dep_gov), str(gov_dep)])+"\n")

    # write oldfrench
    # with open("../../M2/cours/memoire/memoire_outfiles/"+pattern_name+"_bylang_subtypless_oldfrench.csv", "w") as outf, open("../../M2/cours/memoire/memoire_outfiles/"+pattern_name+"_bytb_subtypeless_oldfrench.csv", "w") as outff:
    #     outf.write("langue\t"+pattern+"\n")
    #     outff.write("treebank\t"+pattern+"\n")
    #     for lang in results:
    #         result = results[lang]
    #         hits = 0
    #         for tb in result:
    #             outff.write("\t".join([tb, str(result[tb])])+"\n")
    #             hits = result[tb]
    #         outf.write("\t".join([lang, str(hits)])+"\n")



####################### Actual script ##########################################


# Test query :
# grewUD("../../M2/cours/memoire/ressources/ud-treebanks-v2.1-sample/", "./Patterns/svo.pat", "V")


# <s,v,o> order queries :
# for pattern in ["./Patterns/ovs.pat", "./Patterns/osv.pat",]:
#     print("#PATTERN {}\n".format(pattern))
#     grewUD("../../M2/cours/memoire/ressources/ud-treebanks-v2.1-subtypeless/", pattern, "V")


# Cross-categorial harmony queries :
# for pattern in ["./Patterns/coherence/noun-adj.pat", "./Patterns/coherence/noun-adp.pat", "./Patterns/coherence/verb-aux.pat", "./Patterns/coherence/verb-obj.pat"]:
    # print("#PATTERN {}\n".format(pattern))
    # grewUD("../../M2/cours/memoire/ressources/UDD-subtypeless/", pattern, [])

# Double object pattern to prove it exists:
# pattern = "./Patterns/OO.pat"
# print("#PATTERN {}\n".format(pattern))
# grewUD("../../M2/cours/memoire/ressources/ud-treebanks-v2.1/", pattern, "OO")
