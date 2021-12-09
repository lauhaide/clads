import os, argparse, sys
import numpy as np

np.random.seed(44)

def run(args):

    DATASET_FOLDER = os.path.join(args.home, args.datasetDir)
    PREFIX_LIST = args.filePrefix
    SPLITS = args.splits

    MIN_LEAD_LEN = 20

    CLEAN_LIST=["</ref>", "</nowiki>", "<nowiki>", "</math>", "<noinclude>", "<onlyinclude>", "cdf =|", "</noinclude>", "</table>", "</small>"]

    def prepareSummary(s):
        """ prepare raw extracted summaries as used in the task, e.g. use only first abstract"""
        for clean in CLEAN_LIST:
            s = s.replace(clean, "")
        if s.strip().startswith('<p>'):
            s = s.strip()[3:].strip()
        paras = s.strip().split("<p>")
        ret = paras[0]
        i = 1
        while len(ret.split()) < MIN_LEAD_LEN and i < len(paras):
            print("merge next... " + paras[i])
            ret = " ".join([ret, paras[i]])
        return ret + "\n"

    for PREFIX in PREFIX_LIST:

        print("\t* Creating summaries for: {}".format(PREFIX))
        sum = ".{}_summaries_prep.txt".format(PREFIX)

        generate_files = {}
        for sp in SPLITS:
            generate_files[sp] = {}
            for f in [sum]:
                print("creating file... " +  os.path.join(DATASET_FOLDER, sp + f))
                generate_files[sp][f] = open(os.path.join(DATASET_FOLDER, sp + f), 'w')

        for currentSplit in SPLITS:
            sumEn = open(os.path.join(DATASET_FOLDER, '{}.{}_summaries.txt'.format(currentSplit, PREFIX)), 'r')
            titlesEn = open(os.path.join(DATASET_FOLDER, '{}.{}_titles.txt'.format(currentSplit, PREFIX)), 'r')
            titlesNb = len([1 for l in titlesEn.readlines()])

            for i in range(titlesNb):
                s = sumEn.readline()

                if len(s.strip().split()) <20:
                    print(i)
                    assert False
                s = prepareSummary(s)

                generate_files[currentSplit][sum].write(s)


        for sp in SPLITS:
            for f in [sum]:
                generate_files[sp][f].close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument("--datasetDir", type=str, default='', required=True,
                        help="Folder to the dataset to process.")
    parser.add_argument("--filePrefix", nargs='+', required=True,
                        help="To enumerate the list of file prefixes of the files to be processed.")
    parser.add_argument("--splits", nargs='+', required=True,
                        help="To enumerate the list of file prefixes of the files to be processed.")

    args = parser.parse_args(sys.argv[1:])

    run(args)

