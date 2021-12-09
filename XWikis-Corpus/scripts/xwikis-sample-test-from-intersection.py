import os, argparse, sys
import numpy as np

np.random.seed(44)

def run(args):
    INTER_FOLDER = os.path.join(args.home, 'de-fr-en-cs.20-400.250-5000-per')

    LENGTHS_FILES = os.path.join(args.home, 'de-fr-en.20-400.250-5000-per')
    # create dictionary with (title, doc-len) pairs use english that is the largest,
    # take the files used to compute de-fr-en intersection
    tLenPairs = open(os.path.join(LENGTHS_FILES, 'en_titles.txt'), 'r')
    dLenPairs = open(os.path.join(LENGTHS_FILES, 'en_documents.txt'), 'r')
    tit_len={}
    for t, d in zip(tLenPairs.readlines(), dLenPairs.readlines()):
        tit_len[t.strip()] = len(d.strip().split())
    print("\t* Data for sizes loaded...")

    titlesEn = open(os.path.join(INTER_FOLDER, 'en_titles_shortened.txt'), 'r')
    titlesDe = open(os.path.join(INTER_FOLDER, 'de_titles_shortened.txt'), 'r')
    titlesFr = open(os.path.join(INTER_FOLDER, 'fr_titles_shortened.txt'), 'r')
    titlesCs = open(os.path.join(INTER_FOLDER, 'cs_titles_shortened.txt'), 'r')
    titlesAll = [(tEn, tDe, tFr, tCs, tit_len[tEn.strip()])
                        for tEn, tDe, tFr, tCs in zip(
                                titlesEn.readlines(),
                                titlesDe.readlines(),
                                titlesFr.readlines(),
                                titlesCs.readlines())]
    titlesEn.seek(0)
    titlesDe.seek(0)
    titlesFr.seek(0)
    titlesCs.seek(0)

    frtLenPairs = open(os.path.join(LENGTHS_FILES, 'fr_titles.txt'), 'r')
    frdLenPairs = open(os.path.join(LENGTHS_FILES, 'fr_documents.txt'), 'r')
    frtit_len={}
    for t, d in zip(frtLenPairs.readlines(), frdLenPairs.readlines()):
        frtit_len[t.strip()] = len(d.strip().split())
    print("\t* FR Data for sizes loaded...")

    lens = []
    for _, _, f, _, _ in titlesAll:
        if f.strip() in frtit_len.keys():
            lens.append(frtit_len[f.strip()])
    lens.sort()
    npLengths = np.array(lens[:7000])
    print("\nFrench Mean {}\nMin {}\nMax {}\nStd {}".format(npLengths.mean(), npLengths.min(), npLengths.max(), npLengths.std()))


    titlesAll.sort(key=lambda x: x[4])
    titlesAll = titlesAll[:7000] # discard top K as these are too long docs

    npLengths = np.array([x[4] for x in titlesAll])
    print("\nEnglish Mean {}\nMin {}\nMax {}\nStd {}".format(npLengths.mean(), npLengths.min(), npLengths.max(), npLengths.std()))

    print("About the sample: ")
    np.random.shuffle(titlesAll)
    titlesAll = titlesAll[:7000]
    npLengths = np.array([x[4] for x in titlesAll])
    print("\nSAMPLE English Mean {}\nMin {}\nMax {}\nStd {}".format(npLengths.mean(), npLengths.min(), npLengths.max(), npLengths.std()))

    testEn = open(os.path.join(INTER_FOLDER, 'test/test.en_titles.txt'), 'w')
    testDe = open(os.path.join(INTER_FOLDER, 'test/test.de_titles.txt'), 'w')
    testFr = open(os.path.join(INTER_FOLDER, 'test/test.fr_titles.txt'), 'w')
    testCs = open(os.path.join(INTER_FOLDER, 'test/test.cs_titles.txt'), 'w')

    print("taken: ", len(titlesAll))
    enTakenTitles = np.array([x[0] for x in titlesAll])

    for ten, tde, tfr, tcs in zip(titlesEn.readlines(), titlesDe.readlines(), titlesFr.readlines(), titlesCs.readlines()):
        if ten in enTakenTitles:
            testEn.write(ten)
            testDe.write(tde)
            testFr.write(tfr)
            testCs.write(tcs)


    testEn.close()
    testDe.close()
    testFr.close()
    testCs.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")

    args = parser.parse_args(sys.argv[1:])

    run(args)

