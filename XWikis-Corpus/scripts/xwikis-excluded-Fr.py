import os, argparse, sys

def run(args):

    def loadFrPivot(file):
        """Builds a map of titles from one language to another.
        The dictionary keys correspond the title in language X that we want to map to Fr.
        The dictionary values contain the equivalent title in Fr"""
        ret = {}
        f = open(file, 'r')
        for line in f.readlines():
            l = line.strip().split(" ||| ")
            for lt in l:
                if lt.split('::')[0] == 'fr':
                    ret[l[0].strip()] =  lt.split('::')[1]
        f.close()
        return  ret

    # Collects the French titles used across 4-lang intersection and pairs datasets
    FrTitlesPairs = set()

    FrFiles = ['cs-fr.20-400.250-5000-per/final/cs_fr_tgt_titles.txt',
               'cs-fr.20-400.250-5000-per/final/fr_cs_src_titles.txt',
               'fr-en.20-400.250-5000-per/final/en_fr_tgt_titles.txt',
               'fr-en.20-400.250-5000-per/final/fr_en_src_titles.txt',
               'de-fr.20-400.250-5000-per/final/de_fr_tgt_titles.txt',
               'de-fr.20-400.250-5000-per/final/fr_de_src_titles.txt']

    for file in FrFiles:
        f = open(os.path.join(args.home, file), 'r')
        for l in f.readlines():
            FrTitlesPairs.add(l.strip())

    print(" * French as pair language: #{}".format(len(FrTitlesPairs)))

    DeFiles = ['de-en.20-400.250-5000-per/final/de_en_src_titles.txt',
               'de-en.20-400.250-5000-per/final/en_de_tgt_titles.txt',
               'de-cs.20-400.250-5000-per/final/cs_de_tgt_titles.txt',
               'de-cs.20-400.250-5000-per/final/de_cs_src_titles.txt']

    pivots = ['dewiki-20200620.urls.de-en.lang', 'dewiki-20200620.urls.de-cs.lang']

    # map titles in the De language to Fr  (de-en and de-cs are aligned to we take De titles as representatives)
    for p in pivots:
        deMap = loadFrPivot(os.path.join(args.home, p))
        for file in DeFiles:
            f = open(os.path.join(args.home, file), 'r')
            for l in f.readlines():
                if l.strip() in deMap.keys():
                    FrTitlesPairs.add(deMap[l.strip()])

    print(" * # French from language pairs {}".format(len(FrTitlesPairs)))

    CSFiles = ['cs-en.20-400.250-5000-per/final/cs_en_src_titles.txt',
               'cs-en.20-400.250-5000-per/final/en_cs_tgt_titles.txt']

    pivots = ['cswiki-20200620.urls.cs-en.lang']

    # map titles in the Cs language to Fr  (cs-en are aligned to we take Cs titles as representatives)
    for p in pivots:
        csMap = loadFrPivot(os.path.join(args.home, p))
        for file in CSFiles:
            f = open(os.path.join(args.home, file), 'r')
            for l in f.readlines():
                if l.strip() in csMap.keys():
                    FrTitlesPairs.add(csMap[l.strip()])

    print(" * # French from language pairs {}".format(len(FrTitlesPairs)))


    f = open(os.path.join(args.home, 'de-fr-en.20-400.250-5000-per/fr_titles_shortened.txt'), 'r')
    #note: the intersection of 4 langs is a subset of this one
    for l in f.readlines():
        FrTitlesPairs.add(l.strip())

    print(" * # French from full intersection (3 and 4 langs) {}".format(len(FrTitlesPairs)))

    out = open(os.path.join(args.home, 'fr-excl.20-400.250-5000-per/fr_titles_exclude.txt'), 'w')
    for t in FrTitlesPairs:
        out.write(t + "\n")
    out.close()
    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")

    args = parser.parse_args(sys.argv[1:])

    run(args)

