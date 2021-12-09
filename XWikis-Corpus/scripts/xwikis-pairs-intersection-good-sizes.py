import os, argparse, sys
from tqdm import tqdm


def run(args):

    TARGET_LANG = args.smallestSetLang
    TO_SHORTEN_LANGS = args.toShortenLangs
    HOME_SUBSETS = os.path.join(args.home, args.subsetsDir)
    pivotFile = open(os.path.join(args.home, args.langPivotFile), 'r')
    pivotLang = args.pivotLang

    # total in pivot-file
    total_in = len([1 for l in pivotFile.readlines()])
    pivotFile.seek(0)
    pbar = tqdm(total=total_in)

    smallerTitleSubset = open(os.path.join(HOME_SUBSETS, '{}_{}_titles.txt'.format(TO_SHORTEN_LANGS, TARGET_LANG)), 'r')
    smallerTitleList = [l.strip() for l in smallerTitleSubset.readlines()]
    print("Smaller set title list: {}".format(len(smallerTitleList)))
    smallerTitleNewTitles = open(os.path.join(HOME_SUBSETS, '{}_{}_titles_shortened.txt'.format(TO_SHORTEN_LANGS, TARGET_LANG)), 'w')
    smallerTitleSubsetReduced = open(os.path.join(HOME_SUBSETS, '{}_{}_titles_missing_langs.txt'.format(TO_SHORTEN_LANGS, TARGET_LANG)), 'w')
    reducedSmaller = []

    print("Start -- reduce {} lang titles to those in {}".format(TO_SHORTEN_LANGS, TARGET_LANG))
    titlesMap0 = {}

    i=0
    newTitles0 = open(os.path.join(HOME_SUBSETS, '{}_titles_shortened.txt'.format(TO_SHORTEN_LANGS)), 'w')

    titlesGoodSize0 = open(os.path.join(HOME_SUBSETS, '{}_titles.txt'.format(TO_SHORTEN_LANGS)), 'r')
    smallerTitleList0 = [l.strip() for l in titlesGoodSize0.readlines()]


    for line in pivotFile.readlines():
        toShortenTitle0, toShortenTitle1, smallerTitle = None, None, None
        if (i % 200) == 0:
            pbar.update(200)
            newTitles0.flush()
            smallerTitleSubsetReduced.flush()
            smallerTitleNewTitles.flush()
        l = line.strip().split(" ||| ")
        if TO_SHORTEN_LANGS == pivotLang:
            toShortenTitle0 = l[0].strip()
        if TARGET_LANG == pivotLang:
            smallerTitle = l[0].strip()
        for lt in l[2:]:
            if lt.split('::')[0] == TO_SHORTEN_LANGS:
                toShortenTitle0 = lt.split('::')[1].strip()
            elif lt.split('::')[0] == TARGET_LANG:
                smallerTitle = lt.split('::')[1].strip()
        if smallerTitle in smallerTitleList:
            if not smallerTitle in titlesMap0.keys():
                # if not already seen
                # there are different entries from one lang mapped to same in other lang, e.g.
                # fr::courant électrique ==> de::Elektrischer Strom , de::Elektrische Stromstärke
                if toShortenTitle0 in smallerTitleList0 :
                    titlesMap0[smallerTitle] = toShortenTitle0
                    newTitles0.write(toShortenTitle0 + "\n")
                    smallerTitleNewTitles.write(smallerTitle + "\n")
                else:
                    failed0 = TO_SHORTEN_LANGS if toShortenTitle0 in smallerTitleList0 else ""
                    smallerTitleSubsetReduced.write(smallerTitle + "\n")
                    reducedSmaller.append(smallerTitle)
        i+=1

    newTitles0.close()
    smallerTitleSubsetReduced.close()
    smallerTitleNewTitles.close()

    print("done")
    print("Len of reduced titles: ", len(titlesMap0.values()))
    print("Excluded from smaller", len(reducedSmaller))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument('--subsetsDir', help="Language to find intersection.")
    parser.add_argument('--langPivotFile', help="Language links pivot file.", required=True)
    parser.add_argument('--pivotLang', help="Pivot language of language links file", required=True)
    parser.add_argument('--smallestSetLang', help="Language that has the less number of valid size doc-sum pairs.")
    parser.add_argument('--toShortenLangs', help="The other two languages in the title triplets.")

    args = parser.parse_args(sys.argv[1:])

    run(args)