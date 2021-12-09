import os, argparse, sys
from tqdm import tqdm

def run(args):

    INTERSECTIONLANG = args.intersectedLang


    HOME_SUBSETS = os.path.join(args.home, args.subsetsDirIn)

    HOME_OUT = os.path.join(args.home, args.subsetsDirOut)

    titlesIntersection = open(os.path.join(HOME_OUT, 'test/test.{}_titles.txt'.format(INTERSECTIONLANG)), 'r')
    listTitlesIntersection = [t.strip() for t in titlesIntersection.readlines()]
    titlesIntersection.seek(0)


    titles = open(os.path.join(HOME_SUBSETS, '{}_titles.txt'.format(INTERSECTIONLANG)), 'r')
    nbInstances = len(titles.readlines())

    print("{} titles #{}".format(INTERSECTIONLANG,nbInstances))

    pbar = tqdm(total=nbInstances)

    def load(source):
        dict = {}
        nb=0
        for t, s in zip(titles.readlines(), source.readlines()):
            nb+=1
            if t.strip() in listTitlesIntersection:
                dict[t.strip()] = s

            if (nb % 200) == 0:
                pbar.update(200)
        print("Dataset loaded.")
        return dict

    def getSumDoc(tint, pairDict):
        return pairDict[tint]

    todo = ['sum', 'doc']

    for pair in todo:
        print(" * Generating " + pair)
        if pair == 'sum':
            pairIntersection = open(
                os.path.join(HOME_OUT, 'test/test.{}_summaries.txt'.format(INTERSECTIONLANG)), 'w')
            summaries = open(os.path.join(HOME_SUBSETS, '{}_summaries.txt'.format(INTERSECTIONLANG)), 'r')
            titles.seek(0)
            pairDict = load(summaries)
            summaries.close()
        elif pair =='doc':
            pairIntersection = open(
                os.path.join(HOME_OUT, 'test/test.{}_documents.txt'.format(INTERSECTIONLANG)), 'w')
            documents = open(os.path.join(HOME_SUBSETS, '{}_documents.txt'.format(INTERSECTIONLANG)), 'r')
            titles.seek(0)
            pairDict = load(documents)
            documents.close()

        titlesIntersection.seek(0)
        count = 0
        for titleInt  in titlesIntersection.readlines():
            print(titleInt)
            sumdoc = getSumDoc(titleInt.strip(), pairDict)
            pairIntersection.write(sumdoc); pairIntersection.flush()

            count+=1

        pairIntersection.close()

    print("Done {} for {}".format("/".join(todo), INTERSECTIONLANG))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument('--subsetsDirIn', help="Directory of working language intersection subset --in files")
    parser.add_argument('--subsetsDirOut', help="Directory of working language intersection subset --out files")
    parser.add_argument('--intersectedLang', help="Language from the intersection for which to create the (document, lead) test pairs.", required=True)

    args = parser.parse_args(sys.argv[1:])

    run(args)