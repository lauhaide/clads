""" Makes final dataset and enforces instances order across files."""

import os, argparse, sys
import shutil
from tqdm import tqdm

def run(args):

    HOME_SUBSETS = os.path.join(args.home, args.subsetsDir)
    SRCLANG =  args.srcLang
    TGTLANG =  args.tgtLang

    instanceDetails = []
    for GENERATE_PART in [ 'tgt-sum', 'src-sum', 'src-doc']:

        print("Doing... {} ({}->{})".format(GENERATE_PART, SRCLANG, TGTLANG))

        # generating source part of the files' set
        if GENERATE_PART == 'src-sum':
            summariesSource = open(os.path.join(HOME_SUBSETS, 'final/{}_{}_src_summaries.txt'.format(SRCLANG,TGTLANG)), 'w')
            titlesIntersection = open(os.path.join(HOME_SUBSETS, '{}_titles_shortened.txt'.format(SRCLANG)), 'r')
            titles = open(os.path.join(HOME_SUBSETS, '{}_titles.txt'.format(SRCLANG)), 'r')
            summaries = open(os.path.join(HOME_SUBSETS, '{}_summaries.txt'.format(SRCLANG)), 'r')

        if GENERATE_PART == 'src-doc':
            documentsSource = open(os.path.join(HOME_SUBSETS, 'final/{}_{}_src_documents.txt'.format(SRCLANG,TGTLANG)), 'w')
            titlesIntersection = open(os.path.join(HOME_SUBSETS, '{}_titles_shortened.txt'.format(SRCLANG)), 'r')
            titles = open(os.path.join(HOME_SUBSETS, '{}_titles.txt'.format(SRCLANG)), 'r')
            documents = open(os.path.join(HOME_SUBSETS, '{}_documents.txt'.format(SRCLANG)), 'r')

        # generating the target part of the files' set
        if GENERATE_PART == 'tgt-sum':
            summariesTarget = open(os.path.join(HOME_SUBSETS, 'final/{}_{}_tgt_summaries.txt'.format(SRCLANG,TGTLANG)), 'w')
            titlesIntersection = open(os.path.join(HOME_SUBSETS, '{}_{}_titles_shortened.txt'.format(SRCLANG,TGTLANG)), 'r')
            titles = open(os.path.join(HOME_SUBSETS, '{}_{}_titles.txt'.format(SRCLANG, TGTLANG)), 'r')
            summaries = open(os.path.join(HOME_SUBSETS, '{}_{}_summaries.txt'.format(SRCLANG, TGTLANG)), 'r')



        listTitlesIntersection = [t.strip() for t in titlesIntersection.readlines()]
        titlesIntersection.seek(0)
        str = "#instances shortened: {}".format(len(listTitlesIntersection))
        print(str); instanceDetails.append(str)

        nbInstances = len(titles.readlines())
        titles.seek(0)
        str = "#instances on doc/sum pairs: {}".format(nbInstances)
        print(str); instanceDetails.append(str)

        pbar = tqdm(total=nbInstances)

        textDict = {}
        nb=0
        toread = summaries.readlines() if (GENERATE_PART in ['src-sum', 'tgt-sum']) else documents.readlines()
        for t, text in zip(titles.readlines(), toread):
            nb+=1
            if t.strip() in listTitlesIntersection:
                textDict[t.strip()] = text

            if (nb % 200) == 0:
                pbar.update(200)

        print("Dataset loaded.")


        def getSumDoc(tint):
            return textDict[tint]


        count = 0
        for titleInt  in titlesIntersection.readlines():

            print(titleInt)
            text = getSumDoc(titleInt.strip())

            if text:
                if GENERATE_PART == 'tgt-sum':
                    summariesTarget.write(text); summariesTarget.flush()
                if GENERATE_PART == 'src-doc':
                    documentsSource.write(text); documentsSource.flush()
                if GENERATE_PART == 'src-sum':
                    summariesSource.write(text); summariesSource.flush()
            count+=1

        if GENERATE_PART == 'tgt-sum':
            summariesTarget.close()
            newName = '{}_{}_tgt_titles.txt'.format(SRCLANG, TGTLANG)
            os.rename(os.path.join(HOME_SUBSETS, '{}_{}_titles_shortened.txt'.format(SRCLANG,TGTLANG)),
                      os.path.join(HOME_SUBSETS, newName))
            shutil.move(os.path.join(HOME_SUBSETS, newName),
                        os.path.join(HOME_SUBSETS, 'final/{}'.format(newName)))
            #src titles renamed and moved by hand
        if GENERATE_PART == 'src-sum':
            summariesSource.close()
        if GENERATE_PART == 'src-doc':
            documentsSource.close()
            # as this is last one move the titles file
            newName = '{}_{}_src_titles.txt'.format(SRCLANG, TGTLANG)
            os.rename(os.path.join(HOME_SUBSETS, '{}_titles_shortened.txt'.format(SRCLANG)),
                      os.path.join(HOME_SUBSETS, newName))
            shutil.move(os.path.join(HOME_SUBSETS, newName),
                    os.path.join(HOME_SUBSETS, 'final/{}'.format(newName)))

    print("\n".join(instanceDetails))
    print("Done {}-{} !".format(SRCLANG, TGTLANG))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument('--subsetsDir', help="Language to find intersection.")
    parser.add_argument('--srcLang', help="Language links pivot file.", required=True)
    parser.add_argument('--tgtLang', help="Pivot language of language links file", required=True)

    args = parser.parse_args(sys.argv[1:])

    run(args)