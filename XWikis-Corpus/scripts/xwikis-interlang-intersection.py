"""
Goes through the wikilangs associated to the pivot wikipedia (e.g. De) and checks that the page exists in the given wikilangs
Output: File with titles in all wikilangs
"""

import os, argparse, sys
from tqdm import tqdm


def run(args):
    #total_in = 425350
    #pbar = tqdm(total=total_in)

    HOME = args.home

    pivotFile = args.langPivotFile
    titlesInterlangFiles = [pivotFile]

    # generate a list like this ['fr::', 'en::', 'cs::']
    wikilangs = ['{}::'.format(l) for l in args.wikiLangs.split('-')]
    outFileExtension = args.pivotLang + '-' + args.wikiLangs #'de-fr-en-cs'

    allTitlesFile = open(os.path.join(HOME, pivotFile.split('.lang')[0] + '.' + outFileExtension + '.lang'),'w')
    i = 0
    for titleFileName in titlesInterlangFiles:
        titlesFile = open(os.path.join(HOME, titleFileName), 'r')
        for line in titlesFile.readlines():
            i+=1
            all = True
            for l in wikilangs:
                all = all and l in line
            if all:
                allTitlesFile.write(line)
            titlesFile.close()
            #if (i % 200) == 0:
            #    pbar.update(200)

    allTitlesFile.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupP.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument('--langPivotFile', help="Language links pivot file.", required=True)
    parser.add_argument('--pivotLang', help="Pivot language of language links file", required=True)
    parser.add_argument('--wikiLangs', help="Language to find intersection.")

    args = parser.parse_args(sys.argv[1:])

    run(args)
