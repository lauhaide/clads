import os, argparse, sys
import numpy as np
from tqdm import tqdm

np.random.seed(44)

def run(args):

    RAW_FOLDER = os.path.join(args.home, args.monoDir)
    DATASET_FOLDER = os.path.join(args.datasetHomeDir, args.monoLang)
    LANG = args.monoLang

    sumEn = open(os.path.join(RAW_FOLDER, '{}_summaries.txt'.format(LANG)), 'r')
    docsEn = open(os.path.join(RAW_FOLDER, '{}_documents.txt'.format(LANG)), 'r')
    titlesEn = open(os.path.join(RAW_FOLDER, '{}_titles.txt'.format(LANG)), 'r')
    empty_idx, titlesNb = [], []

    for e, (s, d) in enumerate(zip(sumEn.readlines(), docsEn.readlines())):
        if len(s.strip().split())<20 or len(d.strip().split())<200:
            empty_idx.append(e) #check for ill extracted, but should be OK now
        else:
            titlesNb.append(e)
    sumEn.seek(0)
    docsEn.seek(0)
    totalEx = len(titlesNb) + len(empty_idx)
    print(" * from {} titles/documents are {}=empty".format(len(titlesNb), len(empty_idx)))
    pbar = tqdm(total=totalEx)


    examples = np.random.permutation(titlesNb)
    examples = examples[:args.totalExamples] #en=300000
    training_idx, val_idx, test_idx  =  np.split(examples, [args.svalid, args.stest]) #en=280000, 290000

    print(set(training_idx).intersection(set(empty_idx)))
    print(set(val_idx).intersection(set(empty_idx)))
    print(set(test_idx).intersection(set(empty_idx)))


    title = ".{}_titles.txt".format(LANG)
    doc = ".{}_documents.txt".format(LANG)
    sum = ".{}_summaries.txt".format(LANG)

    generate_files = {}
    for sp in ['train', 'val', 'test']:
        generate_files[sp] = {}
        for f in [title, doc, sum]:
            print("creating file... " +  os.path.join(DATASET_FOLDER, sp + f))
            generate_files[sp][f] = open(os.path.join(DATASET_FOLDER, sp + f), 'w')


    print("splits", len(training_idx), len(val_idx), len(test_idx))

    for i in range(totalEx):
        t = titlesEn.readline()
        d = docsEn.readline()
        s = sumEn.readline()
        if i in training_idx:
            currentSplit = 'train'
        elif i in val_idx:
            currentSplit = 'val'
        elif i in test_idx:
            currentSplit = 'test'
        else:
            continue
        generate_files[currentSplit][title].write(t)
        generate_files[currentSplit][doc].write(d)
        generate_files[currentSplit][sum].write(s)

        if (i % 200) == 0:
            pbar.update(200)
            generate_files[currentSplit][title].flush()
            generate_files[currentSplit][doc].flush()
            generate_files[currentSplit][sum].flush()



    for sp in ['train', 'val', 'test']:
        for f in [title, doc, sum]:
            generate_files[sp][f].close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related files are.")
    parser.add_argument('--datasetHomeDir', help="Directory where the generated splits will be located.")
    parser.add_argument('--monoDir', help="Directory with the monolingual files.")
    parser.add_argument('--monoLang', help="Language of monolingual files.")
    parser.add_argument('--totalExamples', help="Nb. total examples.")
    parser.add_argument('--stest', help="List index to split for test.")
    parser.add_argument('--svalid', help="List index to split for valid.")


    args = parser.parse_args(sys.argv[1:])

    run(args)