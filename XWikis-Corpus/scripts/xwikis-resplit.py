import os, argparse, sys
import numpy as np
from tqdm import tqdm

np.random.seed(44)

def run(args):


    #test set entities to exclude in val/train
    TEST_HOME = os.path.join(args.home, args.testFolder)
    RAW_FOLDER = os.path.join(args.home, args.langPairFolder)
    SRCLANG= args.srcLang
    TGTLANG= args.tgtLang


    SPLITS_FOLDER = os.path.join(args.datasetHomeDir, "{}-{}".format(SRCLANG, TGTLANG))
    print(" * {}-{}".format(SRCLANG, TGTLANG))

    testSRC = open(os.path.join(TEST_HOME, "test.{}_titles.txt".format(SRCLANG)), 'r')
    testTGT = open(os.path.join(TEST_HOME, "test.{}_titles.txt".format(TGTLANG)), 'r')
    sTestTitles = [l.strip() for l in testSRC.readlines()]
    tTestTitles = [l.strip() for l in testTGT.readlines()]

    srcTitlesFile = open(os.path.join(RAW_FOLDER, "final/{}_{}_src_titles.txt".format(SRCLANG, TGTLANG)), 'r')
    tgtTitlesFile = open(os.path.join(RAW_FOLDER, "final/{}_{}_tgt_titles.txt".format(SRCLANG, TGTLANG)), 'r')
    totalNbInst = len([l.strip() for l in tgtTitlesFile.readlines()])
    tgtTitlesFile.seek(0)
    print(" * Get train from remaining instances from {}, {} are taken on test.".format(totalNbInst, len(tTestTitles)))
    srctgtTitles = [(s.strip(), t.strip()) for s,t in zip(srcTitlesFile.readlines(), tgtTitlesFile.readlines()) \
                    if not s.strip() in sTestTitles and not t.strip() in tTestTitles]
    srcTitlesFile.seek(0)
    tgtTitlesFile.seek(0)
    pbar = tqdm(total=len(srctgtTitles))
    print(" * Split remaining {}".format(len(srctgtTitles)))

    srcDocFile = open(os.path.join(RAW_FOLDER, "final/{}_{}_src_documents.txt".format(SRCLANG, TGTLANG)), 'r')
    srcSumFile = open(os.path.join(RAW_FOLDER, "final/{}_{}_src_summaries.txt".format(SRCLANG, TGTLANG)), 'r')
    tgtSumFile = open(os.path.join(RAW_FOLDER, "final/{}_{}_tgt_summaries.txt".format(SRCLANG, TGTLANG)), 'r')

    srct = ".{}_{}_src_titles.txt".format(SRCLANG, TGTLANG)
    tgtt = ".{}_{}_tgt_titles.txt".format(SRCLANG, TGTLANG)
    srcdoc = ".{}_{}_src_documents.txt".format(SRCLANG, TGTLANG)
    srcsum = ".{}_{}_src_summaries.txt".format(SRCLANG, TGTLANG)
    tgtsum = ".{}_{}_tgt_summaries.txt".format(SRCLANG, TGTLANG)

    generate_files = {}
    for sp in ['train', 'val']:
        generate_files[sp] = {}
        for f in [srct, tgtt, srcdoc, srcsum, tgtsum]:
            print("creating file... " +  os.path.join(SPLITS_FOLDER, sp + f))
            generate_files[sp][f] = open(os.path.join(SPLITS_FOLDER, sp + f), 'w')


    examples = np.arange(len(srctgtTitles))
    indices = np.random.permutation(examples.shape[0])
    training_idx, val_idx =  np.split(indices, [int(.95 * len(examples))])

    srcTrain = list(np.array(srctgtTitles)[:,0][training_idx.astype(int)])
    tgtTrain = list(np.array(srctgtTitles)[:,1][training_idx.astype(int)])
    srcVal = list(np.array(srctgtTitles)[:,0][val_idx.astype(int)])
    tgtVal = list(np.array(srctgtTitles)[:,1][val_idx.astype(int)])

    print("Splits' size: train/val {}/{}".format(len(srcTrain), len(srcVal)))

    for i, (s, t) in enumerate(srctgtTitles):
        st = srcTitlesFile.readline()
        tt = tgtTitlesFile.readline()
        sd = srcDocFile.readline()
        ss = srcSumFile.readline()
        ts = tgtSumFile.readline()

        if s in srcTrain:
            currentSplit = 'train'
        elif s in srcVal:
            currentSplit = 'val'
        else:
            continue
        generate_files[currentSplit][srct].write(st)
        generate_files[currentSplit][tgtt].write(tt)
        generate_files[currentSplit][srcdoc].write(sd)
        generate_files[currentSplit][srcsum].write(ss)
        generate_files[currentSplit][tgtsum].write(ts)

        if (i % 200) == 0:
            pbar.update(200)
            generate_files[currentSplit][srct].flush()
            generate_files[currentSplit][tgtt].flush()
            generate_files[currentSplit][srcdoc].flush()
            generate_files[currentSplit][srcsum].flush()
            generate_files[currentSplit][tgtsum].flush()


    for sp in ['train', 'val']:
        for f in [srct, tgtt, srcdoc, srcsum, tgtsum]:
            generate_files[sp][f].close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--home", type=str, default='', required=True,
                        help="Home where all XWikis related raw files are.")
    parser.add_argument('--datasetHomeDir', help="Directory where the generated splits will be located.")
    parser.add_argument('--langPairFolder', help="Folder of a given language pair.", required=True)
    parser.add_argument('--testFolder', help="Folder of the created test split. Titles from this folder/split will be excluded when creating the train/valid splits.", required=True)
    parser.add_argument('--srcLang', help="Language links pivot file.", required=True)
    parser.add_argument('--tgtLang', help="Pivot language of language links file", required=True)

    args = parser.parse_args(sys.argv[1:])

    run(args)