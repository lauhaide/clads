import os
from tqdm import tqdm
import sys, argparse

def readTexts(file):
    ret = []
    doc = []
    f = open(file, 'r')
    for l in f.readlines():
        if l.strip():
            doc.append(l.strip())
        else:
            ret.append(doc)
            doc = []
    return ret

def generateSummarise(args):
    #if DATASET == 'voxeurop':
    #    print("Reading from... ", os.path.join(DATASET_FOLDER, SRCLANG, '{}.{}_articles.txt'.format(SPLIT, SRCLANG)))
    #    summaries = readTexts(os.path.join(DATASET_FOLDER, TGTLANG, '{}.{}_summaries.txt'.format(SPLIT, TGTLANG)))
    #    documents = readTexts(os.path.join(DATASET_FOLDER, SRCLANG, '{}.{}_articles.txt'.format(SPLIT, SRCLANG)))
    #    candidateLead = open(os.path.join(RESULT_DIR, '{}.{}_{}_lead.txt'.format(SPLIT, SRCLANG, TGTLANG)), 'w')
    #else:
    #    if SPLIT == 'test':
    #        if TASK=='mono':
    #            summaries = open(os.path.join(TEST_FOLDER_SUMMONO, '{}.{}_summaries_prep.txt'.format(SPLIT, SRCLANG)), 'r').readlines()
    #        else:
    #            summaries = readTexts(os.path.join(TEST_FOLDER, '{}.{}_summaries.txt'.format(SPLIT, PREFIX)))
    #        documents = readTexts(os.path.join(TEST_FOLDER, '{}.{}_documents.txt'.format(SPLIT, PREFIX)))
    #        #titles = open(os.path.join(DATASET_FOLDER, '{}.{}_titles.txt'.format(SPLIT, SRCLANG)))
    #    else:
    #        if TASK=='mono':
    #            summaries = open(os.path.join(DATASET_FOLDER, \
    #                                          '{}.{}_src_summaries_prep.txt'.format(SPLIT, PREFIX)), 'r').readlines()
    #        else:
    #            summaries = readTexts(os.path.join(DATASET_FOLDER, 'sentence-all', '{}.{}_summaries.txt'.format(SPLIT, PREFIX)))
    #        documents = readTexts(os.path.join(DATASET_FOLDER, 'sentence-all', '{}.{}_documents.txt'.format(SPLIT, PREFIX)))
    #        #titles = open(os.path.join(DATASET_FOLDER, '{}.{}{}_titles.txt'.format(SPLIT, PREFIX, SRC_TGT)))
    #    candidateLead = open(os.path.join(RESULT_DIR, '{}.{}_lead.txt'.format(SPLIT, PREFIX)), 'w')


    documents = readTexts(args.file)
    candidateLead = open(args.outfile, 'w')
    if args.sum_in_sents:
        summaries = readTexts(args.reference)
    else:
        summaries = open(args.reference, 'r').readlines()

    nbTitles = len(summaries)
    assert len(documents) == len(summaries), "d:{} s:{}".format(len(documents), len(summaries))

    pbar = tqdm(total=nbTitles)
    i = 0
    for d, s in zip(documents, summaries):
        i += 1
        if not args.sum_in_sents:
            strLead = s.strip().split()
        else:
            strLead = " ".join(s).split()

        candidateLead.write(" ".join(" ".join(d).split()[:len(strLead)]) + '\n')

        if (i % 200) == 0:
            pbar.update(200)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file', help="directory and file name of documents to summarise.", required=True)
    parser.add_argument('--outfile', help="directory and file name for generated output.", required=True)
    parser.add_argument('--reference', help="directory and file name for reference summary.")
    parser.add_argument('--sum-in-sents', default=False, action='store_true',
                        help="summaries are split into sentences (same as input docs/articles).")

    args = parser.parse_args(sys.argv[1:])

    generateSummarise(args)


