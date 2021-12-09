import os
import sys, argparse
from tqdm import tqdm
from rouge_utils import rouge_calculator_sum, rouge_calculator

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
    #    cross_summaries = readTexts(os.path.join(DATASET_FOLDER, TGTLANG, '{}.{}_summaries.txt'.format(SPLIT, TGTLANG)))
    #    documents = readTexts(os.path.join(DATASET_FOLDER, SRCLANG, '{}.{}_articles.txt'.format(SPLIT, SRCLANG)))
    #    summaries = readTexts(os.path.join(DATASET_FOLDER, SRCLANG, '{}.{}_summaries.txt'.format(SPLIT, SRCLANG)))
    #    oracleSum = open(os.path.join(RESULT_DIR, '{}.{}_{}_oracle.txt'.format(SPLIT, SRCLANG, TGTLANG)), 'w')
    #    nbTitles = len(documents)
    #else:
    #    if SPLIT == 'test':
    #        if PREFIX == 'en':
    #            summaries = readTexts(os.path.join(TEST_FOLDER, '{}.{}_summaries.txt'.format(SPLIT, SRCLANG)))
    #        else:
    #            summaries = open(os.path.join(DATASET_FOLDER, '{}.{}_summaries_prep.txt'.format(SPLIT, SRCLANG)),'r').readlines()
    #        cross_summaries = open(os.path.join(DATASET_FOLDER, '{}.{}_summaries_prep.txt'.format(SPLIT, TGTLANG)),'r').readlines()
    #        documents = readTexts(os.path.join(TEST_FOLDER, '{}.{}_documents.txt'.format(SPLIT, PREFIX)))
    #        titles = open(os.path.join(DATASET_FOLDER, '{}.{}_titles.txt'.format(SPLIT, SRCLANG)))
    #    else:
    #        if PREFIX == 'en':
    #            summaries = readTexts(os.path.join(DATASET_FOLDER, 'sentence-all', '{}.{}_summaries.txt'.format(SPLIT, PREFIX)))
    #        else:
    #            # if doing cross, take the summary in the source language, this is ORACLE!
    #            summaries = open(os.path.join(DATASET_FOLDER, '{}.{}_src_summaries_prep.txt'.format(SPLIT, PREFIX)), 'r').readlines()
    #        cross_summaries = open(os.path.join(DATASET_FOLDER, '{}.{}_tgt_summaries_prep.txt'.format(SPLIT, PREFIX)), 'r').readlines()
    #        documents = readTexts(os.path.join(DATASET_FOLDER, 'sentence-all', '{}.{}_documents.txt'.format(SPLIT, PREFIX)))
    #        titles = open(os.path.join(DATASET_FOLDER, '{}.{}{}_titles.txt'.format(SPLIT, PREFIX, SRC_TGT)))
    #    oracleSum = open(os.path.join(RESULT_DIR, '{}.{}_oracle.v1.txt'.format(SPLIT, PREFIX)), 'w')
    #    titleLines = titles.readlines()
    #    nbTitles = len([1 for l in titleLines])

    documents = readTexts(args.file)
    if not args.sum_in_sents_mono:
        summaries = open(args.reference_mono, 'r').readlines()
    else:
        summaries = readTexts(args.reference_mono)

    if not args.sum_in_sents_cross:
        cross_summaries = open(args.reference_cross, 'r').readlines()
    else:
        cross_summaries = readTexts(args.reference_cross)


    if hasattr(args, 'titles'):
        titles = open(args.titles)
        titleLines = titles.readlines()
        nbTitles = len([1 for l in titleLines])
    else:
        nbTitles = len(documents)

    assert nbTitles == len(documents) == len(summaries), "t:{} / d:{} / s:{}".format(nbTitles, len(documents), len(summaries))

    oracleSum = open(args.outfile, 'w')

    pbar = tqdm(total=nbTitles)
    i = 0
    for d, s, c in zip(documents, summaries, cross_summaries):
        i += 1

        if args.sum_in_sents_mono:
            tokLead = " ".join(s).split()
        else:
            tokLead = s.strip().split()

        if args.sum_in_sents_cross:
            tokCrossLead = " ".join(c).split()
        else:
            tokCrossLead = c.strip().split()


        sentRouge = []
        for e, line in enumerate(d):
            if not line.split() or not tokLead:
                print(d)
                exit()
            rouge = rouge_calculator([line.split()], [tokLead], n=2, type='f')
            sentRouge.append((e, line.split(), rouge[0]))

        sentRouge.sort(key=lambda x: x[2], reverse=True)
        e, sum, score = sentRouge.pop(0)
        print(len(tokCrossLead), len(tokLead), len(" ".join(d).split()), len(sentRouge))
        while True:
            if not sentRouge:
                break
            tmp = []
            for e, (_, s, _) in enumerate(sentRouge):
                rouge = rouge_calculator([sum + s], [tokLead], n=2, type='f')
                #print ("{} >> ".format(rouge[0]), sum+s)
                tmp.append( (e, sum + s, rouge[0]) )

            tmp.sort(key=lambda x: x[2], reverse=True)
            e, tmpSum, tmpScore = tmp.pop(0)
            #if (PREFIX == 'en' and tmpScore > score) or \
            #        (PREFIX != 'en' and len(" ".join(tmpSum).split()) < len(tokCrossLead) ):
            MAXLEN = len(tokLead) if args.task=='mono'  else len(tokCrossLead)
            if len(" ".join(tmpSum).split()) < MAXLEN \
                    and len(" ".join(tmpSum).split()) < len(" ".join(d).split()): ##security constraint, sums smaller than docs????
                sum = tmpSum
                score = tmpScore
                sentRouge.pop(e)
            else:
                break
        oracleSum.write(" ".join(sum) + '\n')
        oracleSum.flush()

        if (i % 200) == 0:
            pbar.update(200)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file', help="directory and file name of documents to summarise.", required=True)
    parser.add_argument('--outfile', help="directory and file name for generated output.", required=True)
    parser.add_argument('--title', help="directory and file corresponding to the title list that is to be summarised.")
    parser.add_argument('--reference-mono', help="directory and file name for monolingual reference summary.", required=True)
    parser.add_argument('--reference-cross', help="directory and file name for crosslingual reference summary.", required=True)
    parser.add_argument('--sum-in-sents-mono', default=False, action='store_true',
                        help="summaries are split into sentences (same as input docs/articles).")
    parser.add_argument('--sum-in-sents-cross', default=False, action='store_true',
                        help="summaries are split into sentences (same as input docs/articles).")
    parser.add_argument('--task', default='mono', choices=['mono', 'cross'], help="directory and file name for generated output.", required=True)

    args = parser.parse_args(sys.argv[1:])

    generateSummarise(args)

