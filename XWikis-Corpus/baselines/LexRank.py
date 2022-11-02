
import sys, argparse
import collections
import math
import numpy as np

from xrank import LexRank
from stopwords import STOPWORDS

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

def generateRankedParagraphsSentence(args):
    print("Unimplemented!")
    pass

def generateRankedParagraphsPara(args):
    threshold = float(args.threshold_matrix) if args.threshold_matrix else None

    text = open(args.file, 'r')
    out = open(args.outfile, 'w')
    textLines = text.readlines()
    trnk = LexRank(textLines, stopwords=STOPWORDS[args.lang])
    print("LexRank loaded...")

    MAX_TOKENS = args.L

    for doc in textLines:
        #print(len(doc.strip().split('<p>')))
        doc = doc.replace(' <h2>', ' <p> <h2>'). \
            replace(' <h3>', ' <p> <h3>'). \
            replace(' <h4>', ' <p> <h4>'). \
            replace(' <h5>', ' <p> <h5>')
        #print(len(doc.strip().split('<p>')))
        paras = doc.strip().split('<p>')
        lex_scores = trnk.rank_sentences(
            paras,
            threshold=threshold)

        sorted_ix = np.argsort(lex_scores)[::-1]

        # ranked_paras = [paras[i] for i in sorted_ix]

        first_ixs = []
        toks = 0
        for i in sorted_ix: # traverse the sorted list take paragraphs up to a nb of tokens
            if toks > MAX_TOKENS:
                break
            nextPara = paras[i]
            toks += len(nextPara.split())
            first_ixs.append(i)

        ranked_paras = []
        for i in range(len(paras)):
            if i in first_ixs:
                ranked_paras.append(paras[i])

        out.write("<p>".join(ranked_paras) + '\n')

    out.close()

def generateRankedParagraphs(args):
    """"""
    print(args)
    print("*\tLexRank source paragraphs, generate ranked files...")
    if args.level == 'paragraph':
        generateRankedParagraphsPara(args)
    else:
        print('Unimplemented.')


def generateSummarise(args):
    """ LexRank extractive summariser."""

    if args.level == 'paragraph':
        print('Unimplemented.')
        pass

    # assume summaries are build from lexrank at sentence level

    if not getattr(args, "reference", None):
        print('Should provide reference file.')
        pass

    print(args)
    print("*\tLexRank summariser...")
    threshold = float(args.threshold_matrix) if args.threshold_matrix else None
    out = open(args.outfile, 'w')

    textLines = readTexts(args.file)
    trnk = LexRank(textLines, stopwords=STOPWORDS[args.lang])
    print("LexRank loaded...")

    if args.sum_in_sents:
        rf = readTexts(args.reference)
        refLen = [len(" ".join(s).split()) for s in rf ]
    else:
        rf = open(args.reference,'r')
        refLen = [len(l.strip().split()) for l in rf.readlines()]

    # double check we work with same summary doc pairs
    assert len(refLen) == len(textLines), "references={} documents={}".format(len(refLen), len(textLines))

    for e, doc in enumerate(textLines):
        lex_scores = trnk.rank_sentences(
            doc,
            threshold=threshold)

        sorted_ix = np.argsort(lex_scores)[::-1]
        ranked_sentences = [doc[i] for i in sorted_ix]
        ranked_sentences = " ".join(ranked_sentences).split()
        ranked_sentences = " ".join(ranked_sentences[:refLen[e]]) # take as many tokens as are in the ref sum
        out.write(ranked_sentences + '\n')
        out.flush()

    out.close()


def evalRankedParagraphs(args):

    sample=4000
    useNTtopics = [int(n) for n in args.top_n_topics.split(",")] if hasattr(args, 'top_n_topics') and args.top_n_topics else []
    weighted = hasattr(args, 'weighted_tf_idf') and args.weighted_tf_idf

    trnk = TopicRank(args.loadmodel)
    datasetName = args.loadmodel[args.loadmodel.rfind("/")+1:]
    text = open(args.file, "r", encoding="utf-8")
    textBoW = open(args.file.replace("src", "bow.src"), "r", encoding="utf-8")
    textTfidf = open(args.file.replace("src", "tfidf.src"), "r", encoding="utf-8")
    tgtf = open(args.file.replace("src", "tgt"), 'r', encoding='utf-8')
    outLog = open("log/topicRankSearch-w{}-{}.log".format(weighted, datasetName), "w", encoding="utf-8")
    outLog.write("sample size ({}), L={}\n\n".format(sample, args.L))

    idfRanked = []
    topicDists = []
    tgts = []
    for e, (exsrc, exsrcbow, exsrctfidf, extgt) in \
            enumerate(zip(text.readlines(), textBoW.readlines(), textTfidf.readlines(), tgtf.readlines())):
        if e > sample:
            break
        parts = exsrc.split(EOT)
        title = parts[0]
        paragraphs = parts[1].replace('\n', ' ').split(EOP)
        scores = exsrctfidf.replace('\n', ' ').split(EOP)
        paragraphsBow = exsrcbow.replace('\n', ' ').split(EOP)

        tvs = trnk.getTopicDistrib([p.split() for p in paragraphsBow], int(args.K), None)

        #New_ParagraphsBow = []
        New_paras_scores = []
        New_tvs = []
        for p, pbow, s, v in zip(paragraphs, paragraphsBow, scores, tvs):
            if v.max()>0.2:
                #New_ParagraphsBow.append(pbow)
                New_paras_scores.append((p,s))
                New_tvs.append(v)

        #topicDists.append(tvs)
        topicDists.append(New_tvs)

        rkp = {'title': title,
               #'ranked_paragraphs': [(p, s) for p, s in zip(paragraphs, scores)]
               'ranked_paragraphs': New_paras_scores
               }
        idfRanked.append(rkp)


        tgts.append(extgt)

        assert len(paragraphs) == len(tvs) and len(paragraphsBow) == len(tvs), \
                        "{}/{}/{}".format(len(paragraphs), len(paragraphsBow), len(tvs))
        assert len(New_paras_scores) == len(New_tvs)

    print("*    Start search on topic-rank...")

    thresholds = [0.2, 0.3, 0.4, 0.7, 0.9, None]
    d_weights = [0.0, 0.4, 0.5, 0.6]#, 1.0]
    topics = useNTtopics + [None] # in search useNTtopics should be a list

    #for each example in the dataset, rank its paragraphs according to topicRank
    resultsDict = {}
    for t in topics:
        for e, (idf_paragraphs, topic_scores, tgttext) in enumerate(zip(idfRanked, topicDists, tgts)):

            similarity_matrix, query_relevance = trnk.rankParagraphSearchMatrix(idf_paragraphs['ranked_paragraphs'], None, None,
                                                           t, topicVectors=topic_scores)

            tgttext = tgttext.replace(SNT, BLANK).replace(EOP, BLANK).split()
            for th in thresholds:
                markov_matrix = trnk.getMarkovMatrix(similarity_matrix, th)

                for dw in d_weights:
                    rnkpara = trnk.getRankedScores(idf_paragraphs['ranked_paragraphs'], markov_matrix, query_relevance, dw)
                    srctext = (" ".join(rnkpara)).split()[0:int(args.L)]

                    combKey = "{}-{}-{}".format(str(t), str(th), str(dw))
                    if not combKey in  resultsDict.keys():
                        resultsDict[combKey] = {'rouge1': [], 'rouge2': [], 'rougeL': [] }
                    resultsDict[combKey]['rouge1'].append(rouge_n_recall([srctext], [tgttext], 1))
                    resultsDict[combKey]['rouge2'].append(rouge_n_recall([srctext], [tgttext], 2))
                    resultsDict[combKey]['rougeL'].append(rouge_l_recall([srctext], [tgttext]))

    print("*    Saving models' rouge scores")
    max_key = ''
    max_rouge1, max_rouge2, max_rougel = 0.0, 0.0, 0.0
    for k in resultsDict.keys():
        parts = k.split("-")
        outLog.write("\nweigthed:{} N:{}, threshold:{}, d:{} \n".format(weighted, parts[0], parts[1], parts[2]))
        rouge1 = np.mean(resultsDict[k]['rouge1'])
        rouge2 = np.mean(resultsDict[k]['rouge2'])
        rougel = np.mean(resultsDict[k]['rougeL'])
        outLog.write("ROUGE-1 R: {}\n".format(rouge1))
        outLog.write("ROUGE-2 R: {}\n".format(rouge2))
        outLog.write("ROUGE-L R: {}".format(rougel))
        outLog.flush()
        if rougel > max_rougel:
            max_rougel = rougel
            max_rouge1 = rouge1
            max_rouge2 = rouge2
            max_key = k

    parts = max_key.split("-")
    outLog.write("\n\nHighest scoring model: {}\n\tROUGE-1 R: {}\tROUGE-2 R: {}\tROUGE-L R: {}"
                 .format("weigthed:{} N:{}, threshold:{}, d:{} \n".format(weighted, parts[0], parts[1], parts[2]),
                            max_rouge1, max_rouge2, max_rougel))


    outLog.close()
    print("Goodbye")

def _rank_reference_paragraphs(wiki_title, references_content, weighted=False):
  """DEPRECATED"""
  """BASED on wikisum.py if we want to re-rank paragraphs here..."""
  """Rank and return reference (means citations) paragraphs by tf-idf score on title tokens."""
  #title_tokens = _tokens_to_score(set(
  #    tokenizer.encode(text_encoder.native_to_unicode(wiki_title))))
  title_tokens = wiki_title.split()
  ref_paragraph_info = []
  doc_counts = collections.defaultdict(int)
  target_positions = {}
  for r, paragraph in enumerate(references_content):
      counts = _token_counts(paragraph.split(), title_tokens)

      for token in title_tokens:
        if counts[token]:
          doc_counts[token] += 1
      info = {"content": paragraph, "counts": counts}
      ref_paragraph_info.append(info)


  for info in ref_paragraph_info:
    score = 0.
    for token in title_tokens:
      if weighted:
        term_frequency = (info["counts"][token] / max(len(info["content"].split()),1) )
      else:
        term_frequency = info["counts"][token]
      inv_doc_frequency = (
          float(len(ref_paragraph_info)) / max(doc_counts[token], 1))
      score += term_frequency * math.log(inv_doc_frequency)
    info["score"] = score

  #ref_paragraph_info.sort(key=lambda el: el["score"], reverse=True)
  return [(info["content"], info["score"]) for info in ref_paragraph_info]

def _token_counts(text, token_set=None):
  counts = collections.defaultdict(int)
  #for token in tokenizer.encode(text_encoder.native_to_unicode(text)):
  for token in text:
    if token_set and token not in token_set:
      continue
    counts[token] += 1
  return counts

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file', help="directory and file name of paragraphs to rank.", required=True)
    parser.add_argument('--outfile', help="directory and file name for generated output.", required=True)
    parser.add_argument('--reference', help="directory and file name for reference summary.")
    parser.add_argument('--sum-in-sents',  action='store_true', help="summaries are split into sentences (same as input docs/articles).")
    parser.add_argument('--lang',
                        nargs='?',
                        choices=['en', 'de', 'fr', 'cs'],
                        help='language')
    parser.add_argument('--loadmodel', help="directory and file name of XLM-R model.")
    parser.add_argument('--weighted-tf-idf', help="whether to use weighted tf-idf ranking.", action='store_true')
    parser.add_argument('--threshold-matrix', help="threshold for similarity matrix construction, This threshold is "
                                                   "applied to Jannon-Shensen divergence (i.e. closer to zero is better "
                                                   " and goes beyond 1). If not given the continuous version is used.")
    parser.add_argument('--task',
                        nargs='?',
                        choices=['select', 'summarise', 'rank'],
                        help='hyperparameter selection (select), generate summaries (summarise), rank document paragraphs (rank).')
    parser.add_argument('--algo',
                        nargs='?',
                        choices=['lexrank', 'neuralrank'],
                        help='which lexrank variant to use')
    parser.add_argument('--level',
                        nargs='?',
                        choices=['sentence', 'paragraph'],
                        help='at which level to apply lexrank (input file will have diff. format for these)')
    parser.add_argument('--L', metavar='N', default=None, type=int, help='Nb of tokens to take from the input')
    parser.add_argument('--num-tasks', metavar='N', default=None, type=int, help='Nb of tasks to run on parallel. (used for generate).')
    parser.add_argument('--task-id', metavar='N', default=None, type=int, help='Actual task to execute. (used for generate).')

    args = parser.parse_args(sys.argv[1:])


    if args.algo == 'lexrank':
        if args.task == 'select':
            print("unimplemented!")
        elif args.task == 'summarise':
            generateSummarise(args)
        elif args.task == 'rank':
            generateRankedParagraphs(args)

    elif args.algo == 'neuralrank':
        print("unimplemented!")

    print("Goodbye")

