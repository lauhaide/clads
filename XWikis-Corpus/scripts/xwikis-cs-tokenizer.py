
import sys, argparse

from ufal.morphodita import *

MORPHODITA_MODELDIR = '/home/lperez/wikigen/data/crosslingualDS/czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger'
tagger = Tagger.load(MORPHODITA_MODELDIR)
forms = Forms()
lemmas = TaggedLemmas()
tokens = TokenRanges()
tokenizer = tagger.newTokenizer()

def main(args):
    infile = open(args.infile,'r')
    outfile = open(args.outfile, 'w')
    for line in infile.readlines():
        tokenised = []
        text = line.strip()
        tokenizer.setText(text)
        t = 0
        while tokenizer.nextSentence(forms, tokens):
            for i in range(len(tokens)):
                token = tokens[i]
                tokenised.append(text[token.start : token.start + token.length])
                t = token.start + token.length
        outfile.write(" ".join(tokenised) + '\n')

    outfile.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
       description=__doc__,
       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', help="directory and file name of paragraphs to rank with topic salience.", required=True)
    parser.add_argument('--outfile', help="directory and file name for generated output.", required=True)
    args = parser.parse_args(sys.argv[1:])
    main(args)
