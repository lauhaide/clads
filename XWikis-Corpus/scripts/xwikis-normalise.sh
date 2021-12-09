##
# These are the files we want to normalise for further non-Neural processing, e.g. baselines and statistics
##


SPMENC=$1/sentencepiece/build/src/spm_encode
SPMDEC=$1/sentencepiece/build/src/spm_decode
MODEL=$2/pretrained/mbart.trimXWikis2/sentence.bpe.model
DATA=$2/datasets/XWikis-prepa

PREXFOLD='en-fr' #'fr-en' 'fr' 'cs-fr' 'de-fr'
PREX='en_fr' #'fr_en' 'fr' 'cs_fr' 'de_fr'
SENTFOLD='sentence-lrank2'
PREXLANG='fr' #used when mono or test

MONODATASET_FOLDER='cross' #'cross' 'mono'
declare -a SPLITS=('val' 'train') # 'test' 'val' 'train'

for SPLIT in ${SPLITS[@]}; do
      echo ${DATA}/${PREXFOLD}' '${SPLIT}

      if [ $MONODATASET_FOLDER == 'mono' ]; then
            INFILE=${SPLIT}'.'${PREXLANG}'_documents.txt'
            OUTFILE=${SPLIT}'.'${PREXLANG}'_documents_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            INFILE=${SPLIT}'.'${PREXLANG}'_summaries_prep.txt'
            OUTFILE=${SPLIT}'.'${PREXLANG}'_summaries_prep_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            INFILE=${SPLIT}'.'${PREX}'_summaries.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_summaries_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

      else
        if [ $SPLIT == "test" ]; then
            INFILE=${SPLIT}'.'${PREXLANG}'_documents.txt'
            OUTFILE=${SPLIT}'.'${PREXLANG}'_documents_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/test/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/test/${OUTFILE}

            INFILE=${SPLIT}'.'${PREXLANG}'_summaries_prep.txt'
            OUTFILE=${SPLIT}'.'${PREXLANG}'_summaries_prep_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/test/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/test/${OUTFILE}

            INFILE=${SPLIT}'.'${PREX}'_summaries.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_summaries_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/test/${SENTFOLD}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/test/${SENTFOLD}/${PREXFOLD}/${OUTFILE}

            INFILE=${SPLIT}'.'${PREX}'_documents.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_documents_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/test/${SENTFOLD}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/test/${SENTFOLD}/${PREXFOLD}/${OUTFILE}

        else
            INFILE=${SPLIT}'.'${PREX}'_src_documents.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_src_documents_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            INFILE=${SPLIT}'.'${PREX}'_src_summaries_prep.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_src_summaries_prep_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            INFILE=${SPLIT}'.'${PREX}'_src_summaries.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_src_summaries_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            INFILE=${SPLIT}'.'${PREX}'_tgt_summaries_prep.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_tgt_summaries_prep_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            INFILE=${SPLIT}'.'${PREX}'_tgt_summaries.txt'
            OUTFILE=${SPLIT}'.'${PREX}'_tgt_summaries_norm.txt'

            ${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${OUTFILE}
            mv ${DATA}/${PREXFOLD}/${OUTFILE} ${DATA}/${PREXFOLD}/${INFILE}

            #deprecated, now we normalise everything before running any text processing
            #INFILE=${SPLIT}'.'${PREX}'_summaries.txt'
            #OUTFILE=${SPLIT}'.'${PREX}'_summaries_norm.txt'
            #${SPMENC} --model=${MODEL} < ${DATA}/${PREXFOLD}/${SENTFOLD}/${INFILE} | ${SPMDEC} --model=${MODEL} > ${DATA}/${PREXFOLD}/${SENTFOLD}/${OUTFILE}

        fi
      fi
done
