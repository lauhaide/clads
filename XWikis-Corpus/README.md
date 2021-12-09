
# XWikis Corpus creation process

Steps to extract Wikipedia based cross-lingual summarisation corpus. Most of the descriptions below will be given for 
English, German, French and Czech but can be modified to extract other
sets of languages.

## Download Wikipedia dumps

Download Wikipedia dumps En, Fr, De, Cz from [Wikimedia Downloads](https://dumps.wikimedia.org/enwiki/).
We used the dump from 20-06-2020 (i.e.,
[https://dumps.wikimedia.org/enwiki/20200620/](https://dumps.wikimedia.org/enwiki/20200620/) )
we kept a copy of these.

```
https://dumps.wikimedia.org/dewiki/20200620/enwiki-20200620-pages-articles.xml.bz2
https://dumps.wikimedia.org/dewiki/20200620/dewiki-20200620-pages-articles.xml.bz2
https://dumps.wikimedia.org/frwiki/20200620/frwiki-20200620-pages-articles.xml.bz2
https://dumps.wikimedia.org/cswiki/20200620/cswiki-20200620-pages-articles.xml.bz2
```

We use the following DB dump
[wikidatawiki-20200620-wb_items_per_site.sql.gz](https://dumps.wikimedia.org/wikidatawiki/20200620/wikidatawiki-20200620-wb_items_per_site.sql.gz)
for the table with inter-language links. These dumps may not be available for downloading, we kept a copy of them, you can write us for these.


## Generate the initial list of title-URL pairs available for each Wikipedia

We use [WikiExtractor](https://github.com/attardi/wikiextractor) version 2.75 (note that there is a new one now).
We first filter Wikipedia pages with less than 100 lines (already discards overall short articles).
Lists are not extracted (e.g. sections in Weblinks @bottom of this [page](https://de.wikipedia.org/wiki/Archimedes)).
Section titles are kept enclosed in html header tags (e.g., ```<h1></h1>```).
Our modified version [MyWikiExtractor.py](wikiextractor/) will extract a file *XXwiki-20200620.urls* (e.g., ```dewiki-20200620.urls```)
which contains a list of title and URL. This will be the candidate Wikipedia pages to be used for aligning across languages and building the corpus.

Note: use python==3.6.6 for this.

```
python ${PATHTO}/wikiextractor/MyWikiExtractor.py \
    -o ${HOME}/${DATALANG}wiki-20200620/ \
    -s -e --no-templates \
    --min_text_length 100 \
    ${HOME}/${DATALANG}wiki-20200620-pages-articles.xml.bz2 \
    --home ${HOME} \
    --xwikisDir 'xwikis-raw' \
    --outFileName ${OUTFILENAME}

```


## Match cross-language URLs

This script takes title-url files **XXwiki-20200620.urls** in a given language (e.g., dewiki-20200620.urls) as input, we call this the pivot
URLs, and extracts the corresponding titles in all languages that we ask (e.g, when we run this script we
asked for the four languages in our experiments --En, De, Fr and Cs-- . We also collected Zh, Es and Ru titles
correspondences, these are made available with our data too).

We use MySQL server with the database dump (point 1. above). For convenience, we first need to create views in
the DB for each pivot language wikipedia (e.g., *dewiki*).

```
CREATE VIEW dewiki AS
    select * from wb_items_per_site where ips_site_id='dewiki';
```

Then we run the script
```
python scripts/xwikis-interlang-sql.py --user MYSQLUSER --password MYSQLPASS \
        --pivot dewiki --pivot-urls dewiki-20200620.urls \
        --targets frwiki enwiki cswiki zhwiki eswiki ruwiki
```

This will generate a **language links** file named **XXwiki-20200620.urls.lang** (e.g., dewiki-20200620.urls.lang).
This generated file has a line per title. In this line, first comes the pivot (title and url)
then the title in each of the requested languages.
Note that if a title does not exists in a requested target language, it will
simple not be present in the line. Below an example of the content of the generated
file:
```
Ampere ||| https://de.wikipedia.org/wiki?curid=111 ||| zh::安培 ||| es::Amperio ||| ru::Ампер ||| fr::Ampère ||| en::Ampere ||| cs::Ampér
```

Due to the size of Wikipedias, we first generate the list of titles and language links that are candidates for creating the corpus, we then extract mono- and cross-lingual (document, lead) pairs.

## Extract title subsets from language links files

Given a pivot **language links** file named **XXwiki-20200620.urls.lang** (e.g., dewiki-20200620.urls.lang) and a set of desired languages (e.g., fr),
it will generate the subset of titles in the **XXwiki-20200620.urls.lang** that exists across all the desired languages (and the pivot's language). It will generate a file with the intersection (e.g., dewiki-20200620.urls.de-fr.lang).

```
python scripts/xwikis-interlang-intersection.py \
    --home ${HOME}/'xwikis-raw' \
    --langPivotFile ${PIVOTFILE} \
    --pivotLang ${PIVOTLANG} \
    --wikiLangs ${WIKILANGS}
```

## Create monolingual (body, lead) pairs for titles in language links files

Given a language links file (e.g., dewiki-20200620.urls.de-fr.lang),
we extract monolingual (body, lead) pairs for each language in the intersection (e.g., de and fr).
**While extracting we check length constraints**, do not inlude those out of length, we
considered leads with length in [20, 400] tokens and documents in [250, 5000] tokens.
Note that the extraction script will apply some heuristics to chunk documents longer than 5,000 tokens.
Again we use a modified version of WikiExtractor: *MyWikiCreator.py*.
This will generate files: **XX_documents.txt** (body of the wikipedia article in language XX), **XX_summaries.txt** (lead section of the article in language XX), **XX_titles.txt** (title of the article in language XX)  -- it will also produce a log file named XX_titles_outoflen.txt with the titles that do satisfy lengths (just for debugging).


```
python ${PATHTO}/wikiextractor/MyWikiCreator.py \
    -o ${HOME}/${DATALANG}wiki-20200620-tmp/ \
    -s -e --no-templates \
    --min_text_length 100 \
    ${HOME}/${DATALANG}wiki-20200620-pages-articles.xml.bz2 \
    --processes 200 \
    --home ${HOME} \
    --lang ${DATALANG} \
    --pivotName  ${PIVOTNAME} \
    --pivotLang  ${PIVOTLANG} \
    --outDirPrefix ${OUTDIRPREX} \
    --xwikisDir 'xwikis-raw'
```

For example, DATALANG='fr', PIVOTNAME='dewiki-20200620.urls.de-fr.lang', PIVOTLANG='de' will extract French monolingual (body, lead) pairs for the french titles in the language links intersection file (and will generate fr_documents.txt, fr_summaries.txt, , fr_titles.txt and fr_titles_outoflen.txt).

Note: temporary directories, e.g., *dewiki-20200620-tmp*, are not used, they are there just for compatibility of existing WikiExtractor code, so these
can be erased after extraction. The generated files will be saved.

Now we can refine the set of titles in the intersection language links file. We should rule out those titles that do not satisfy the length constraints (not satisfaction of the constraints in one of the languages is enough to exclude the title).


## Create cross-lingual (body, lead) pairs for documents in the monolingual sets


We can now extract cross-lingual lead sections in a target language YY for documents in language XX obtained in the previous step. Here create cross (body, lead) pairs. Files will be generated: **XX_YY_summaries.txt, XX_YY_titles.txt**

```
python ${PATHTO}/wikiextractor/MyWikiXAbstract.py \
    -o ${HOME}/${DATALANG}wiki-20200620-tmp/ \
    -s -e --no-templates \
    --min_text_length 100 \
    ${HOME}/${DATALANG}wiki-20200620-pages-articles.xml.bz2 \
    --processes 200 \
    --home ${HOME} \
    --lang ${DATALANG} \
    --pivotName  ${PIVOTNAME} \
    --pivotLang  ${PIVOTLANG} \
    --srcLang ${SRCLANG} \
    --xwikisDir 'xwikis-raw'
```

Note ```--lang ${DATALANG}``` where ```DATALANG}``` is the target language YY.

## Revise extracted title subsets from language links 

Note: These scripts can be re-written as convenient for creating intersections with good sizes. Another alternative would be to further modify WikiExtractor to extract texts on several languages at the time evaluating lengths and computing intersections.

#### Filter from pairs of languages intersection (e.g. de-en) those title tuples where a title does not satisfy the length constraints

To revise intersections of only two languages, the script is the following. This is basically re-computing the intersection based on the titles files for the monolingual extraction (e.g., de_titles.txt) and the corresponding cross-lingual extraction (e.g., de_en_titles.txt --the last one generated by the *MyWikiXAbstract.py* script).

```
python scripts/xwikis-pairs-intersection-good-sizes.py \
    --home ${HOME}/'xwikis-raw' \
    --subsetsDir ${SUBSETSDIR} \
    --langPivotFile ${PIVOTFILE} \
    --pivotLang ${PIVOTLANG} \
    --smallestSetLang ${SMALLEST_SET_LANG} \
    --toShortenLangs ${TOSHORTENLANG}
```

#### Filter from the de-fr-en intersection those title triplets where the length constraints are not satisfied

We create the monolingual files for each language in the intersection **dewiki-20200620.urls.de-fr-en.lang**. We then use the title files of each language: **de_titles.txt**, **fr_titles.txt**, **en_titles.txt** to revise the intersection removing title triplets where one of the titles in the triplet is not in its corresponding XX_titles.txt file (i.e., it did not satisfied the length constraints). The following script will do this for the **de-fr-en** intersection. It will generate a subset of the intersection where all titles have body-lead pairs satisfying length constraints. It will output this in the following files: **de_titles_shortened.txt**, **fr_titles_shortened.txt**, and **en_titles_shortened.txt**.

```
python scripts/xwikis-intersection-good-sizes-3.py \
    --home ${HOME}/'xwikis-raw' \
    --subsetsDir ${SUBSETSDIR} \
    --langPivotFile ${PIVOTFILE} \
    --pivotLang ${PIVOTLANG} \
    --smallestSetLang ${SMALLEST_SET_LANG} \
    --toShortenLangs ${OTHERLANGS}
```

We filter in the same way to add a forth language, e.g.,  **cs**. We use the following script ```xwikis-intersection-good-sizes-4.py``` instead. It will revise those files *XX_titles_shortened.txt* already be created for de, fr, and de, and will generate the one for cs. Note that this could be done in a single step, here the goal was to also compute the intermediate intersection of *3* languages. 


We want to obtain the intersection of *4* languages from the monolingual sets to create a parallel subset. Note that titles in the final *XX_titles.txt* files satisfy length constraints.


## Generate final (body, lead) pairs

We now have in *XX_titles.txt* those titles in each language that are in the desired intersection set and which satisfy the length constraints. The next step is to generate the final files that make the XWikis corpus. Here we generate subsets with all language pairs (i.e., intersection of two languages) and the intersection of all given languages, *de-fr-en-cs*, we will use this title parallel subset for testing.

#### Generate data pairs on two languages intersection (i.e., all language pairs)

This will create the final language pair subset, it will create files under the ```SUBSETSDIR/final``` (e.g., de-en.20-400.250-5000-per/final/). For instance, given *SRCLANG='de'*  and *TGTLANG='en'*, it will create the following files *de_en_src_documents.txt*, *de_en_src_summaries.txt*, *de_en_src_titles.txt*, *de_en_tgt_summaries.txt*, *de_en_tgt_titles.txt*.

```
python scripts/xwikis-pairs-dataset-good-sizes.py \
    --home ${HOME}/'xwikis-raw' \
    --subsetsDir ${SUBSETSDIR} \
    --srcLang ${SRCLANG} \
    --tgtLang ${TGTLANG}
```

#### Generate test split from the four languages intersection

Run the script ```python scripts/xwikis-sample-test-from-intersection.py --home ${HOME}/'xwikis-raw'``` to generate a sample of titles from the *4* languages titles intersection (**XWikis parallel**). Document/summary length statistics will be computed and we chose a sample with those closer to the statistics of the language pairs subsets (**XWikis comparable**). You may want to revise it as convenient for creating other XWikis language sets. This script will generate a folder named *test/* with the following files: *test.XX_titles.txt* where XX is a language in the intersection (i.e., de, fr, en, or cs). 

The following script will create the (document, lead) files for those titles in the test set files, test.XX_titles.txt :
```
python scripts/xwikis-dataset-good-sizes-test.py \
    --home ${HOME}/'xwikis-raw' \
    --subsetsDirIn ${SUBSETSDIRIN} \
    --subsetsDirOut ${SUBSETSDIROUT} \
    --intersectedLang ${INTERSECTIONLANG}
```

#### Generate train/valid splits from language pairs

Generate a train/valid split from the (document, lead) pairs in the two language intersection subsets. It will exclude those titles that are also in the 4 languages intersection and thus in the test split.
Depending on the source and target given languages, e.g., *SRCLANG='de'*  and *TGTLANG='en'*, it will generate the files for the (document, lead) cross-lingual pairs in train/valid splits and leave them under the folder *de-en* (note that this would be the German-to-English direction). For instance, it will create the following files, *train.de_en_src_documents.txt*, *train.de_en_src_summaries.txt*, *train.de_en_src_titles.txt*, *train.de_en_tgt_summaries.txt*, *train.de_en_tgt_titles.txt*.

```
python scripts/xwikis-resplit.py \
    --home ${HOME}/'xwikis-raw' \
    --datasetHomeDir ${DATASET_HOME} \
    --langPairFolder ${LANGPAIR_FOLDER} \
    --testFolder ${TEST_HOME} \
    --srcLang ${SRCLANG} \
    --tgtLang ${TGTLANG}
```

## Generate monolingual (body, lead) pairs with titles not included in the XWikis Corpus sets generated so far

This expects a file with the list of titles to be excluded when doing the extraction named as: *XX_titles_exclude.txt* where XX is the language ```--lang``` of extraction.
To generate the file with titles to exclude, we use a script that goes through the subsets generated for XWikis corpus (all subsets, all languages) and collect titles. Titles in other languages than the language that we want to extract, are mapped to the extraction language using the language links to the target language (see example script ```python scripts/xwikis-excluded-En.py --home ${HOME}/'xwikis-raw' ```).

```
python ${PATHTO}/wikiextractor/MyWikiExcluded.py
    -o ${HOME}/${DATALANG}wiki-20200620-tmp/ \
    -s -e --no-templates \
    --min_text_length 100 \
    ${HOME}/${DATALANG}wiki-20200620-pages-articles.xml.bz2 \
    --processes 200 \
    --home ${HOME} \
    --lang ${DATALANG} \
    --xwikisDir 'xwikis-raw'
```

Generate train/valid/test splits:

```
python scripts/xwikis-sample-datatest-from-MONOOnlyData.py \
    --home ${HOME}/'xwikis-raw' \
    --datasetHomeDir ${DATASET_HOME} \
    --monoDir ${MONODIR} \
    --monoLang ${MONOLANG} \
    --totalExamples ${TOTAL} \
    --stest ${STEST} \
    --svalid ${SVALID}
```


## Lead paragraph for all sets: final (document, summary) pairs in our task

On the extraction from Wikipedia dumps we recover the entire lead section and article body (we shrink with some heuristics those bodies with > 5k).
However for our task, for our cross-lingual summarisation task we only use the lead paragraph, i.e., the first paragraph of the lead section.
Once all the language sets and splits are created, before generating the binaries, it should be taken the lead paragraph of each (document, lead) data point. 
The script to do this is the following:
```
python scripts/xwikis-prepare-dataset.py \
    --home ${HOME} \
    --datasetDir ${DATASETDIR} \
    --filePrefix ${PREFIX_LIST} \
    --splits ${SPLITS}
```


## Pre-processing scripts

#### Initial extractive step

To deal with very long documents, we carry out an initial extractive step. Again, before generating the binaries,
we rank and select top ranked paragraphs with the following script:

```
  python baselines/LexRank.py \
   --file ${HOME}/${SUBSET_FOLDER}/${SPLIT}.${PREFIX}_documents.txt \
   --outfile ${HOME}/${SUBSET_FOLDER}/${SPLIT}.${PREFIX}_documents.lexrank.s${L}.txt \
   --task rank --algo lexrank --level paragraph --lang ${LANG} --L ${L}
```

#### Segmentation into sentences

In a similar way [mBART](https://arxiv.org/abs/2001.08210)'s training data is pre-processed, we split texts into sentences.
We used **czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger** for Czech and **stanford-corenlp-4.1.0** for other languages.


#### Other scripts

```./scripts/xwikis-normalise.sh ${SPMPATH} ${HOME}``` normalises extracted texts. Note that we use the given BPE models that are used [BPE data here](../fariseq2020/examples/clads/README.md) for details of the model. Note: in addition to the path to the sentence-piece code and data/model folder (```HOME```), you need to revise other variables/arguments in the script before running.



# Baselines

The code for the tree baselines we use in our paper (Oracle, LexRank and Lead) can be found under the [baselines/](baselines/) folder.

LexRank:
```
python baselines/LexRank.py \
    --file ${DATA}/${src}/${SPLIT}.${src}_articles.txt \
    --outfile ${RESULTDIR}/${SPLIT}.${OUTFILE} \
    --reference ${DATA}/${tgt}/${SPLIT}.${tgt}_summaries.txt \
    --task summarise --algo lexrank --level sentence --lang ${src} \
    --sum-in-sents
```

Lead:
```
python baselines/Lead.py \
    --file ${DATA}/${SPLIT}.${PREFIX}_documents.txt \
    --outfile ${RESULTDIR}/${SPLIT}.${OUTFILE} \
    --reference ${DATA}/${tgt}/${SPLIT}.${tgt}_summaries.txt
```

Oracle:
```
python baselines/Oracle.py \
    --file $DOCUMENTS \
    --outfile ${RESULTDIR}/${OUTFILE} \
    --reference-mono '${DATA}/${SPLIT}.${SRCLANG}'_summaries_prep.txt' \
    --reference-cross '${DATA}/${SPLIT}.${TGTLANG}'_summaries_prep.txt' \
    --titles ${TITLES} \
    --task $TASK
```

