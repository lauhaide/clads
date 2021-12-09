## adapted from https://github.com/crabcamp/lexrank


import math
from collections import Counter, defaultdict

import numpy as np

from power_method import stationary_distribution
from text import tokenize


from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

class LexRank:
    def __init__(
            self,
            documents,
            stopwords=None,
            keep_numbers=False,
            keep_emails=False,
            keep_urls=False,
            include_new_words=True,
    ):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words


        self.idf_score = self._calculate_idf(documents)

    def get_summary(
        self,
        sentences,
        summary_size=1,
        threshold=.03,
        fast_power_method=True,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

        tf_scores = [
            Counter(self.tokenize_sentence(sentence)) for sentence in sentences
        ]

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    def sentences_similarity(self, sentence_1, sentence_2):
        tf_1 = Counter(self.tokenize_sentence(sentence_1))
        tf_2 = Counter(self.tokenize_sentence(sentence_2))

        similarity = self._idf_modified_cosine([tf_1, tf_2], 0, 1)

        return similarity

    def tokenize_sentence(self, sentence):
        tokens = tokenize(
            sentence,
            self.stopwords,
            keep_numbers=self.keep_numbers,
            keep_emails=self.keep_emails,
            keep_urls=self.keep_urls,
        )

        return tokens

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = math.log(doc_number_total + 1)

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def _calculate_similarity_matrix(self, tf_scores):
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity


    def _markov_matrix(self, weights_matrix):

        n_1, n_2 = weights_matrix.shape
        if n_1 != n_2:
            raise ValueError('\'weights_matrix\' should be square')

        row_sum = weights_matrix.sum(axis=1, keepdims=True)

        return weights_matrix / row_sum

    def _markov_matrix_discrete(self, weights_matrix, threshold):
        discrete_weights_matrix = np.zeros(weights_matrix.shape)
        ixs = np.where(weights_matrix >= threshold)
        discrete_weights_matrix[ixs] = 1

        return self._markov_matrix(discrete_weights_matrix)

    def getMarkovMatrix(self, similarity_matrix, threshold):
        """"""

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )
        return markov_matrix


class NeuralRank:
    def __init__(
        self,
        ldaModel,
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True,
    ):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words

        self.ldaModel = gensim.models.ldamodel.LdaModel.load(ldaModel)
        self.topickeys = getTopicKeys(self.ldaModel)

        #self.idf_score = self._calculate_idf(documents)

    def get_summary(
        self,
        sentences,
        summary_size=1,
        threshold=.03,
        fast_power_method=True,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def getRankedParagraphs(self, paraFullText, paraPreprocBoW, d, K, useNTtopics, threshold, topicVectors=None):
        """

        :param paraFullText: a list of strings, i.e. each is a paragraph
        :param paraPreprocBoW: a list of lists of words, i.e. list of words in a paragraph
        :return:
        """

        scores = self.rankParagraph(paraFullText, paraPreprocBoW, d, K, useNTtopics, threshold, topicVectors=topicVectors)

        sortedIndex = np.argsort(scores)[::-1]

        if paraPreprocBoW is not None:
            return [paraFullText[i][0] for i in sortedIndex], [paraPreprocBoW[i] for i in sortedIndex]
        else:
            return [paraFullText[i][0] for i in sortedIndex], None


    def getTopicDistrib(self, paraPreprocBoW, K, useNTtopics):
        """"""

        return topicDistrib(paraPreprocBoW, self.ldaModel, K, useNTtopics, self.topickeys)


    def rankParagraph(self, paraFullText, paraPreprocBoW, d, K, useNTtopics, threshold, fast_power_method=True, topicVectors=None):
        """"""

        assert len(paraFullText)==(len(topicVectors) if topicVectors else len(paraPreprocBoW))

        if topicVectors:
            if not useNTtopics:
                topic_scores = topicVectors
            else:
                # use onlly top N topics
                topic_scores = []
                for tv in topicVectors:
                    max_n_idx = tv.argsort()[-useNTtopics:]
                    top_tv = np.zeros(tv.shape)
                    for i in max_n_idx:
                        top_tv[i] = tv[i]
                    topic_scores.append(top_tv)
                assert len(tv)==len(top_tv)
        else:
            topic_scores= topicDistrib(paraPreprocBoW, self.ldaModel, K, useNTtopics, self.topickeys)

        similarity_matrix = self._calculate_topic_similarity_matrix(topic_scores)

        query_relevance = np.zeros([len(topic_scores)] * 2)
        for i, (p,s) in enumerate(paraFullText):
            query_relevance[:,i] = s

        row_sum = query_relevance.sum(axis=1, keepdims=True)
        query_relevance = query_relevance / row_sum

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        Q = ( d * query_relevance )  + ( (1-d) * markov_matrix)

        scores = stationary_distribution(
            Q,
            increase_power=fast_power_method,
            normalized=False,
        )

        #print("scores")
        #print(scores)
        return scores


    def rankParagraphSearchMatrix(self, paraFullText, paraPreprocBoW, K, useNTtopics, topicVectors=None):
        """"""

        assert len(paraFullText)==(len(topicVectors) if topicVectors else len(paraPreprocBoW)), \
            "{}/{}".format(len(paraFullText), (len(topicVectors) if topicVectors else len(paraPreprocBoW)))

        if topicVectors:
            if not useNTtopics:
                topic_scores = topicVectors
            else:
                # use onlly top N topics
                topic_scores = []
                for tv in topicVectors:
                    max_n_idx = tv.argsort()[-useNTtopics:]
                    top_tv = np.zeros(tv.shape)
                    for i in max_n_idx:
                        top_tv[i] = tv[i]
                    topic_scores.append(top_tv)
                assert len(tv)==len(top_tv)
        else:
            topic_scores= topicDistrib(paraPreprocBoW, self.ldaModel, K, useNTtopics, self.topickeys)

        similarity_matrix = self._calculate_topic_similarity_matrix(topic_scores)

        query_relevance = np.zeros([len(topic_scores)] * 2)
        for i, (p,s) in enumerate(paraFullText):
            query_relevance[:,i] = s

        row_sum = query_relevance.sum(axis=1, keepdims=True)
        query_relevance = query_relevance / row_sum

        return similarity_matrix, query_relevance


    def getMarkovMatrix(self, similarity_matrix, threshold):
        """"""

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )
        return markov_matrix

    def getRankedScores(self, paraFullText, markov_matrix, query_relevance, d, fast_power_method=True):
        """"""
        Q = ( d * query_relevance )  + ( (1-d) * markov_matrix)

        scores = stationary_distribution(
            Q,
            increase_power=fast_power_method,
            normalized=False,
        )

        #print("scores")
        #print(scores)

        sortedIndex = np.argsort(scores)[::-1]

        return [paraFullText[i][0] for i in sortedIndex]




    def rank_sentences(
        self,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

        tf_scores = [
            Counter(self.tokenize_sentence(sentence)) for sentence in sentences
        ]

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    def sentences_similarity(self, sentence_1, sentence_2):
        tf_1 = Counter(self.tokenize_sentence(sentence_1))
        tf_2 = Counter(self.tokenize_sentence(sentence_2))

        similarity = self._idf_modified_cosine([tf_1, tf_2], 0, 1)

        return similarity

    def tokenize_sentence(self, sentence):
        tokens = tokenize(
            sentence,
            self.stopwords,
            keep_numbers=self.keep_numbers,
            keep_emails=self.keep_emails,
            keep_urls=self.keep_urls,
        )

        return tokens

    def _calculate_idf(self, documents):
        bags_of_words = []

        for doc in documents:
            doc_words = set()

            for sentence in doc:
                words = self.tokenize_sentence(sentence)
                doc_words.update(words)

            if doc_words:
                bags_of_words.append(doc_words)

        if not bags_of_words:
            raise ValueError('documents are not informative')

        doc_number_total = len(bags_of_words)

        if self.include_new_words:
            default_value = math.log(doc_number_total + 1)

        else:
            default_value = 0

        idf_score = defaultdict(lambda: default_value)

        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)

        return idf_score

    def _calculate_similarity_matrix(self, tf_scores):
        length = len(tf_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity



    def _calculate_topic_similarity_matrix(self, topic_scores):
        length = len(topic_scores)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity = self._topic_similarity(topic_scores, i, j)

                #print("{},{}\t{}".format(i,j,similarity))

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _topic_similarity(self, topic_scores, i, j):
        if i == j:
            #return 1
            return 0

        # Jensen-Shannon similarity measurer
        return JSD(topic_scores[i], topic_scores[j])
        #return max(1 - JSD(topic_scores[i], topic_scores[j]), 0)

    def _markov_matrix(self, similarity_matrix):

        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        similarity_matrix = 1 - (similarity_matrix / row_sum)
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        return similarity_matrix / row_sum

    def _markov_matrix_discrete(self, similarity_matrix, threshold):
        markov_matrix = np.zeros(similarity_matrix.shape)

        for i in range(len(similarity_matrix)):
            #columns = np.where(similarity_matrix[i] > threshold)[0]
            columns = np.where(similarity_matrix[i] < threshold)[0]
            markov_matrix[i, columns] = 1 / len(columns)

        #print( 1 / len(columns))
        #exit()
        return markov_matrix
