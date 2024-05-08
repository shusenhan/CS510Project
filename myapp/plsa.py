from nltk.corpus import stopwords
from numpy import zeros, int8, log
from pylab import random
# import sys
import jieba
import re
# import time
import codecs
import heapq

def plsa(text1, text2):
    # segmentation, stopwords filtering and document-word matrix generating
    # [return]:
    # N : number of documents
    # M : length of dictionary
    # word2id : a map mapping terms to their corresponding ids
    # id2word : a map mapping ids to terms
    # X : document-word matrix, N*M, each line is the number of terms that show up in the document
    def preprocessing(Inputdocs):

        # read the stopwords file
#         file = codecs.open(stopwordsFilePath, 'r', 'utf-8')
#         stopwords = [line.strip() for line in file]
#         file.close()
        stop_words_set = set(stopwords.words('english'))

        documents = [document.strip() for document in Inputdocs]


        # number of documents
        N = len(documents)

        wordCounts = [];
        word2id = {}
        id2word = {}
        currentId = 0;
        # generate the word2id and id2word maps and count the number of times of words showing up in documents
        for document in documents:
            segList = jieba.cut(document)
            wordCount = {}
            for word in segList:
                word = word.lower().strip()
                if len(word) > 1 and word not in stop_words_set and not re.search('[0-9]', word):
                    if word not in word2id.keys():
                        word2id[word] = currentId;
                        id2word[currentId] = word;
                        currentId += 1;
                    if word in wordCount:
                        wordCount[word] += 1
                    else:
                        wordCount[word] = 1
            wordCounts.append(wordCount);

        # length of dictionary
        M = len(word2id)

        # generate the document-word matrix
        X = zeros([N, M], int8)
        for word in word2id.keys():
            j = word2id[word]
            for i in range(0, N):
                if word in wordCounts[i]:
                    X[i, j] = wordCounts[i][word];

        return N, M, word2id, id2word, X

    def initializeParameters():
        for i in range(0, N):
            normalization = sum(lamda[i, :])
            for j in range(0, K):
                lamda[i, j] /= normalization;

        for i in range(0, K):
            normalization = sum(theta[i, :])
            for j in range(0, M):
                theta[i, j] /= normalization;

    def EStep():
        for i in range(0, N):
            for j in range(0, M):
                denominator = 0;
                for k in range(0, K):
                    p[i, j, k] = theta[k, j] * lamda[i, k];
                    denominator += p[i, j, k];
                if denominator == 0:
                    for k in range(0, K):
                        p[i, j, k] = 0;
                else:
                    for k in range(0, K):
                        p[i, j, k] /= denominator;

    def MStep():
        # update theta
        for k in range(0, K):
            denominator = 0
            for j in range(0, M):
                theta[k, j] = 0
                for i in range(0, N):
                    theta[k, j] += X[i, j] * p[i, j, k]
                denominator += theta[k, j]
            if denominator == 0:
                for j in range(0, M):
                    theta[k, j] = 1.0 / M
            else:
                for j in range(0, M):
                    theta[k, j] /= denominator

        # update lamda
        for i in range(0, N):
            for k in range(0, K):
                lamda[i, k] = 0
                denominator = 0
                for j in range(0, M):
                    lamda[i, k] += X[i, j] * p[i, j, k]
                    denominator += X[i, j];
                if denominator == 0:
                    lamda[i, k] = 1.0 / K
                else:
                    lamda[i, k] /= denominator

    # calculate the log likelihood
    def LogLikelihood():
        loglikelihood = 0
        for i in range(0, N):
            for j in range(0, M):
                tmp = 0
                for k in range(0, K):
                    tmp += theta[k, j] * lamda[i, k]
                if tmp > 0:
                    loglikelihood += X[i, j] * log(tmp)
        return loglikelihood

    # output the params of model and top words of topics to files
    def output(topicWordsNum):
        topicwords = {}
        for j in range(K):
            topicwords[j] = {}
            temp = [(-theta[j,t],t) for t in range(M)]
            heapq.heapify(temp)
            for _ in range(topicWordsNum):
                _,t = heapq.heappop(temp)
                topicwords[j][id2word[t]] = theta[j,t]
        res = [{} for _ in range(N+1)]
        for i in range(0, N):
            for j in range(0, K):
                res[i][j] = {}
                res[i][j]['prob'] = lamda[i,j]
                res[i][j]['words'] = topicwords[j]

        for j in range(K):
            res[-1][j] = {'prob':1}
            for i in range(N):
                res[i][j]['prob'] *= lamda[i,j]
            res[-1][j]['words'] = topicwords[j]

        return res

    # set the default params and read the params from cmd
    Inputdocs = [text1,text2]
#     stopwordsFilePath = 'stopwords.dic'
    K = 5    # number of topic
    maxIteration = 30
    threshold = 10.0
    topicWordsNum = 20
    # docTopicDist = 'docTopicDistribution.txt'
    # topicWordDist = 'topicWordDistribution.txt'
    # dictionary = 'dictionary.dic'
    # topicWords = 'topics.txt'


    # preprocessing
    N, M, word2id, id2word, X = preprocessing(Inputdocs)

    # lamda[i, j] : p(zj|di)
    lamda = random([N, K])

    # theta[i, j] : p(wj|zi)
    theta = random([K, M])

    # p[i, j, k] : p(zk|di,wj)
    p = zeros([N, M, K])

    initializeParameters()

    # EM algorithm
    oldLoglikelihood = 1
    newLoglikelihood = 1
    for i in range(0, maxIteration):
        EStep()
        MStep()
        newLoglikelihood = LogLikelihood()
        # print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLoglikelihood))
        if(oldLoglikelihood != 1 and newLoglikelihood - oldLoglikelihood < threshold):
            break
        oldLoglikelihood = newLoglikelihood

    topicword = output(topicWordsNum)

    # 3*5矩阵，3 rows前两个是text1，text2的popular topic，第三个是common popular topic。每个row里面的五个element是五个tuple （这个topic最popular的词，这个词在这个topic的概率）
    return topicword