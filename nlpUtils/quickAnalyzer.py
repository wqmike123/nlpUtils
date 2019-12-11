#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text pre-analyzer:
    - wordCloud
    - wordCount
    - preprocess
    - posTag
    - sentiment
        - sentimentAnalyzer_blob
        - sentimentAnalyzer_dict
"""

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk import FreqDist,pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import ciseau
from nltk.stem.porter import PorterStemmer
# setProxy doiwnload()
from nlpUtils.tokenizer.unitok.configs import english as englishConfig
from nlpUtils.tokenizer.unitok import unitok as tok
import tensorflow as tf
import tensorflow_hub as hub
from nltk.stem import WordNetLemmatizer

# spacy
# error then: set https_proxy= | python -m spacy download en
#import spacy
#nlp = spacy.load('en_core_web_md')

# sentiment scoring
from textblob import TextBlob


# basic visualization
import seaborn as sns
#from sklearn.manifold import TSNE

# LDA
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
#from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.matutils import corpus2csc

# NMF
from sklearn.decomposition import NMF
import re


NNPLIST = ['NNP']

def wordCloud(string, maxWord = 200, maxFontSize = 40, scale = 3, title = None, saveDir = None, collocations = False):
    """
       given string, plot the word cloud
    """
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=maxWord,
        max_font_size=maxFontSize, 
        scale=scale,
	collocations = collocations,
	relative_scaling = 1.,
	repeat = False,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(string)

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)        
    if saveDir is None:
        plt.show()
    else:
        fig.savefig(saveDir)

def wordCount(string,top=None):
    """
        count the word distribution
    """
    if isinstance(string, str):
        string = word_tokenize(string)
    useful_words = [word for word in string if word not in stopwords.words('english')]
    freqList = FreqDist(useful_words)
    if top is None:
        top = len(freqList)
    return freqList.most_common(top)

class topicExtract_lda:
    
    def __init__(self,docs,nTopic = 20):
        """
            extract topic using LDA
            input:
                list of list of words, each list is a token of a doc
        """
        for i,idoc in enumerate(docs):
            if isinstance(idoc, str):
                docs[i] = word_tokenize(idoc)
        self.wordDict = Dictionary(docs)
        self.corpus_docs = [self.wordDict.doc2bow(doc) for doc in docs]
        corpus_csc = corpus2csc(self.corpus_docs)
        #tfidf_model = models.TfidfModel(self.corpus_docs)
        #tfidf_corpus = tfidf_model[self.corpus_docs]
        self.nmf = NMF(n_components = nTopic, random_state = 42)
        self.W = self.nmf.fit_transform(corpus_csc)
        
        self.topics = {'Topic '+ str(i):' '.join(list(self.get_topic_words(i)[1].values())) for i in range(nTopic)}
        self.lda2 = LdaModel(corpus=self.corpus_docs, id2word=self.wordDict, num_topics=nTopic, update_every=1, chunksize=1000, passes=4, random_state = 24)
        #self.lda2.show_topics(num_topics=-1, num_words=4)
        
            
    
    def get_topic_words(self, component_number):
        """
            NMF topics with a gensim corpus represented by component vectors
        """
        sorted_idx = np.argsort(self.W[:,component_number])[::-1][:5]
        component_words = {self.W[:, component_number][number]:self.wordDict[number] for number in sorted_idx[:5]}
        return sorted_idx, component_words
    
    def get_doc_components(self, doc_number):
        sorted_idx = np.argsort(self.nmf.components_[:,doc_number])[::-1][0:3]
        result = {number: self.nmf.components_[:,doc_number][number] for number in sorted_idx}
        return result
    
    def get_document_details(self, doc_number):
        results = []
        for item, val in self.get_doc_components(doc_number).items():
            print("document is composed of topic %d with weight %.4f" % (item, val))
            result = self.get_topic_words(item)[1]
            results.append(result)
        return results
    
    def show_lda(self, doc_num, threshold = 0.05, nWord = 5):
        topic_list = []
        for topic, weight in self.lda2[self.corpus_docs[doc_num]]:
            if weight > threshold:
                topic_list.append({(topic, weight):self.lda2.show_topic(topic, topn = nWord)})
        return topic_list    

    def showTopics(self, nWord= 4):
        output = self.lda2.show_topics(num_topics=-1, num_words = nWord)
        for i in output:
            print(i)

def sentimentAnalyzer_blob(string):
    """
        sentiment analyzer using blob
    """    
    return {'polarity':TextBlob(string).sentiment.polarity, 
            'subjectivity':TextBlob(string).sentiment.subjectivity}

def sentimentAnalyzer_dict(string):
    """
        sentiment analyzer using dict map
    """
    pass
    
def posTag(string, request = None, ):
    """
        get pos tagging, return in pairs
        future: using pos_tag_sents for efficiency
    """
    if isinstance(string, str):
        string = word_tokenize(string)
    res = pos_tag(string, tagset = 'universal')
    if request is not None:
        res = [pair for pair in res if pair[1] == request]
    return res


def preprocess(string):
    """
    0. Remove Unicode Strings and Noise
    1. Replcae Slang and Abbreviations
    2. Replace Contractions
    #3. Remove Numbers
    4. Replace Repetitions of Punctuation
    5. Replace Negations with Antonyms
    6. Remove Punctuation
    7. Handling Capitalized Words
    8. Lowercase
    #9. Remove Stopwords 
    10. Replace Elongated Words
    11. Spelling Correction
    #12. Part of Speeck Tagging
    #13. Lemmatizing 
    #14. Stemming
    """


#%% clearing functions
    
def join_not(text):
    """
        replace not a -? not_a
    """
    replace_these = re.findall(r'not\s+\w+', text)
    for item in replace_these:
        tmp = item.replace(' ', '_')
        text = text.replace(item, tmp)
    return text

def regex_replace(texts, substitute = '', regex_pattern = r"[^a-zA-z' ]|https?:\/\/.*[\r\n]*" ):
    """
        delete http
    """
    pattern = re.compile(regex_pattern)
    
    result = []
    for text in texts:
        replaced = pattern.sub(substitute, text)
        replaced = replaced.replace(r"\n", '').replace('  ', ' ').lower().strip()
        result.append(replaced)
    
    return result

#def clean_text(texts, noisy = {}):
#    pipeline = nlp.pipe(texts, batch_size = 1000)
#    
#    # removes punctuation and pronouns, random words, normalizes words by lemmatization
#    word_lists=[]
#    for sent in pipeline:
#        words = []
#        for word in sent:
#            if (word.pos_ != 'PUNCT' and word.lemma_ != '-PRON-') and \
#            (not word.is_space and not word.is_stop):
#                if word.lemma_ not in {}:
#                    words.append(word.lemma_)
#        word_lists.append(words)
#    return word_lists

def preprocessPipeline(sentence, stem = False,lemmazation = False, removeName = False, stopword = False,lower = False, padding=0,clean = False, tokenTool = 'ciseau'):
    """
        preprocess the raw text input
    """
    if lower and not removeName:
        sentence = sentence.lower()
    if tokenTool == 'ciseau':
        sentence = ciseauTokenizer(sentence)
    elif tokenTool == 'unitok':
        sentence = unitokTokenizer(sentence)
    elif tokenTool == 'simple':
        sentence = sentence.split(' ')
    else:
        raise Exception("Tool not defined")
    if removeName:
        sentence = nameRemover(sentence,lower)          
    #if replaceword_list:
	#    for i,itoken in enumerate(sentence):
	#        if itoken in replaceword_dict.keys():
	#           sentence[i] = replaceword_dict[itoken]
    if lemmazation:
        sentence = lemmatizer(sentence)
    if clean:
        sentence = strCleaner(sentence,kind='list',TREC=lower)
    if stopword:
        sentence = removeStopword(sentence)
    if stem:
        sentence = porterStemmer(sentence,'list')
    if padding>0:
        sentence = paddingSentence(sentence,padding)
    return sentence


def nameRemover(sentence,lower = False):
    if lower:
        sentence = [i[0].lower() for i in pos_tag(sentence) if i[1] not in NNPLIST]
    else:
        sentence = [i[0] for i in pos_tag(sentence) if i[1] not in NNPLIST]
    return sentence

def unitokTokenizer(text):
    '''Tokenises using unitok http://corpus.tools/wiki/Unitok the text. Given
    a string of text returns a list of strings (tokens) that are sub strings
    of the original text. It does not return any whitespace.
    String -> List of Strings
    '''
    tokens = tok.tokenize(text, englishConfig)
    return [token for tag, token in tokens if token.strip()]

def ciseauTokenizer(sentence):
    return ciseau.tokenize(sentence)

def removeStopword(sentlist):
    en_stop = stopwords.words('english')
    #for sen in res:
    #    sen = [i for i in sen if not i in en_stop]
    sentlist = [i for i in sentlist if not i in en_stop]
    return sentlist

def porterStemmer(sentlist,kind="sentence"):
    p_stemmer = PorterStemmer()
    if kind=="sentence":
        return p_stemmer.stem(sentlist)
    if kind=="list":
        sentlist = [p_stemmer.stem(i) for i in sentlist]
        return sentlist
   
def lemmatizer(words):
    lem = WordNetLemmatizer()
    wordList = [lem.lemmatize(i) for i in words]
    return wordList
    
def strCleaner(string,kind="sentence",TREC = True):
    if kind=="sentence":
        return singleStrCleaner(string,TREC)
    else:
        res = []
        for i,istr in enumerate(string):
            temp = singleStrCleaner(istr,TREC)
            if temp != '':
                res.append(temp)     
        return res

def singleStrCleaner(string,TREC):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip() if TREC else string.strip().lower() 
    
def paddingSentence(sentlist,length,paddingWord = "<pad>"):
    nsent = len(sentlist)
    if nsent<=length:
        sentlist.extend([paddingWord]*(length-nsent))
    return sentlist[:length]  

def text2Vec(strList, session = None, embed = None):
    if embed is None:
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        embed = hub.Module(module_url)
    if session is None:
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            newsEmbedding = session.run(embed(strList))
    else:
        newsEmbedding = session.run(embed(strList))
    return newsEmbedding


#def simpleAnalysis(x):
#    fut = ['MD']
#    doc = nlp(x)
#    sentiment = 0
#    tense = 0
#    for token in doc:
#        #print(token.pos_)
#        if token.tag_ in fut:
#            tense += 1
#        sentiment += token.sentiment
#    return tense,sentiment


def word2vec(senlist, word2vec, vocDoc=None, learnMissing=False, min_df=10):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    res = []
    if isinstance(list(word2vec.values())[0], list):
        k = len(list(word2vec.values())[0])
    else:
        k = 1
    if vocDoc:
        for word in senlist:
            if word not in word2vec and vocDoc[word] >= min_df:
                word2vec[word] = np.random.uniform(-0.25, 0.25, k)
            if word in word2vec:
                res.append(word2vec[word])
        return np.array(res)
    else:
        for word in senlist:
            if word not in word2vec and learnMissing:
                word2vec[word] = np.random.uniform(-0.25, 0.25, k)
            if word in word2vec:
                res.append(word2vec[word])
        return np.array(res)
