import pandas as pd
import getpass
import os, ssl
import numpy as np
from sklearn.model_selection import train_test_split
from nlpUtils.quickAnalyzer import unitokTokenizer,singleStrCleaner

def setProxy():
    """
       set proxy for external data download
    """
    user = getpass.getuser()
    password = getpass.getpass("proxy password:")
    # origin: os.environ["https_proxy"] = f"http://{user}@{password}@proxypac.ubs.net"

    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["https_proxy"] = "https://%s:%s@inet-proxy-a.adns.ubs.net:8080" % (user, password)

def getRelevantDF(df,target,columns = None):
    """
        given the topic list, we would like to derive the
    """
    if columns:
        df = df[columns]
    return df[df.apply(matchList,args = (target,))]


def matchList(x,target):
    for itopic in target:
        if itopic in x:
            return True
    return False


def toMap(df,key,val):
    return df[[key,val]].set_index(key).to_dict()[val]


def label2Binary(x):
    from sklearn.preprocessing import LabelBinarizer
    if len(x.shape)==1:
        return LabelBinarizer().fit_transform(x.reshape([-1,1]))
    else:
        return LabelBinarizer().fit_transform(x)


def splitData(x,y,ratio = [0.7, 0.1, 0.2],random_state = 711):
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=ratio[-1], random_state=random_state)
    testRatio = ratio[1] / (ratio[0] + ratio[1])
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=testRatio, random_state=random_state)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def batch(dataList,batchsize):
    n = len(dataList[0])
    resList = [[] for i in range(len(dataList))]
    nbatch = int(n/batchsize) + 1
    for i,idata in enumerate(dataList):
        for step in range(nbatch):
            offset = (step * batchsize)
            data = idata[offset:offset+batchsize]
            resList[i].append(data)
    return resList

class EmbeddingModel:
    """
        simplified gensim model 
    """
    def __init__(self,gensimModel, tokenizer = None):
        w2v = {}
        w2id = {}
        ct = 0
        self.vector_size = gensimModel.wv.vector_size
        for idx,iword in enumerate(gensimModel.wv.index2entity):
            if tokenizer:
                wordKey = tokenizer(iword)
            else:
                wordKey = iword
            if wordKey in w2v:
                continue
            w2v[wordKey] = gensimModel.wv[iword]
            w2id[wordKey] = ct
            ct += 1
        if '<unk>' not in w2v:
            w2v['<unk>'] = np.zeros(self.vector_size)
            w2id['<unk>'] = ct
            ct += 1
            
        self._w2v = w2v
        self._w2id = w2id
        self._nVoc = ct
        
        
    def getVec(self,word):
        
        if word in self._w2v:
            return self._w2v[word]  
        else:
            return None
            
    def getIndex(self,word):
        
        if word in self._w2id:
            return self._w2id[word]  
        else:
            return None
        
    def update(self,word, vector = None):
        
        assert word not in self._w2v
        if vector is None:
            vector = np.zeros(self.vector_size)
        self._w2v[word] = vector
        self._w2id[word] = self._nVoc
        self._nVoc += 1
        return self._nVoc - 1
   
    @property
    def unkown(self):
        return '<unk>'
    
    @property
    def index2entity(self):
        return list(self._w2v.keys())
    
    def getW(self):
        max_voc = len(self._w2v)
        w = []
        for i in range(max_voc):
            w.append( self.getVec(self.index2entity[i]))
        return np.array(w).astype(np.float32)
    
    def updateW(self,w):
        assert w.shape[0] == self._nVoc
        assert w.shape[1] == self.vector_size
        for i in range(w.shape[0]):
            self._w2v[self.index2entity[i]] = w[i,:]
    
    def removeNoisy(self,docs):
        """
            remove unfrequent words by docs
        """
        
    
def loadw2vModel(filePath):
    """
        return the gensim model
    """
    import gensim
    return gensim.models.Word2Vec.load(filePath)

    
def tokens2Ind(tokens, textModel, allowAddWord = True):
    """
        given a list of tokens, return list of index
    """
    wordEmbed = []
    for token in tokens:
        idx = textModel.getIndex(token)
        if idx is None:
            if allowAddWord:
                idx = textModel.update(token)
            else:
                idx = textModel.getIndex(textModel.unkown)
        wordEmbed.append(idx)        
    return wordEmbed

def prepareInputs(tokens, wordModel, allowAddWord = True):
    """
        given signal sentence, provide wordEmbedding, CharEmbedding, sentenceEmbedding (otpional)

    """
    wordEmbedding = tokens2Ind(tokens,wordModel,allowAddWord)
    return wordEmbedding
    
    
def getSentDict(dictDir, stem='none'):
    mcDict = pd.read_excel(dictDir)
    mcDict.Word = mcDict.Word.apply(str).apply(str.lower)  # str.encode('utf-8').
    mcDict.Word = mcDict.Word.apply(singleStrCleaner, args=(False,))
    if stem == 'nltk':
        p_stemmer = PorterStemmer()
        mcDict.index = mcDict.Word.apply(p_stemmer.stem)
    elif stem == 'unitok':
        mcDict.index = mcDict.Word.apply(unitokTokenizerWrap)
    else:
        mcDict.index = mcDict.Word
    mcDict['senti'] = 0.
    mcDict.loc[mcDict[mcDict.Negative > 0].index, 'senti'] = -1.
    mcDict.loc[mcDict[mcDict.Positive > 0].index, 'senti'] = 1.
    mcDict = mcDict[['senti']].to_dict()['senti']
    return mcDict

def unitokTokenizerWrap(x):
    res = unitokTokenizer(x)
    if len(x) > 0:
        return res[0]
    else:
        return x
