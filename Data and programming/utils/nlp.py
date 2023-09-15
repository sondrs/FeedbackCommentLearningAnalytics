import os
import re
from numpy import array, zeros
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_array, hstack, vstack
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from joblib import dump, load
from tqdm import tqdm
from datetime import datetime
import multiprocessing
from tqdm.notebook import tqdm
from multiprocessing import Pool
import IProgress
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('tableau-colorblind10')




def getLargeLemmatizer(getLemmas=False,
                      writeToDisk=False,
                      importLemmas=True):
    if getLemmas:
        updater = LemmaUpdater()
        from datetime import datetime
        from tqdm import tqdm
        path = 'resources/talk-of-norway/annotations/'
        totalNumber=251000
        batchSize=10000
        main_bar = tqdm(total=totalNumber/batchSize,desc='Learning lemmatizations',leave=True)
        main_bar.update(0)

        for i in range(round(totalNumber/batchSize)):
            print('loading batch '+str(i)+' of '+str(totalNumber/batchSize))
            tonAnnot = load_corpus(path,
                                   i*batchSize,
                                   i*batchSize-batchSize,
                                   'tale',
                                   '.tsv',
                                   '\n',
                                   lower=True)

            sub_bar = tqdm(total=len(tonAnnot)-1, desc="Processing batch", leave=True)
            sub_bar.update(0)
            batch_beg = datetime.now()
            for i,speech in enumerate(tonAnnot):
                processed = [item.split('\t') for item in speech]
                updater.update(processed)
                if i%1000==0:
                    sub_bar.update(i)
            batch_end = datetime.now()
            sub_bar.close()
            lemmaReg = updater.getLemmas()
            print('spent:',batch_end-batch_beg)
            print('n keys:',len(lemmaReg.keys()))
            print('n lemmas:',len(set(lemmaReg.values())))
            main_bar.update(i)
    if writeToDisk:
        print('writing to disk')
        filename = ('trainedLemmatizer'
                    '.csv')
        pd.DataFrame(lemmaReg,index=['lemmatized']).T.reset_index().to_csv('resources/fittedModels/'+filename,index=False)
        print('done, and written to disk as '+filename)
    if importLemmas:
        import pandas as pd
        lemmaReg = dict(zip(pd.read_csv(
            'resources/fittedModels/trainedLemmatizer.csv')['index'],pd.read_csv(
            'resources/fittedModels/trainedLemmatizer.csv')['lemmatized']))
    return lemmaReg


def preprocessFunction(corpusToProcess):
    
    from collections import Counter
    import re
    from utils.nlp import getLargeLemmatizer
    lemmaReg = getLargeLemmatizer()
    
    with open('resources/fittedModels/stopwords_from_corpus.txt','r') as stopfile:
            stops = stopfile.read().split('\n')
    
    def stopTester(w):
        """taking a string consiting of one word and possibly punctuation
        and returning the word and if present the punctuation 
        (.,!? or combinations) 
        as two elements of a list. Additionally, if the word has a lemmatized
        form in the lemmaReg input, that form is returned"""
        w = list(w)
        t=[]
        while w and w[-1] in ['.',',','!','?']:
            t.append(w.pop())
        w = ''.join(w)
        #
        t = ''.join(t)
        return [w,t]

    def tokenizer(doc):
        """taking a string containing a sentence/document,
        splitting on words and removing special characters"""
        removeChars = '[^A-Za-z0-9.æäöøåÆØÅÄÖ]'
        #removeChars1 = '[^A-Za-z0-9.]'
        tokens = []
        for word in doc.lower().split():
            if word[-1:].isalnum():
                word = re.sub(removeChars,'',word)
                tokens.append(word) 
            else:
                words = stopTester(word)
                word = re.sub(removeChars,'',words[0])
                tokens.append(word)
                tokens.append(words[1])     
        return tokens

    def wordCounter(doc):
        """"taking a tokenized document, returning 
        individual word count"""
        return Counter(doc)

    #creating the preprocessing function to perform all preprocessing 
    #tasks with one document
    def preprocess(doc,stops,lemmaReg):
        """taking a non-tokenized document and a list of stopwords,
        returning a preprocessed (tokenized) document"""
        if type(doc) == str:

            return (' ').join([str(lemmaReg.get(word,word)) for word in tokenizer(doc) if word 
                    and word not in stops])
        else:
            return doc
    
    #from tqdm import tqdm
    #main_bar = tqdm(total=len(corpusToProcess),desc='cleaning, tokenizing, removing stopwords, lemmatizing',leave=True)
    #main_bar.update(0)
    #from datetime import datetime
    #t0=datetime.now()
    corpus_preprocessed = []
    for i,doc in enumerate(corpusToProcess):
        corpus_preprocessed.append((preprocess(doc,stops,lemmaReg)))
    #    if i%10000 == 0:
    #        main_bar.update(i)
    #t1=datetime.now()
    #main_bar.close()
    #print('done, samples:',len(corpusToProcess))
    #print('time:',t1-t0)
    return corpus_preprocessed


def stopTester(w):
        """taking a string consiting of one word and possibly punctuation
        and returning the word and if present the punctuation 
        (.,!? or combinations) 
        as two elements of a list. Additionally, if the word has a lemmatized
        form in the lemmaReg input, that form is returned"""
        w = list(w)
        t=[]
        while w and w[-1] in ['.',',','!','?']:
            t.append(w.pop())
        w = ''.join(w)
        #
        t = ''.join(t)
        return [w,t]

def tokenizer(doc):
    """taking a string containing a sentence/document,
    splitting on words and removing special characters"""
    
    from utils.nlp import stopTester
    
    removeChars = '[^A-Za-z0-9.æäöøåÆØÅÄÖ]'
    #removeChars1 = '[^A-Za-z0-9.]'
    tokens = []
    for word in doc.lower().split():
        if word[-1:].isalnum():
            word = re.sub(removeChars,'',word)
            tokens.append(word) 
        else:
            words = stopTester(word)
            word = re.sub(removeChars,'',words[0])
            tokens.append(word)
            tokens.append(words[1])     
    return tokens



def loadVectorizerSVD(readVectorizer=True,
                      readSVD=True,
                      comps=300,
                      lenStr=16000):
    tfidf_matrix=np.array([0])
    
    
    if readVectorizer:
        vectorizer = load('resources/fittedModels/fittedTfidfVectorizer_'
             +str(lenStr)
             +'.joblib')
        #print('read trained TfidfVectorizer from disk') 

    if readSVD:
        svd_tfidf = load('resources/fittedModels/fittedSVD_'
             +str(lenStr)
             +'_'
             +str(comps)
             +'.joblib')
        #print('read fitted svd to reduce Tfidf Matrix from disk')
    return vectorizer, svd_tfidf





class LabelModelTrainer:
    """a collection of functions to take an iterable holding 
    training labels, an iterable holding individual document 
    representations in array shape and returning a trained 
    model for label prediction"""
    def __init__(self, 
                 label='', 
                 labels='', 
                 representations=[],
                 vocabSize=16000,
                 nComps=300
                 ):#, svd1=None, a=False, svd2=None, b=False):
        from numpy import array
        from scipy.sparse import csr_matrix
        
        
        if label and type(labels)!=str and type(representations)!=list:
            self.label = label
            self.labels = labels
            self.representations = representations
            self.labelCentroid, self.othersCentroid, self.mask = self.learnCentroids() #getCentroids
            self.boolLabels = [1 if self.label in item else 0 for item in [[item] if not type(item)==list else item for item in self.labels]]
        elif label:
            self.label = label
            self.vocabSize = vocabSize
            self.nComps = nComps
            self.labelCentroid, self.othersCentroid = self.loadCentroids(self.vocabSize,
                                                                         self.nComps,
                                                                        self.label)
        else:
            pass
   

    def learnCentroids(self): #getCentroids
        from numpy import array, zeros
        from scipy.sparse import csr_matrix
        from sklearn.preprocessing import MinMaxScaler

        representations = self.representations
       
        labeled = self.labels
        
        mask = array([(self.label in item) if type(item)==list else False for item in labeled])
        if sum(mask)==0:
            labelCentroid = zeros(representations.shape[1])+99
            othersCentroid = zeros(representations.shape[1])-99
            print('no items found with mask on '+str(self.label))
        else:
            labelCentroid = representations[mask].mean(axis=0)
            othersCentroid = representations[~mask].mean(axis=0)

        return labelCentroid, othersCentroid, mask
    
    def loadCentroids(self, vocabSize, nComps, label):
        if nComps!=0:
            rep = 'svd'
        else:
            rep = 'tfidf'
        labelCentroid = load('resources/trainedClassifiers/centroids_vocabSize_'+str(vocabSize)+'_all.joblib')[rep][label]['labelCentroid']
        othersCentroid= load('resources/trainedClassifiers/centroids_vocabSize_'+str(vocabSize)+'_all.joblib')[rep][label]['othersCentroid']
        return labelCentroid, othersCentroid
        
    def getLabelScoresCosine(self, representations):
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        if type(representations) != np.ndarray:
            representations = representations.toarray()
        labelScores = MinMaxScaler(feature_range=(-1,1)).fit_transform(np.asarray(
            representations.dot((
            self.labelCentroid - self.othersCentroid).reshape(-1,1))))
        
        self.probabilities = labelScores        

        return labelScores

    def predictCosine(self, representations, threshold=0.5):        
        labelScores = self.getLabelScoresCosine(representations)
        predictions = [[self.label] if score>threshold else [None] for score in labelScores]
        self.predictions = predictions
        self.representationsTest = representations
        return predictions
    
    def getLabelScoresEuclidean(self, representations):
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        if type(representations) != np.ndarray:
            representations = representations.toarray()

        distances_to_label = np.linalg.norm(representations - self.labelCentroid, axis=1)
        distances_to_others = np.linalg.norm(representations - self.othersCentroid, axis=1)
        
        similarity_label = 1 / (1 + distances_to_label)
        similarity_others = 1 / (1 + distances_to_others)
        
        scores = similarity_label / (similarity_label + similarity_others)
        scores = MinMaxScaler(feature_range=(-1,1)).fit_transform(scores.reshape(-1,1))
         
        self.probabilities = scores  

        return scores
    
    def predictEuclidean(self, representations, threshold=0.5):
               
        scores = self.getLabelScoresEuclidean(representations)
        
        predictions = [[self.label] if score >= threshold else [None] for score in scores]
        self.representationsTest = representations
        self.predictions = predictions
        return predictions
    
    def trainKnn(self, k=3):
        
        from sklearn.neighbors import KNeighborsClassifier

        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.representations, self.boolLabels)
    
    def predictKnn(self, representations):
        
        if not hasattr(self, 'knn'):
            raise ValueError("train knn first")

        predictions = self.knn.predict(representations)
        
        label_predictions = [[self.label] if pred == 1 else [None] for pred in predictions]
        
        return label_predictions
    
    def predictKnnProb(self, representations):
        
        if not hasattr(self, 'knn'):
            raise ValueError("train knn first")

        predictionsProb = self.knn.predict_proba(representations)[:,1]
        self.probabilities = predictionsProb        
        return predictionsProb

    

    

def productionPreds(reps):
    if reps.shape[1]<15000:
        nComps = reps.shape[1]

    predTypes = ['sentiment','topic']
    labels={}
    labels['sentiment'] = ['Positive', 'Negative']
    labels['topic'] = ['administrativt', 'digitalt', 'eksamen', 
                       'foreleser', 'karakter', 'korona', 
                       'pensum', 'språk', 'undervisningsopplegget']

    resultsDF = pd.DataFrame(index=range(reps.shape[0]))
    resultsDF['sentiment_scored'] = [[None]]*resultsDF.shape[0]
    resultsDF['topic_scored'] = [[None]]*resultsDF.shape[0]

    for lt in ['sentiment','topic']:
        resCol = lt+'_scored'
        for l in labels[lt]:
            LMT = LabelModelTrainer(label=l
                            , nComps=nComps
                           )
            predicted = LMT.predictEuclidean(reps,0.5)
                            
            resultsDF[resCol] = [item+predicted[i] for i,item in enumerate(resultsDF[resCol])]
        
        if lt=='sentiment':
            predicted = []
            for cell in resultsDF[resCol]:
                result = ''
                if 'Positive' in cell:
                    if not 'Negative' in cell:
                        result = 'Positive'
                if 'Negative' in cell:
                    if not 'Positive' in cell:
                        result = 'Negative'
                predicted.append(result.split())
            resultsDF[resCol] = predicted
        if lt=='topic':
            resultsDF[resCol] = [pd.Series(cell).dropna().values 
                                for cell in resultsDF[resCol]]
        
        resultsDF = resultsDF.replace(np.nan, '');
        
    return resultsDF



class Values:
    def __init__(self):
        self.validTopics = ['administrativt','digitalt','eksamen','foreleser','karakter',
                            'korona','pensum','språk','undervisningsopplegget']
        self.validSentiments = ['Positive','Negative']
        self.validPredictions = ['sentiment','topic']
        self.validLabels = {'sentiment':self.validTopics,
                            'topic':self.validSentiments}
        
        
