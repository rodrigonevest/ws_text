import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from bertVectorizer import bertVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
nltk.download('rslp')
nltk.download('stopwords') 
nltk.download('punkt')

stop_words = nltk.corpus.stopwords.words('portuguese') 
stemmer = nltk.stem.RSLPStemmer()


class Pipeline:
    
    def __init__(self, path_dataframe) -> None:
        
        self.STOPWORDS = nltk.corpus.stopwords.words('portuguese')
        self.stemmer = nltk.stem.RSLPStemmer() 
        self.df = pd.read_excel(path_dataframe, usecols=['headline', 'class'])
        #self.nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
    
    def cosseno(self,x,y):
        
        dist = cosine(x,y)
        if np.isnan(dist):
            return 1
        return dist

    def get_representations(self):
        
        dic_tw = {'Binary'        : CountVectorizer(stop_words=stop_words, binary=True),
                  'TF'            : CountVectorizer(stop_words=stop_words),
                  'TF-IDF'        : TfidfVectorizer(stop_words=stop_words),
                  'Bert-Base'     : 'bert-base-multilingual-cased',
                  'Distilbert'    : 'distilbert-base-multilingual-cased',
                  'Bertimbau'     : 'neuralmind/bert-base-portuguese-cased',
                  }
        
        return dic_tw
    
    def get_models(self):
        
        models = {"MLP"        : MLPClassifier(),
                  "SVM"        : SVC(),
                  "KNN"        : KNeighborsClassifier(metric=self.cosseno),
                  "GNB"        : GaussianNB(),
                  "DTC"        : DecisionTreeClassifier()}
        
        return models
        
    def get_classification_report(self,y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        #df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
        aux = df.values.tolist()
        return aux

    def clean_text(self, data):
        
        dt = []
        for row in range(0, len(data)):
            aux = ''
            text_top = str(data.iloc[row,]) 
            top_sub = re.sub('[0-9][^w]', '' , text_top)
            top_sub = re.sub(r'\n', '' , top_sub)
            aux += ' ' + str(top_sub)
            dt.append(aux)     
        
        return dt

    def steming(self,dt):
        dt_st = []
        for st in dt:
            tx_st = [self.stemmer.stem(a) for a in st.split()]
            s = " "
            for tk in tx_st:
                s+= tk+" "
            dt_st.append(s.strip())
            
        return dt_st

    def main(self):
                
        X_col = 'headline'						# Column used to forecasting
        y_col = 'class'						# column name to dependent variable
        res_columns = ['precision', 'recall', 'f1-score', 'support']
        res_ind = ['0', '1', 'accuracy', 'macro avg', 'weighted avg']
                  
        mdls = self.get_models()
        representations = self.get_representations()
        
        for n_vec, vectorizor in representations.items():
            
            if n_vec == 'Bert-Base' or n_vec == 'Distilbert' or n_vec == 'Bertimbau':
                txts = self.df[X_col].to_list()
                model = SentenceTransformer(vectorizor)
                X = model.encode(txts)
            else:
                txt = self.clean_text(self.df[X_col])
                txts = self.steming(txt)
                matrix = vectorizor.fit_transform(txts)
                #XX = pd.DataFrame(matrix.todense(), columns=vect.get_feature_names(), index=df.index)                
                XX = pd.DataFrame(matrix.todense(), columns=vectorizor.get_feature_names())
                X = XX.values
            
            y = self.df[y_col].values
            entries =[]
            
            train_index = list(range(0, int(len(X) * 0.8)))
            test_index = list(range(int(len(X) * 0.8), len(X)))
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            """
            print("Tamanho", X.shape[0])
            print("0 - treino: ", np.count_nonzero(y_train == 0))
            print("1 - treino: ", np.count_nonzero(y_train == 1))
            
            print("\n0 - teste: ", np.count_nonzero(y_test == 0))
            print("1 - teste: ", np.count_nonzero(y_test == 1))
            
            """
            
            for name_clf, classifier in mdls.items():
                
                try:
                    clf = classifier
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    
                    res = self.get_classification_report(y_test, y_pred)
                    df_res = pd.DataFrame(res, columns=res_columns, index=res_ind)
                    df_res.to_csv('res_'+n_vec+'_'+name_clf+'.csv')                    
                    print(df_res)
                    
                    f1 = f1_score(y_test, y_pred, average='macro')
                    print('Representation: ', n_vec, ' - Classificador: ', name_clf, 'F1-Score: ', f1)
                    
                except:
                    print("Erro:", n_vec, name_clf)
        
        
if __name__ == '__main__':
    
    dados = 'headline_labels.xlsx'
    pipeline = Pipeline(dados).main()

    print(pipeline)