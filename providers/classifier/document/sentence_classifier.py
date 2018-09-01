from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class SentenceClassifier():

    def __init__(self, *args, **kwargs):
        pass
    
    def preprocessing(self):
        # list of text documents
        text = ["hello i love you ai",'hello ai ai','hello blockchain i you']
        # create the transform
        vectorizer = CountVectorizer()
        # tokenize and build vocab
        vectorizer.fit(text)
        # summarize
        print(vectorizer.vocabulary_)
        # encode document
        vector = vectorizer.transform(text)
        # vector = vectorizer.fit_transform(text)
        # vector = vectorizer.fit(text)
        # summarize encoded vector
        # print(vector.shape)
        # print(type(vector))
        print(vector.toarray())
        # print(vector)
        tf_transformer = TfidfTransformer(use_idf=False).fit(vector)
        X_train_tf = tf_transformer.transform(vector)
        print(X_train_tf.toarray())

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(vector)

        print(X_train_tfidf.toarray())
        pass

    def create_model(self):
        pass

    def train(self):
        self.preprocessing()

    def classify(self):
        pass
