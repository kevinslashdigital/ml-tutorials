from sklearn.datasets import fetch_20newsgroups
from libs.text import CountVectorizer
from libs.text import TfidfTransformer
from libs.text import HashingVectorizer
from libs.text import TfidfVectorizer

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

        print('===================== norm none')
        tf_transformer = TfidfTransformer(norm=False,sublinear_tf=False).fit(vector)
        X_train_tf = tf_transformer.transform(vector)
        print(X_train_tf.toarray())

        print('===================== l1')
        tf_transformer = TfidfTransformer(norm='l1',sublinear_tf=False).fit(vector)
        X_train_tf = tf_transformer.transform(vector)
        print(X_train_tf.toarray())

        print('===================== l2')
        tf_transformer = TfidfTransformer(norm='l2',sublinear_tf=False).fit(vector)
        X_train_tf = tf_transformer.transform(vector)
        print(X_train_tf.toarray())

        print('===================== l3')
        tf_transformer = TfidfTransformer(norm='max',sublinear_tf=False).fit(vector)
        X_train_tf = tf_transformer.transform(vector)
        print(X_train_tf.toarray())

        # tfidf_transformer = TfidfTransformer()
        # X_train_tfidf = tfidf_transformer.fit_transform(vector)

        # print(X_train_tfidf.toarray())

        # # create the transform
        # h_vectorizer = HashingVectorizer(n_features=5)
        # # encode document
        # h_vector = h_vectorizer.transform(text)
        # # summarize encoded vector
        # print(h_vector.shape)
        # print(h_vector.toarray())

        # # create the transform
        # tfid_vectorizer = TfidfVectorizer()
        # # encode document
        # tfid_vector = tfid_vectorizer.fit_transform(text)
        # # summarize encoded vector
        # print(tfid_vector.shape)
        # print(tfid_vector.toarray())
        pass

    def create_model(self):
        pass

    def train(self):
        self.preprocessing()

    def classify(self):
        pass
