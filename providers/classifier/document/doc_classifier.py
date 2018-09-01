from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

class DocClassifier():

    def __init__(self, *args, **kwargs):
        pass
    
    def preprocessing(self):
        pass

    def create_model(self):
        pass

    def train(self):
        categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
        twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

        # print('twenty_train',twenty_train.target[:10])
        # print('target_names',twenty_train.target_names)
        # twenty_train.target_names['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
        # print(len(twenty_train.data),len(twenty_train.filenames))

        # count_vect = CountVectorizer()
        # X_train_counts = count_vect.fit_transform(twenty_train.data)
        # print(X_train_counts[:1])
        # print(X_train_counts[:1])
        # print(count_vect.vocabulary_.get(u'algorithm'))
        # print(X_train_counts)
        # pass

        # list of text documents
        text = ["hello i love you ai",'hello ai','hello blockchain']
        # create the transform
        vectorizer = CountVectorizer()
        # tokenize and build vocab
        vectorizer.fit(text)
        # summarize
        print(vectorizer.vocabulary_)
        # encode document
        # vector = vectorizer.transform(text)
        vector = vectorizer.fit_transform(text)
        # summarize encoded vector
        print(vector.shape)
        print(type(vector))
        print(vector.toarray())

    def classify(self):
        pass
