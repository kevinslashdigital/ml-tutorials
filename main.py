from providers.classifier.image.dog_cat_classifier_v2 import DogCatClassifier
from providers.classifier.document.sentence_classifier import SentenceClassifier

if __name__ == "__main__":
    classifier = SentenceClassifier()
    classifier.train()
    # classifier.classify()   

