from providers.classifier.image.dog_cat_classifier_v2 import DogCatClassifier
from providers.classifier.document.sentence_classifier import SentenceClassifier
from providers.regression.linear_regression.house_price_prediction_us import HousePredictor

if __name__ == "__main__":
    regression = HousePredictor()
    model = regression.train()
    # classifier.classify()   
    # regression.predict(model,[[3219,1479,4.1473]])


