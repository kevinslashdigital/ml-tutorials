from providers.classifier.dog_cat_classifier_v2 import DogCatClassifier

if __name__ == "__main__":
    classifier = DogCatClassifier()
    # classifier.train()
    classifier.classify()

