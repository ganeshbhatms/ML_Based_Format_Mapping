import re
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder


def Ngrams(text, n=2):
    """
    Function to find ngram
    :param text: str
    :param n: int
    :return: list
    """
    remove = string.punctuation
    remove = remove.replace("#", "")
    pattern = r"[{}]".format(remove)
    text = re.sub(pattern, r'', text)
    ngrams = zip(*[text[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


count_vect = CountVectorizer(analyzer=Ngrams)
tfidf_transformer = TfidfTransformer()
le = LabelEncoder()


def CosineSimilarity(text1, text2):
    """
    Function to find cosine similarity
    :param text1: list
    :param text2: list or str
    :return: list or str
    """
    if isinstance(text2, list):
        cos = dict()
        for i in range(len(text2)):
            targetText = text2[i].lower()
            sourceText = text1.lower()
            vec1 = [sourceText, targetText]
            vec2 = count_vect.fit_transform(vec1)
            tfidf = tfidf_transformer.fit_transform(vec2)
            similarity = (tfidf * tfidf.T).A[0, 1]
            cos.update({text2[i]: similarity})
        return [max(cos, key=cos.get), int(round(max(cos.values()) * 100))]
    elif isinstance(text2, str):
        targetText = "".join([item.lower() for item in text2])
        sourceText = "".join([item.lower() for item in text1])
        vec1 = [sourceText, targetText]
        vec2 = count_vect.fit_transform(vec1)
        tfidf = tfidf_transformer.fit_transform(vec2)
        similarity = (tfidf * tfidf.T).A[0, 1]
        return int(round(similarity * 100))


def FitModel(x, y):
    """
    Function to train k-nearest Neighbors Classifier
    :param x: list
    :param y: list
    :return: k-nearest Neighbors Classifier
    """
    encode = le.fit_transform(y)

    X_train_counts = count_vect.fit_transform(x)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    clf = knn.fit(X_train_tfidf, encode)
    return clf
