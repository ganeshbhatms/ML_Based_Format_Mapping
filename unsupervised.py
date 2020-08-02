import pandas as pd
import json
import numpy as np
from pickle5 import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

import re
import string

from flask import request, Flask, jsonify

trainData = pd.DataFrame(columns=['sourceField'])
targetData = []

app = Flask(__name__)

@app.route('/train/format/learn', methods=['POST'])
def supervisedLearning():
    def ngrams(text, n=2):
        remove = string.punctuation
        remove = remove.replace("#", "") # don't remove hyphens
        pattern = r"[{}]".format(remove) # create the pattern
        text = re.sub(pattern,r'', text)
        ngrams = zip(*[text[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]


    def cosine_sim(text1, text2):
        cos = dict()
        count_vect = CountVectorizer(analyzer=ngrams)
        tfidf_transformer = TfidfTransformer()

        for i in range(len(text2)):
            targetText = text2[i].lower()
            sourceText = text1.lower()
            vec1 = [sourceText, targetText]
            vec2 = count_vect.fit_transform(vec1)
            tfidf = tfidf_transformer.fit_transform(vec2)
            similarity = ((tfidf * tfidf.T).A)[0,1]
            cos.update({text2[i]:similarity})
        return round(max(cos.values())*100)

    importData = request.get_json()
    trainData = pd.DataFrame(importData['mappings'])
    targetData = importData['target']['formatFields']
    
    trainData['confidence']= trainData['sourceField'].apply(lambda x: cosine_sim(x, trainData['targetField']))
    
    le = LabelEncoder()
    encode = le.fit_transform(trainData['sourceField'])
    
    count_vect = CountVectorizer(analyzer=ngrams)
    X_train_counts = count_vect.fit_transform(trainData['targetField'])
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    # KNN Classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    clf = knn.fit(X_train_tfidf, encode)

    pickle.dump(clf, open('/home/binarymonk/Desktop/ml/model/text_learning.pickle', 'wb'))
    
    return json.dumps({"sourceformatName": importData['source'].get('formatName'),
               "targetformatName": importData['target'].get('formatName'),
               "overallConfidence": np.mean(trainData['confidence']),"Message" : "Learned the mappings"})

    
if __name__ =='__main__':
    app.run(debug=True, port=4001)

#"mappings": trainData.to_dict(orient='records')