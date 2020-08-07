import pandas as pd
import numpy as np
import json
from pickle5 import pickle
from flask import request, Flask
from simtext import CosineSimilarity, FitModel

app = Flask(__name__)


@app.route('/train/format/learn', methods=['POST'])
def unsupervisedLearning():
    """
    This function is mapped to the POST request of the REST interface
    """
    importData = request.get_json()
    trainData = pd.DataFrame(importData['mappings'])

    trainData['confidence'] = [CosineSimilarity(x, y) for x, y in
                               zip(trainData['sourceField'], trainData['targetField'])]

    clf = FitModel(trainData['sourceField'], trainData['targetField'])
    pickle.dump(clf, open('text_learning.pickle', 'wb'))

    return json.dumps({"sourceformatName": importData['source'].get('formatName'),
                       "targetformatName": importData['target'].get('formatName'),
                       "overallConfidence": np.mean(trainData['confidence']),
                       "Message": "Learned the mappings"})


if __name__ == '__main__':
    app.run(debug=True, port=4000)
