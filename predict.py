from pickle5 import pickle
from flask import request, Flask, jsonify
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    predictData = request.get_json()
    predictdata = predictData['source']['formatField']
    modelFile = pickle.load(open('text_matching.pickle', 'rb'))
    model = modelFile['model']
    labelEncoder = modelFile['labelEncoder']
    count_vec_fit = modelFile['count_vec_fit']
    tfidf_fit = modelFile['tfidf_fit']

    count_vec_transform = count_vec_fit.transform(predictdata)
    tfidf_transform = tfidf_fit.transform(count_vec_transform)

    prediction = model.predict(tfidf_transform)

    return jsonify({"prediction": labelEncoder.inverse_transform(prediction).tolist()})


if __name__ == '__main__':
    app.run(debug=True, port=4002)
