import json
import urllib.request
from pathlib import Path
from flask import Flask, jsonify
from flask import request
from fastai.vision import *
app = Flask(__name__)

@app.route('/ping')
def ping():
    return {'success': 'pong'}, 200

@app.route('/predict')
def predict():
    path = Path('./models')
    learner = load_learner(path, 'export.pkl')
    img = open_image('./tmp/image.jpg')
    pred_class,pred_idx,outputs = learner.predict(img)
    proc = str(outputs[pred_idx])[9:11]+'%'
    result = f"breed: {pred_class}, accuracy: {proc}"
    return {'success': result}, 200

@app.route('/classification')
def classification():

    path = Path('./models')
    learner = load_learner(path, 'export.pkl')
    
    image = request.args['image']
    urllib.request.urlretrieve(image, './tmp/image.jpg')
    img = open_image('./tmp/image.jpg')
    pred_class,pred_idx,outputs = learner.predict(img)

    result = json.dumps({
        "predictions": sorted(
            zip(learner.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })
    print(result)

    return {'success': 'pong'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')