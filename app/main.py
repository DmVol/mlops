from pathlib import Path
from flask import Flask, jsonify
from flask import request
from fastai.vision import *
import os
import shutil

app = Flask(__name__)

@app.route('/ping')
def ping():
    return {'success': 'pong'}, 200

@app.route('/predict')
def predict():
    path = Path('./models')
    learner = load_learner(path, 'export.pkl')

    src_folder = './ml_image_processing/'
    file_names = os.listdir(src_folder)

    for file_name in file_names:
        img = open_image(os.path.join(src_folder, file_name))
        pred_class,pred_idx,outputs = learner.predict(img)
        proc = str(outputs[pred_idx])[9:11]
        if int(proc) >= 90:
            shutil.move(os.path.join(src_folder, file_name), f'./accurate_predictions/{pred_class}_{proc}.jpg')
        else:
            shutil.move(os.path.join(src_folder, file_name), f'./inaccurate_predictions/{pred_class}_{proc}.jpg')

    result = f"breed: {pred_class}, accuracy: {proc}"
    return {'success': result}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')