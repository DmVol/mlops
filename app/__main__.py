from pathlib import Path
from flask import Flask, jsonify, abort
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
    learner = load_learner(path, 'best_resnet34.pkl')

    src_folder = './ml_image_processing/'
    file_names = os.listdir(src_folder)

    predictions = {}

    for file_name in file_names:
        img = open_image(os.path.join(src_folder, file_name))
        pred_class,pred_idx,outputs = learner.predict(img)
        proc = str(outputs[pred_idx])[9:11]
        predictions[file_name] = [pred_class, proc]
        if int(proc) >= 90:
            shutil.move(os.path.join(src_folder, file_name), f'./accurate_predictions/{pred_class}_{proc}.jpg')
        else:
            shutil.move(os.path.join(src_folder, file_name), f'./inaccurate_predictions/{pred_class}_{proc}.jpg')

    return {'success': "prediction finished"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)