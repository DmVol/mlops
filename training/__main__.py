from fastai.vision import *
from fastai.metrics import error_rate
import time
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run

logging.basicConfig(level=logging.INFO)

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

def main():

    remote_server_uri = "http://mlops_mlflow-server_1:5000"
    mlflow.set_tracking_uri(remote_server_uri)

    params = {
    'size': 224,
    'epochs': 4,
    'bs': 8
    }

    mlflow.set_experiment("/my-experiment")

    path_img = '/training/catsdogsrabbits'
    fnames = get_image_files(path_img)
    np.random.seed(2)
    pat = r'/([^/]+)_\d+.jpg$'

    mlflow.fastai.autolog()

    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=params['size'], bs=params['bs']).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    logging.info("loading latest parameters")

    learn.load('/training/models/catsdogsrabbits_resnet34',strict=False,remove_module=True)

    logging.info("start fit")

    with mlflow.start_run() as run:
        learn.fit_one_cycle(params['epochs'])

    logging.info("end fit")

    learn.save('stage-1')

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

main()