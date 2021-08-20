from fastai.vision import *
from fastai.metrics import error_rate
import time
import logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run

logging.basicConfig(level=logging.INFO)

def main():

    
    mlfclient = mlflow.tracking.MlflowClient(tracking_uri="http://mlops_mlflow-server_1:5000")
    mlfexp = mlfclient.create_experiment("training-pets1")
    run34 = mlfclient.create_run(experiment_id=mlfexp)

    params = {
    'size': 224,
    'epochs': 4,
    'bs': 64
    }

    for k,v in params.items():
        mlfclient.log_param(run_id=run34.info.run_uuid, key=k, value=v)


    class MLFlowTracking(LearnerCallback):
        "A `LearnerCallback` that tracks the loss and other metrics into MLFlow"
        def __init__(self, learn:Learner, client:MlflowClient, run_id: Run):
            super().__init__(learn)
            self.learn = learn
            self.client = client
            self.run_id = run_id
            self.metrics_names = ['train_loss', 'valid_loss'] + [o.__name__ for o in learn.metrics]

        def on_epoch_end(self, epoch, **kwargs:Any)->None:
            "Send loss and other metrics values to MLFlow after each epoch"
            if kwargs['smooth_loss'] is None or kwargs["last_metrics"] is None:
                return
            metrics = [kwargs['smooth_loss']] + kwargs["last_metrics"]
            for name, val in zip(self.metrics_names, metrics):
                self.client.log_metric(self.run_id, name, np.float(val))
    
    path_img = '/training/data'
    fnames = get_image_files(path_img)
    np.random.seed(2)
    pat = r'/([^/]+)_\d+.jpg$'

    mlflow.fastai.autolog()

    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=params['size'], bs=params['bs']).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate, callback_fns=[partial(MLFlowTracking, client=mlfclient, run_id=run34.info.run_uuid)])

    logging.info("loading latest parameters")

    learn.load('/training//models/best_resnet34',strict=False,remove_module=True)

    logging.info("start fit")

    learn.fit_one_cycle(params['epochs'])

    logging.info("end fit")

    learn.save('/models/stage-1')

    mlfclient.set_terminated(run_id=run34.info.run_uuid)

main()