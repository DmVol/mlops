from fastai.vision import *
from fastai.metrics import error_rate
import time
import logging
import mlflow.fastai

logging.basicConfig(level=logging.INFO)

def main():

    bs = 64
    path_img = '/training/data'
    fnames = get_image_files(path_img)
    np.random.seed(2)
    pat = r'/([^/]+)_\d+.jpg$'

    mlflow.fastai.autolog()

    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                    ).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    logging.info("loading latest parameters")

    learn.load('/training//models/best_resnet34',strict=False,remove_module=True)

    logging.info("start fit")

    with mlflow.start_run() as run:
        learn.fit_one_cycle(4)
        mlflow.fastai.log_model(learn, "model")

    logging.info("end fit")

    learn.save('/models/stage-1')

main()