from fastai.vision import *
from fastai.metrics import error_rate
import time

bs = 16
path_img = '/training/test'
fnames = get_image_files(path_img)
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
print("start fit")
learn.fit_one_cycle(4)
print("end fit")
learn.save('/training/stage-1')

time.sleep(2000)