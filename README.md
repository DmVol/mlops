# mlops
Repository for ML Ops course homework

# Repository Structure
- airflow - folder with Airflow DAGs
- app - folder with fastai models and python flask app, used for inference
- lobeai - folder created by Lobe AI auto-ml tool
- mlflow - folder with MLFlow object, MLFlow is used for training monitoring process
- training - folder with flask app, which executes training. Training script allows to extend and retrain the model.
- homework1.ipynb, Dockerfile, config.yml, environment.yaml,run.sh - files required for fastai jupyter notebook startup (docker build -t mlops/fastai ., docker run -d -p 8888:8888 --ipc=host mlops/fastai)
- docker-compose.yaml - config file for cluster start-up.

# How To Init Cluster
docker compose up -d --build

# Cluster Structure
 Airflow - is used for files movement and flask servers triggering. Both inference and training scripts are wrapped by Flask servers.
  - redis
  - postgresql
  - scheduler
  - worker
  - web-ui
 
 MLFlow - web-server which is used for training process monitoring, it stores experiments and corresponding training parameters.
 
 Ml-Flask - web-server built on a top of trained .pkl model, used for images class prediction. It will receive images and move them to different folders based on predictions accuracy.
 
 Ml-Training - training script wrapped with web-server. The data from central storage will be used for model retraining, and the script will save the new model to the specified location. Ml-Flask will use the new .pkl file for inference. 
 The training script will also log the main parameters to the MLFlow.
 
