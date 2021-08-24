from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

import datetime
from datetime import date, timedelta
import airflow
import os
import shutil

default_args = {
    "depends_on_past" : False,
    "start_date"      : airflow.utils.dates.days_ago( 1 ),
    "retries"         : 1,
    "retry_delay"     : datetime.timedelta( hours= 5 ),
}
today = datetime.datetime.today() 
yesterday = date.today() - timedelta(days=1)
 
def _sequential_file_mover(src_folder , dst_folder):
    
    file_names = os.listdir(src_folder)
    
    dst_files = os.listdir(dst_folder)
    
    max_values = {}

    for file in dst_files:
        if file.split("_")[0] not in max_values.keys():
            max_values[file.split("_")[0]] = int(file.split("_")[1].split(".")[0])
        else:
            if int(file.split("_")[1].split(".")[0]) > max_values[file.split("_")[0]]:
                max_values[file.split("_")[0]] = int(file.split("_")[1].split(".")[0])

    for file_name in file_names:
        file_label = file_name.split("_")[0]
        max_values[file_label] += 1
        shutil.move(os.path.join(src_folder, file_name), os.path.join(dst_folder, f"{file_label}_{max_values[file_label]}.jpg"))

with airflow.DAG("mlops_training_pipeline", default_args= default_args, schedule_interval= "@once") as dag:
    start_task  = DummyOperator(task_id="start")
    stop_task   = DummyOperator(task_id="stop")
    sensor_task = FileSensor(task_id="file_sensor_task", poke_interval=30,  filepath="/bitnami/labeled_data/existing_entries/")
    move_labeled_files = PythonOperator(task_id='move_new_images', 
                                        python_callable=_sequential_file_mover, 
                                        op_args=['/bitnami/labeled_data/existing_entries/', '/bitnami/data/']
                                        )
    send_curl = BashOperator(task_id='send_curl', bash_command="curl -XGET 'ml-flask:5500/train'")

start_task >> sensor_task >> move_labeled_files >> send_curl  >> stop_task