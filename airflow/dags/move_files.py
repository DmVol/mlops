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
 
def _file_mover(src_folder , dst_folder):
    file_names = os.listdir(src_folder)

    for file_name in file_names:
        shutil.move(os.path.join(src_folder, file_name), dst_folder)

with airflow.DAG( "mlops_file_pipeline", default_args= default_args, schedule_interval= "@once"  ) as dag:
    start_task  = DummyOperator(task_id="start")
    stop_task   = DummyOperator(task_id="stop")
    sensor_task = FileSensor(task_id="file_sensor_task", poke_interval=30,  filepath="/bitnami/new_images/"])
    put_file    = PythonOperator(task_id='move_new_images', 
                                 python_callable=_file_mover, 
                                 op_args=['/bitnami/new_images/', '/bitnami/ml_image_processing/']
                                )
    send_curl = BashOperator(task_id='send_curl', bash_command="curl -XGET 'ml-flask:5500/predict'")

start_task >> sensor_task >> put_file >> send_curl  >> stop_task