import subprocess

# (In terminal) set config first
# $ export WORKING_DIRECTORY=$(pwd)
# $ export AIRFLOW_HOME=~/airflow
# $ python file_to_db.py --path ${WORKING_DIRECTORY}/event_data.csv

# extract 10000 new data for train and saved in .csv format
subprocess.call("python ${WORKING_DIRECTORY}/db_to_file.py",shell=True)

# train with new data
subprocess.call("python /opt/ml/code/train.py --model_dir /opt/ml/code/models --asset_dir /opt/ml/code/asset --data_dir ${WORKING_DIRECTORY} --file_name data.csv",shell=True)

# exec server
subprocess.call("python ${WORKING_DIRECTORY}/server/server.py --port 6006",shell=True)

# reset db
subprocess.call("airflow db reset -y",shell=True)

# exec airflow scheduler
subprocess.call("airflow scheduler",shell=True)

# (In new terminal) exec web server at 2431 base port
# root path : /opt/ml/code/
# $ python flask_server.py

