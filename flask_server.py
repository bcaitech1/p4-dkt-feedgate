import os
import torch
import sys
import os.path as p
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
import datetime
import torch.nn as nn
from flask import Flask, render_template
from flask import request
import flask_inference
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config_dir import config
from args import parse_args
from train import YamlConfigManager
import pandas as pd 
import sqlite3

app = Flask(__name__)

def insert_data(data):
    con = sqlite3.connect('/opt/ml/code/airflow_serving/data.db')
    cur = con.cursor()
    for row in data :
        #row_index, userID, assessmentItemId, testId, answerCode, Timestamp, KnowledgeTag
        qry_values = "NULL,'{}','{}','{}', '{}', '{}', '{}'"\
            .format(row[0], row[1],row[2], row[3], row[4], row[5])
    
        query = '''INSERT INTO problems VALUES ({})'''.format(qry_values)
        cur.execute(query)
    con.commit()
    con.close()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/get_num_of_events', methods=['GET'])
def get_num_of_events():
    con = sqlite3.connect('/opt/ml/code/airflow_serving/data.db')
    cur = con.cursor()
    qry = 'SELECT row_index FROM problems ORDER BY row_index DESC LIMIT 1;'
    cur.execute(qry)
    rows = cur.fetchall()
    num_of_event = str(rows[0][0])
    return num_of_event

@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_data = []
    # make current time-stamp for str type
    t = str(datetime.datetime.now())[:-7] 
    
    df = pd.read_csv("questions.csv")
    
    # create dummy user_id in asc
    lst_row = df.tail(n=1)
    lst_user_id = lst_row['userID'].values[0]
    new_user_id = lst_user_id + 1

    for d in data:
        if 'answer' in d:
            row = [new_user_id,d['assess_id'],d['test_id'],d['answer'],t,d['tag']]
            user_data.append(row)

    print(user_data)
    insert_data(user_data) # insert to db

    score = flask_inference.inference(user_data)
    score = int(score*100)
    return str(score)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2431, threaded=False, debug=True)




