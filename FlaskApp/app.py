from flask import Flask, render_template, request, jsonify
from run_model import get_top_similars
from architecture import BertForSTS
import torch
from collections import defaultdict
from transformers import BertTokenizer
from run_model import get_embeddings
from config import DATABASE_NAME, DATABASE_USER, DATABASE_PASSWORD, DATABASE_HOST, DATABASE_PORT
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String
import uuid
import psycopg2
import json

app = Flask(__name__)

# Connect to the database
conn = psycopg2.connect(database=DATABASE_NAME, user=DATABASE_USER,
                        password=DATABASE_PASSWORD, host=DATABASE_HOST, port=DATABASE_PORT)
cur = conn.cursor()

cur.execute(
    '''CREATE TABLE IF NOT EXISTS projects (project_id uuid PRIMARY KEY, eproject_status varchar(255), description text, image_url varchar(255), report_link varchar(255), student_limit int, title varchar(255), youtube_link varchar(255), group_id uuid, project_type_id uuid, embedding varchar(40000) );''')



# commit the changes
conn.commit()

# close the cursor and connection
cur.close()
conn.close()



@app.route("/ask", methods=['POST'])
def ask():
    try:
        data = json.loads(request.data)
        query_id = data['id']
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Invalid JSON format or missing "id" field'}), 400

    #get all projects from database
    conn = psycopg2.connect(database=DATABASE_NAME, user=DATABASE_USER,
                            password=DATABASE_PASSWORD, host=DATABASE_HOST, port=DATABASE_PORT)
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects")
    projects = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    #get uuid and embeddings of projects
    projects = [(project[0],list(map(float, project[10].split()))) for project in projects]
    target_embedding = [text for id, text in projects if id == query_id]
    if len(target_embedding) == 0:
        return jsonify({'status': 'error', 'message': 'Project not found'}), 404
    target_embedding = target_embedding[0]
    target_embedding = [float(i) for i in target_embedding]
    other_embeddings = [(id, embeddings) for id, embeddings in projects if id != query_id]


    bot_response = get_top_similars(target_embedding,other_embeddings)

    return jsonify({'status': 'OK', 'id': query_id, 'similar_ids': bot_response})

@app.route("/addNewProject", methods=['POST'])
def add_new_project():
    try:
        data = json.loads(request.data)
        project_abstract = data['abstract']
        project_keywords = data['keywords']
    except:
        return jsonify({'status': 'error', 'message': 'Invalid JSON format or missing "abstract" or "keywords" field'}), 400
    text = project_abstract + " ".join(project_keywords)
    embeddings = get_embeddings(text)

    return jsonify({'status': 'OK', 'embeddings': embeddings})



if __name__ == "__main__":

    app.run(port=5000)



