import os
import psycopg2
from dotenv import load_dotenv

load_dotenv('.env')

def connect_to_database():
    try:
        connection = psycopg2.connect(host=os.environ.get('HOST'), database=os.environ.get('NAME'), 
                                    user=os.environ.get('USER'), password=os.environ.get('PASSWORD'),
                                    port=os.environ.get('PORT'))
        print('Connected')
    except:
        print('Unable to connect to the database')

