import psycopg2
import pandas as pd
#from decouple import config
import tkinter as tk
from tkinter import filedialog


def connection_database():
    conn_db = psycopg2.connect(
        host='ec2-34-247-94-62.eu-west-1.compute.amazonaws.com',
        database='d4on6t2qk9dj5a',
        user='nxebpjsgxecqny',
        password='1da2f1f48e4a37bf64e3344fe7670a6547c169472263b62d042a01a8d08d2114'
    )
    return conn_db

def get_data_to_df(data_name, db, file_path):
    if data_name == 'Diabet inde':
        sql_query = "SELECT * FROM diabete_inde;"
        cursor = db.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)

    elif data_name == 'Vin':
        sql_query = "SELECT * FROM vin;"
        cursor = db.cursor()
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)   

    else:
        # Si l'utilisateur veut charger un DataFrame depuis son bureau
        root = tk.Tk()
        root.withdraw()  # Cacher la fenêtre principale de tkinter
        #file_path = filedialog.askopenfilename(title="Sélectionnez un fichier CSV")
        
        # Lire le fichier CSV en DataFrame
        df = pd.read_csv(file_path, sep=',')
    
    return df 

