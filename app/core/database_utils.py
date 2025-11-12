import mysql.connector
import pandas as pd

db_config = {
    'host': 'localhost',
    'user': 'seu_usuario',
    'password': 'sua_senha'
}
db_name = 'students_db'

def create_db(db_config, db_name):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.close()

def run_sql(db_config, db_name, sql_path):
    conn = mysql.connector.connect(database=db_name, **db_config)
    cursor = conn.cursor()
    with open(sql_path, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    for statement in sql_script.split(';'):
        stmt = statement.strip()
        if stmt:
            cursor.execute(stmt)
    conn.commit()
    conn.close()

def insert_csv(db_config, db_name, csv_path, table_name):
    df = pd.read_csv(csv_path)
    conn = mysql.connector.connect(database=db_name, **db_config)
    cursor = conn.cursor()
    cols = ",".join(df.columns)
    for _, row in df.iterrows():
        vals = ",".join([f"'{str(x).replace('\'','\\\'')}'" if pd.notnull(x) else "NULL" for x in row])
        cursor.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({vals})")
    conn.commit()
    conn.close()

def drop_db(db_config, db_name):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.close()