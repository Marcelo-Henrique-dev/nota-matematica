import mysql.connector
import pandas as pd
import os

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '125896'
}
db_name = 'students_db'

def create_database(db_config, db_name):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.close()

def execute_sql_file(db_config, db_name, sql_file):
    conn = mysql.connector.connect(database=db_name, **db_config)
    cursor = conn.cursor()
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    for statement in sql_script.split(';'):
        stmt = statement.strip()
        if stmt:
            cursor.execute(stmt)
    conn.commit()
    conn.close()

def insert_csv_to_table(db_config, db_name, csv_file, table_name):
    df = pd.read_csv(csv_file)
    conn = mysql.connector.connect(database=db_name, **db_config)
    cursor = conn.cursor()
    cols = ",".join(df.columns)
    for _, row in df.iterrows():
        vals = ",".join([f"'{str(x).replace('\'','\\\'')}'" if pd.notnull(x) else "NULL" for x in row])
        cursor.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({vals})")
    conn.commit()
    conn.close()

def drop_database(db_config, db_name):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.close()

if __name__ == "__main__":
    create_database(db_config, db_name)
    execute_sql_file(db_config, db_name, "core/data/sor_students.sql")
    execute_sql_file(db_config, db_name, "core/data/sot_students.sql")
    insert_csv_to_table(db_config, db_name, "students.csv", "sor_students")
    # Exemplo: para sot_students, gere o CSV processado e insira
    # insert_csv_to_table(db_config, db_name, "students_processed.csv", "sot_students")
    drop_database(db_config, db_name)