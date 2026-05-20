import os
import mysql.connector
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

def get_conn():
    return mysql.connector.connect(
        host    =os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user    =os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )