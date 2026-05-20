import mysql.connector

def get_conn():
    return mysql.connector.connect(
        host="193.203.175.198",
        database="u700242432_appprodutos",
        user="u700242432_appprodutos",
        password="OxEstrutur@25"
    )