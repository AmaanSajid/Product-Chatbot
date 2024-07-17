import psycopg2

conn=psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="Amaansajid@1808",
        port=5432)

cur=conn.cursor()
cur.close()
conn.close()
