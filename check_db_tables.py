import sqlite3
con = sqlite3.connect('database.db')
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
print('tables:', cur.fetchall())
con.close()
