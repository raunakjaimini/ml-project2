import pandas as pd
import sqlite3

# Path to your CSV file
csv_file_path = 'reportexcel.csv'
df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

# Path to SQLite database file
db_file_path = 'analytics_db'

# Connect to SQLite database (creates the file if it doesn't exist)
conn = sqlite3.connect(db_file_path)

# Write the DataFrame to SQLite database
df.to_sql('analytics', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("CSV file has been converted to SQLite database.")