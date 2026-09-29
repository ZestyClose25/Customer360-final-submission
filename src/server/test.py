import sqlite3
from pathlib import Path

database_path = Path(__file__).resolve().parent / "server.db"
connection = sqlite3.connect(database_path)
connection.row_factory = sqlite3.Row

rows = connection.execute("""
    SELECT *
    FROM checkpoints
    ORDER BY sequence_id DESC
""").fetchall()

print(f"Database: {database_path}")
print(len(rows))
connection.close()