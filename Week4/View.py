import sqlite3
import pandas as pd

DB_FILE = "safety_data.db"

def main():
    try:
        # Connect to database
        conn = sqlite3.connect(DB_FILE)

        # Read full table into pandas
        df = pd.read_sql_query("SELECT * FROM safety_logs", conn)

        conn.close()

        if df.empty:
            print("\nDatabase is empty.\n")
            return

        print("\n====== SAFETY LOGS TABLE ======\n")
        print(df.to_string(index=False))
        print("\nTotal Rows:", len(df))

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
