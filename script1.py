import sqlite3
import hashlib

# Function to create the users table if it doesn't exist
def create_table():
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
        """)
        conn.commit()
        print("Table created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

# Function to hash the password before storing it
def hash_password(password: str) -> str:
    # Using hashlib to hash the password securely (SHA256 for simplicity)
    hashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return hashed

# Function to add a user to the database
def add_user(username: str, password: str):
    try:
        # Hash the password before storing it
        hashed_password = hash_password(password)

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        # Insert user data into the database
        cursor.execute("""
        INSERT INTO users (username, password) VALUES (?, ?)
        """, (username, hashed_password))

        conn.commit()
        print("User added successfully.")
    except sqlite3.IntegrityError:
        print("Error: Username already exists.")
    except sqlite3.Error as e:
        print(f"Error adding user: {e}")
    finally:
        conn.close()

# Call create_table to ensure the table exists
create_table()

# Example to add a new user
username = input("Enter username: ")
password = input("Enter password: ")
add_user(username, password)
