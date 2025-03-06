import json
import os
import hashlib

USERS_FILE = "users.json"

def load_users():
    """Load users from the users.json file or create it if it doesn't exist."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": []}, f)

    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"users": []}  # Return empty user list if JSON is corrupted

def save_users(users):
    """Save the updated users list to users.json."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Return a secure hash of the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def login_user(username, password):
    """Check if the username and password match in users.json."""
    users = load_users()["users"]
    password_hash = hash_password(password)  # Hash the entered password
    
    for user in users:
        if user["username"] == username and user["password"] == password_hash:
            return True  # ✅ Login successful
    return False  # ❌ Invalid credentials

def signup_user(username, password):
    """Register a new user if the username is not taken."""
    users_data = load_users()
    users = users_data["users"]

    # Check if username already exists
    if any(user["username"] == username for user in users):
        return False  # ❌ Username already exists

    # Add new user with hashed password and save
    users.append({"username": username, "password": hash_password(password)})
    save_users(users_data)
    
    return True  # ✅ Signup successful