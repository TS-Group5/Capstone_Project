from sqlite_connector import SQLiteConnector
import sqlite3
def connect_db():
    return  sqlite3.connect("db/quiz_db.db")
def check_login(email, password):
    print("email",email)
    print("password",password)
    """
    Validate student login using registration number and password.
    """
    try:
        # Connect to the database
        db = connect_db()
        db.row_factory = sqlite3.Row  # Allows accessing columns by name
        cursor = db.cursor()

        # Define the query to check the student's credentials
        query = """
        SELECT id, name, user_type
        FROM user
        WHERE email = ? AND password = ?
        """

        print("Calling DB to validate login...")
        cursor.execute(query, (email, password))

        # Fetch the student's record
        user = cursor.fetchone()

        # Close the database connection
        db.close()

        # Check if a student record was found
        if user is None:
            print("Login failed: Invalid registration number or password.")
            return None  # Return None if credentials are invalid
        
        print(f"Login successful: User {user['name']} found.")
        return dict(user)  # Return the student information as a dictionary if login is valid
    
    except Exception as e:
        print(f"Error occurred while checking login: {e}")
        return None  # Return None in case of any error
