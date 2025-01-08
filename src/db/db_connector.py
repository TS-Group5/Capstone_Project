from sqlite_connector import SQLiteConnector
import sqlite3
def connect_db():
    return  sqlite3.connect("db/resume_db.db")
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
def insert_lipsync_data( userid, scene, type_, url):
    """
    Inserts a new record into the lipsync table or updates the URL if a record already exists.

    Args:
        db_path (str): Path to the SQLite database file.
        userid (str): User ID.
        scene (str): Scene identifier.
        type_ (str): Type of data (e.g., video, audio).
        url (str): URL to be stored or updated.
    """
    try:
        # Connect to the SQLite database
        conn =  connect_db()
        cursor = conn.cursor()


        # Insert or update the record
        cursor.execute("""
        INSERT OR REPLACE INTO lipsync (scene, type, url, userid)
        VALUES (
            (SELECT id FROM lipsync WHERE userid = ? AND scene = ? AND type = ?),
            ?, ?, ?, ?
        )
        """, (userid, scene, type_, scene, type_, url, userid))

        # Commit the transaction
        conn.commit()
        print("Record inserted or updated successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        if conn:
            conn.close()



def get_url_by_id( record_id):
    """
    Retrieves the URL from the lipsync table based on the provided ID.

    Args:
        db_path (str): Path to the SQLite database file.
        record_id (int): ID of the record to retrieve.

    Returns:
        str: The URL of the record if found, otherwise None.
    """
    try:
        # Connect to the database
        conn =  connect_db()
        cursor = conn.cursor()

        # Query to retrieve the URL
        cursor.execute("SELECT url FROM lipsync WHERE id = ?", (record_id,))
        result = cursor.fetchone()

        # Check if a result was found
        if result:
            return result[0]  # URL is in the first column of the result
        else:
            print(f"No record found with ID {record_id}.")
            return None
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # Close the connection
        if conn:
            conn.close()
          # ID of the record you want to retrieve


# Example usage
#insert_lipsync_data(db_path, "user123", "scene1", "video", "https://example.com/new-url")