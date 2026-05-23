from app.core.database import engine

try:
    with engine.connect() as conn:
        print("Connection to the database was successful!")
except Exception as e:
    print(f"Error: {e}")