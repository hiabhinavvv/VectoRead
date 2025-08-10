import chromadb
import os
import time
from datetime import datetime, timedelta

DB_PATH = "./chroma_db"
EXPIRATION_HOURS = 24

def cleanup_old_collections():
    print(f"--- Starting cleanup process at {datetime.now()} ---")
    
    if not os.path.exists(DB_PATH):
        print("Database path not found. Nothing to clean up.")
        return

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()

        if not collections:
            print("No collections found. Cleanup complete.")
            return

        print(f"Found {len(collections)} collections. Checking for old sessions...")
        
        now = time.time()
        expiration_limit = now - (EXPIRATION_HOURS * 3600)
        deleted_count = 0

        for collection in collections:
            try:
                collection_db_file = os.path.join(DB_PATH, collection.id + ".sqlite3")

                if os.path.exists(collection_db_file):
                    file_mod_time = os.path.getmtime(collection_db_file)
                    
                    if file_mod_time < expiration_limit:
                        print(f"Deleting old collection: {collection.name} (Last modified: {datetime.fromtimestamp(file_mod_time)})")
                        client.delete_collection(name=collection.name)
                        deleted_count += 1
                else:
                    print(f"Warning: Could not find DB file for collection '{collection.name}'. Skipping.")

            except Exception as e:
                print(f"Error processing collection '{collection.name}': {e}")

        print(f"Cleanup complete. Deleted {deleted_count} old collections.")

    except Exception as e:
        print(f"An error occurred during the cleanup process: {e}")


if __name__ == "__main__":
    cleanup_old_collections()