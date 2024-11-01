import os
import mysql.connector
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm.auto import tqdm
import time
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()

# pinecone config
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(
    cloud="aws",
    region="us-east-1"
)

index_name = 'gis-index'
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# Check if index already exists (it shouldn't if this is the first time)
if index_name not in existing_indexes:
    # If it does not exist, create the index
    pc.create_index(
        name=index_name,
        dimension=768,  # dimensionality of the embedding model
        metric='dotproduct',
        spec=spec
    )
    # Wait for the index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)
time.sleep(1)

# MySQL configuration
db_connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password=os.getenv('DB_PASSWORD'),
    database='gis'
)
cursor = db_connection.cursor()


# Google Generative AI API configuration
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to fetch senior high data
def fetch_senior_high_data():
    query = "SELECT * FROM senior_high_student_data"
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    return data

# Function to fetch college data
def fetch_college_data():
    query = "SELECT * FROM college_student_data"
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    return data

# Function to fetch event report data
def fetch_event_data():
    query = "SELECT * FROM event_reports"
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(cursor.fetchall(), columns=columns)
    return data

# Function to sync with Pinecone
def sync_with_pinecone(data, text_key):
    batch_size = 100
    total_batches = (len(data) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(data), batch_size), desc='Processing Batches', unit='batch', total=total_batches):
        i_end = min(len(data), i + batch_size)
        batch = data.iloc[i:i_end]

        # Generate unique IDs
        ids = [str(row['stud_id'] if 'stud_id' in row else row['id']) for _, row in batch.iterrows()]

        # Combine text fields for embedding
        texts = [
            f"{row['year']} {row['strand'] if 'strand' in row else row['course']} {row['previous_school']} "
            f"{row['city']} {row['province']} {row['barangay']} {row['full_address']}"
            if 'stud_id' in row else f"{row['type']} affecting {row['number_of_students_affected']} students in an area of {row['total_area']} km²"
            for _, row in batch.iterrows()
        ]

        # Embed text
        embeds = embed_model.embed_documents(texts)

        # Get metadata to store in Pinecone
        metadata = [
            row.to_dict() for _, row in batch.iterrows()
        ]

        # Upserting Vectors
        with tqdm(total=len(ids), desc='Upserting Vectors', unit='vector') as upsert_pbar:
            index.upsert(vectors=zip(ids, embeds, metadata))
            upsert_pbar.update(len(ids))  # Update the upsert progress bar

def main():
    # Sync each table's data with Pinecone
    senior_high_data = fetch_senior_high_data()
    sync_with_pinecone(senior_high_data, 'Senior High School Data')

    college_data = fetch_college_data()
    sync_with_pinecone(college_data, 'College Student Data')

    event_data = fetch_event_data()
    sync_with_pinecone(event_data, 'Event Report Data')

if __name__ == "__main__":
    main()

# Close the cursor and connection
cursor.close()
db_connection.close()