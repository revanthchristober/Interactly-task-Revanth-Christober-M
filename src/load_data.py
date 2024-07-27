import pandas as pd
from pymongo import MongoClient

# Load the candidate data
df = pd.read_excel('/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/data/RecruterPilot candidate sample input dataset.xlsx')

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['job_matching']
collection = db['candidates']

# Insert data into MongoDB
data = df.to_dict(orient='records')
collection.insert_many(data)

print("Data loaded successfully into MongoDB.")
