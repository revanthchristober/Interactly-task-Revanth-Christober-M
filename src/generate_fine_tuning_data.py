import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['job_matching']
collection = db['candidates']

# Load candidate data
candidates = list(collection.find({}, {'_id': 0}))

# Create a fine-tuning dataset
with open('/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/data/fine_tuning_data.txt', 'w') as f:
    for candidate in candidates:
        candidate_text = (
            f"Name: {candidate['Name']}\n"
            f"Contact Details: {candidate['Contact Details']}\n"
            f"Location: {candidate['Location']}\n"
            f"Job Skills: {candidate['Job Skills']}\n"
            f"Experience: {candidate['Experience']}\n"
            f"Projects: {candidate['Projects']}\n"
            f"Comments: {candidate['Comments']}\n"
        )
        f.write(candidate_text + "\n---\n")

print("Fine-tuning data generated successfully!")
