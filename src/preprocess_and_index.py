import faiss
import numpy as np
import openai
from pymongo import MongoClient

# Set your OpenAI API key
# openai.api_key = 'your-openai-api-key'
openai.api_key = 'sk-proj-hoTILlyXBr0xZrycnwCgT3BlbkFJD58jtXZgkNTayPyJqJca'

# Load candidate data from MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['job_matching']
collection = db['candidates']
candidates = list(collection.find({}, {'_id': 0}))

# Preprocess and create embeddings for candidates
def preprocess_candidate(candidate):
    return (
        f"Name: {candidate['Name']}\n"
        f"Contact Details: {candidate['Contact Details']}\n"
        f"Location: {candidate['Location']}\n"
        f"Job Skills: {candidate['Job Skills']}\n"
        f"Experience: {candidate['Experience']}\n"
        f"Projects: {candidate['Projects']}\n"
        f"Comments: {candidate['Comments']}\n"
    )

def create_embedding(text):
    response = openai.Embedding.create(input=[text], model='text-embedding-ada-002')
    return np.array(response['data'][0]['embedding'])

embeddings = [create_embedding(preprocess_candidate(c)) for c in candidates]
embedding_matrix = np.vstack(embeddings)

# Indexing with FAISS
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Save candidate data and FAISS index
faiss.write_index(index, '/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/candidates.index')
with open('/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/candidates_data.npy', 'wb') as f:
    np.save(f, np.array(candidates))

print("Data preprocessed and indexed successfully.")
