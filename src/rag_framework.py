import faiss
import numpy as np
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set your OpenAI API key
# openai.api_key = 'your-openai-api-key'
openai.api_key = 'sk-proj-hoTILlyXBr0xZrycnwCgT3BlbkFJD58jtXZgkNTayPyJqJca'

# Load the FAISS index and candidate data
index = faiss.read_index('/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/candidates.index')
with open('/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/candidates_data.npy', 'rb') as f:
    candidates = np.load(f, allow_pickle=True).tolist()

# Load fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/fine_tuned_model")
model = GPT2LMHeadModel.from_pretrained("/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/fine_tuned_model")

# Function to match job description to candidates
def create_embedding(text):
    response = openai.Embedding.create(input=[text], model='text-embedding-ada-002')
    return np.array(response['data'][0]['embedding'])

def match_candidates(job_description):
    job_embedding = create_embedding(job_description)
    D, I = index.search(np.array([job_embedding]), 10)
    matched_candidates = [candidates[i] for i in I[0]]
    return matched_candidates

# Function to generate response using RAG
def generate_response(matched_candidates, job_description):
    input_text = f"Job Description: {job_description}\nMatched Candidates:\n"
    for candidate in matched_candidates:
        # f"Name: {candidate['Name']}\n"
        # f"Contact Details: {candidate['Contact Details']}\n"
        # f"Location: {candidate['Location']}\n"
        # f"Job Skills: {candidate['Job Skills']}\n"
        # f"Experience: {candidate['Experience']}\n"
        # f"Projects: {candidate['Projects']}\n"
        # f"Comments: {candidate['Comments']}\n"
        input_text += f"Name: {candidate['Name']}, Skills: {candidate['Job Skills']}, Experience: {candidate['Experience']}\n"

    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
