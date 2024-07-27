# Interactly Profile Matching Task - Data Science Intern

## Overview

This project implements a profile matching system using a Retrieval-Augmented Generation (RAG) framework. The system takes a job description as input and retrieves the top matching candidate profiles from a database.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/revanthchristober/Interactly-task-Revanth-Christober-M.git
cd Interactly-task-Revanth-Christober-M
```

### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages using `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Prepare the Data

Place your candidate data file in the `data/` directory. Ensure the data is in the format required by the preprocessing scripts.

### 5. Preprocess and Index the Data

Run the preprocessing and indexing scripts to prepare the data.

```bash
python src/preprocess_and_index.py
```

### 6. Fine-Tune the LLM

Fine-tune the LLM on a representative sample of the candidate data.

```bash
python src/fine_tune_llm.py
```

### 7. Run the Application

You can choose between running the application with a command-line interface or a web interface. 

#### CLI Interface

```bash
python main.py
```

#### Web Interface

1. Ensure Flask is installed (`pip install flask`).
2. Run the Flask application.

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your web browser to access the chat interface.

### 8. Testing

Test the system using different job descriptions to ensure it retrieves relevant profiles and generates appropriate responses. Example Prompt to test it out:

“Pick up the top 10 profiles for the following job description, We are looking for a skilled UI Developer to join our dynamic team. The ideal candidate will have a strong background in front-end development, with proficiency in HTML, CSS, JavaScript, and modern frameworks like React or Angular. Your primary responsibility will be to create visually appealing and user-friendly web interfaces that enhance user experience and align with our brand guidelines.”
