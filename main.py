from src.match_candidates import match_candidates, generate_response

def main():
    print("Welcome to the RAG System Chat Interface")

    while True:
        job_description = input("\nEnter job description (or type 'exit' to quit): ")
        if job_description.lower() == 'exit':
            break

        try:
            # Match candidates based on job description
            matched_candidates = match_candidates(job_description)

            # Generate a response using RAG
            response = generate_response(matched_candidates, job_description)

            # Print the response
            print("\nResponse from RAG system:\n")
            print(response)

        except Exception as e:
            print("An error occurred:", e)

if __name__ == '__main__':
    main()

