import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/query"
EVAL_LLM_MODEL = "mistral"  # The local LLM RAGAs will use to "grade" the answers

# --- 1. Define Your "Golden Set" of Questions ---
# These are the questions we will test the API with.
golden_set = [
    {
        "question": "What is a neural network?",
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
    },
    {
        "question": "Explain backpropagation in simple terms.",
    },
    {
        "question": "What is a Convolutional Neural Network (CNN) used for?",
    },
    {
        "question": "What is transfer learning?",
    }
]
print(f"--- Loaded {len(golden_set)} test questions ---")


# --- 2. Function to Run Evaluation ---
def run_evaluation():
    """
    Runs the full evaluation against the running RAG API.
    """

    # --- 2a. Run all questions against our API to get answers and contexts ---
    questions = []
    answers = []
    contexts_list = []

    print(f"--- Querying API at {API_URL} for all test questions... ---")
    for item in golden_set:
        query = item["question"]
        try:
            # Call our FastAPI backend
            response = requests.post(API_URL, json={"query": query}, timeout=120)

            if response.status_code == 200:
                data = response.json()
                questions.append(query)
                answers.append(data.get("answer"))

                # RAGAs expects contexts as a list of strings
                contexts = [src.get("content_snippet", "") for src in data.get("sources", [])]
                contexts_list.append(contexts)
            else:
                print(f"Warning: API returned error {response.status_code} for question: {query}")
        except Exception as e:
            print(f"Error querying API for question '{query}': {e}")

    if not answers:
        print("--- No answers were generated. Cannot run evaluation. ---")
        print("--- Is your API server running? ---")
        return

    # --- 2b. Format the results into a dataset for RAGAs ---
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    dataset = Dataset.from_dict(dataset_dict)

    # --- 2c. Configure RAGAs to use our local Mistral for evaluation ---
    # We are using an LLM to "judge" the answers from our *other* LLM.
    print(f"--- Initializing RAGAs with local '{EVAL_LLM_MODEL}' model ---")
    eval_llm = OllamaLLM(model=EVAL_LLM_MODEL)
    eval_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # We will score on two key metrics:
    # 1. Faithfulness: Does the answer stick to the provided context? (Prevents hallucination)
    # 2. Answer Relevance: Is the answer relevant to the question?
    metrics = [
        faithfulness,
        answer_relevancy,
    ]

    # --- 2d. Run the evaluation ---
    print("--- Running RAGAs evaluation... This may take a few minutes... ---")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False  # Don't stop the whole run if one row fails
    )

    print("--- Evaluation Complete ---")
    return result


# --- 3. Main execution ---
if __name__ == "__main__":
    # 1. Make sure your API server is running!
    #    (Use your "Run API Server" configuration in PyCharm)

    # 2. Run this script
    evaluation_results = run_evaluation()

    if evaluation_results:
        print("\n--- Evaluation Results ---")
        df = evaluation_results.to_pandas()
        print(df)

        print("\n--- Average Scores ---")
        print(f"Faithfulness:     {df['faithfulness'].mean():.2f}")
        print(f"Answer Relevancy: {df['answer_relevancy'].mean():.2f}")