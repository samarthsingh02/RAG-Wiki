import streamlit as st
import requests  # The library to make HTTP requests

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/query"  # The URL of your FastAPI backend

# --- Step 7: Re-built Frontend ---

st.set_page_config(page_title="Wikipedia RAG v2", layout="wide")
st.title("Wikipedia RAG Assistant (v2: API-Powered) 🚀")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Ask a Question")
    st.markdown(
        "This app is now powered by a separate FastAPI backend. "
        "Type your question, and it will be sent to the API, "
        "which will perform the RAG and return an answer."
    )

    # User input
    query = st.text_input("Your question:", key="query_input")

    if st.button("Get Answer", type="primary"):
        if query:
            with st.spinner("Sending query to API..."):
                try:
                    # --- THIS IS THE NEW LOGIC ---
                    # 1. Prepare the JSON payload to send to the API
                    payload = {"query": query}

                    # 2. Make the POST request to your FastAPI backend
                    response = requests.post(API_URL, json=payload, timeout=120)

                    # 3. Handle the response
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer")
                        sources = data.get("sources", [])

                        st.subheader("Answer from API:")
                        st.write(answer)

                        # (Bonus) Display the source URLs from the API
                        with st.expander("Show Sources (from API)"):
                            if sources:
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:** [{source.get('url')}]({source.get('url')})")
                                    st.markdown(
                                        f"> {source.get('content_snippet')}"
                                    )
                                    st.markdown("---")
                            else:
                                st.write("No sources returned from API.")

                    else:
                        st.error(f"Error from API: {response.status_code} - {response.text}")
                    # -----------------------------

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the API: {e}")
                    st.warning("Is the backend_api/main.py server running?")
        else:
            st.warning("Please enter a question.")

with col2:
    st.header("New Architecture")
    st.markdown(
        """
        1.  **Frontend (Streamlit):** You are interacting with this app. It does **not** have any LLMs or vector stores.
        2.  **API Call (JSON):** When you ask a question, this Streamlit app sends a `POST` request to `http://127.0.0.1:8000/query` with your question.
        3.  **Backend (FastAPI):** The server (running from `backend_api/main.py`) receives the request.
        4.  **RAG Pipeline:** The backend server performs the *entire* RAG process: embedding your question, searching ChromaDB, and asking `mistral` to generate an answer.
        5.  **API Response (JSON):** The backend sends the answer and sources back to this Streamlit app, which then displays it for you.
        """
    )