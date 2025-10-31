import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- Configuration ---
PERSIST_DIRECTORY = "./chroma_db"  # Path to your persistent vector store
LLM_MODEL = "mistral"  # The LLM you pulled
EMBEDDING_MODEL = "nomic-embed-text"  # The embedding model you pulled


# --- Helper Function to Format Context ---
def format_docs(docs):
    """Prepares the context string from retrieved documents."""
    return "\n\n---\n\n".join([d.page_content for d in docs])


# --- Step 4: The RAG Core (Query Pipeline) ---

@st.cache_resource  # Caches the "brain" of our app for performance
def load_rag_pipeline():
    """Loads all the necessary components for the RAG pipeline."""
    try:
        # 1. Initialize your models
        llm = Ollama(model=LLM_MODEL)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 2. Load your existing database from disk
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        # 3. Create the retriever (fetches relevant documents)
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}  # Retrieve the top 5 most similar chunks
        )

        # 4. Define the RAG Prompt Template
        # This tells the LLM how to behave
        template = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Be concise and helpful.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. Build the RAG Chain using LangChain Expression Language (LCEL)
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        return rag_chain, retriever

    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"An error occurred during pipeline setup: {e}")
        return None, None


# --- Step 5: Interactive UI ---

st.set_page_config(page_title="Wikipedia RAG Assistant", layout="wide")
st.title("Wikipedia RAG Assistant (RAG-Wiki-v1) 📚🤖")
st.markdown("---")

# Load the RAG pipeline
rag_chain, retriever = load_rag_pipeline()

if rag_chain is None:
    st.error(
        "**Failed to load RAG pipeline.** "
        "Did you run `build_vector_store.py` first? "
        "Is your `chroma_db` folder in the correct location?"
    )
else:
    st.success(
        f"RAG Pipeline loaded successfully. "
        f"Using **{LLM_MODEL}** and **{EMBEDDING_MODEL}**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Ask a Question")
        st.markdown(
            "Ask any question about the topics you crawled (e.g., Deep Learning, AI, etc.). "
            "The assistant will find relevant info from your 500+ articles and generate an answer."
        )

        # User input
        query = st.text_input("Your question:", key="query_input")

        if st.button("Get Answer", type="primary"):
            if query:
                with st.spinner("Thinking... (This may take a moment)"):
                    try:
                        # --- THIS IS WHERE THE MAGIC HAPPENS ---
                        # 1. The RAG chain is invoked with your query
                        # 2. The retriever finds relevant docs
                        # 3. The docs and query go to the LLM
                        # 4. The LLM generates an answer
                        answer = rag_chain.invoke(query)
                        # ----------------------------------------

                        st.subheader("Answer:")
                        st.write(answer)

                        # (Bonus) Display the source URLs
                        with st.expander("Show Sources (Retrieved Context)"):
                            # We can use the retriever separately to show the docs
                            context_docs = retriever.invoke(query)
                            for i, doc in enumerate(context_docs):
                                st.markdown(f"**Source {i + 1}:** [View Article]({doc.metadata['source']})")
                                st.markdown(
                                    f"> {doc.page_content[:250]}..."
                                )
                                st.markdown("---")

                    except Exception as e:
                        st.error(f"An error occurred while generating the answer: {e}")
            else:
                st.warning("Please enter a question.")

    with col2:
        st.header("How this works")
        st.markdown(
            """
            1.  **Question:** You ask a question.
            2.  **Embed:** Your question is converted into a vector (a list of numbers) using `nomic-embed-text`.
            3.  **Retrieve:** The app searches the **ChromaDB** for text chunks with similar vectors (the 5 most relevant chunks are retrieved).
            4.  **Augment:** Your question and the retrieved text chunks are combined into a single prompt.
            5.  **Generate:** The `mistral` LLM receives this big prompt and generates a final answer based *only* on the provided context.
            """
        )