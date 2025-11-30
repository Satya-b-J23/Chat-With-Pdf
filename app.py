import os
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings

# -------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# Suppress unnecessary warnings and load environment variables for API keys.
# -------------------------------------------------------------------------
warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="DocuChat AI", page_icon="📄", layout="wide")

# -------------------------------------------------------------------------
# CONSTANTS & MODEL CONFIGURATION
# Define model parameters centrally to allow for easy updates.
# -------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
EMBED_BATCH_SIZE = 4
TOP_K_RESULTS = 5

# -------------------------------------------------------------------------
# CUSTOM CSS STYLING
# Inject custom CSS to create a distinct chat interface (user vs. bot bubbles).
# -------------------------------------------------------------------------
st.markdown("""
<style>
    .chat-msg { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; flex-direction: column; }
    .user-msg { background-color: #f0f2f6; color: #31333F; }
    .bot-msg { background-color: #ffffff; border: 1px solid #e0e0e0; color: #31333F; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# MODEL LOADING
# Use Streamlit's caching to prevent reloading the 80MB+ model on every interaction.
# -------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Load the SentenceTransformer model for generating text embeddings."""
    return SentenceTransformer(EMBED_MODEL_NAME)

# -------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# Gracefully handle the absence of FAISS (used for faster vector search).
# -------------------------------------------------------------------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# -------------------------------------------------------------------------
# DOCUMENT PROCESSING FUNCTIONS
# Functions to extract, split, and embed text from uploaded PDF files.
# -------------------------------------------------------------------------
def get_pdf_text(pdf_files):
    """Extracts raw text content from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def get_chunks(text):
    """Splits raw text into manageable chunks with overlap to maintain context."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_embeddings(texts, model):
    """Converts a list of text chunks into numerical vectors (embeddings)."""
    if not texts:
        return np.array([])
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)

# -------------------------------------------------------------------------
# VECTOR SEARCH FUNCTIONS
# Functions to index embeddings and retrieve relevant context for queries.
# -------------------------------------------------------------------------
def create_faiss_index(embeddings):
    """Builds a FAISS index for high-speed similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search_documents(query, model, index, embeddings, chunks):
    """
    Searches for the most relevant document chunks based on the user query.
    Uses FAISS if available; otherwise falls back to standard cosine similarity.
    """
    query_vec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    
    if FAISS_AVAILABLE and index is not None:
        faiss.normalize_L2(query_vec.reshape(1, -1))
        _, indices = index.search(query_vec.reshape(1, -1), TOP_K_RESULTS)
        top_indices = indices[0]
    else:
        # Fallback: Numpy-based cosine similarity
        norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_query = query_vec / np.linalg.norm(query_vec)
        sims = np.dot(norm_emb, norm_query)
        top_indices = np.argsort(sims)[::-1][:TOP_K_RESULTS]

    results = []
    for idx in top_indices:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
    return results

# -------------------------------------------------------------------------
# GEN-AI INTEGRATION
# Connects to Google Gemini API to generate natural language responses.
# -------------------------------------------------------------------------
def get_ai_response(query, context):
    """Generates an answer using Google Gemini based strictly on the provided context."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: API Key missing."
    
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=api_key)
        prompt = f"""
        Answer the question based strictly on the context below. 
        If the answer is not in the context, state that you don't know.
        
        Context:
        {context}
        
        Question: {query}
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"AI Service Error: {str(e)}"

# -------------------------------------------------------------------------
# MAIN APPLICATION LOGIC
# Orchestrates the UI layout, state management, and interaction flow.
# -------------------------------------------------------------------------
def main():
    # Initialize session state for chat history and vector store
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = {"index": None, "embeddings": None, "chunks": []}

    embed_model = load_embedding_model()

    # Sidebar: File Upload & Processing
    with st.sidebar:
        st.title("📁 Document Hub")
        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        
        if st.button("Analyze Documents", type="primary"):
            if not uploaded_files:
                st.warning("Please select files first.")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(uploaded_files)
                    if raw_text:
                        chunks = get_chunks(raw_text)
                        embeddings = get_embeddings(chunks, embed_model)
                        
                        # Build Index
                        index = None
                        if FAISS_AVAILABLE:
                            index = create_faiss_index(embeddings)
                        
                        # Update State
                        st.session_state.vector_store = {
                            "index": index,
                            "embeddings": embeddings,
                            "chunks": chunks
                        }
                        st.success(f"Processed {len(chunks)} text segments successfully.")
                    else:
                        st.error("Could not extract text from documents.")

    # Main Chat Interface
    st.header("Chat Interface")
    
    # Display Chat History
    for msg in st.session_state.messages:
        role_class = "user-msg" if msg["role"] == "user" else "bot-msg"
        st.markdown(f"<div class='chat-msg {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Handle User Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='chat-msg user-msg'>{prompt}</div>", unsafe_allow_html=True)

        store = st.session_state.vector_store
        if not store["chunks"]:
            response = "Please upload and process documents before asking questions."
        else:
            with st.spinner("Thinking..."):
                # Retrieve relevant context
                context_chunks = search_documents(
                    prompt, 
                    embed_model, 
                    store["index"], 
                    store["embeddings"], 
                    store["chunks"]
                )
                context_text = "\n\n".join(context_chunks)
                
                # Generate AI Response
                response = get_ai_response(prompt, context_text)

        # Display and Save Bot Response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"<div class='chat-msg bot-msg'>{response}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()