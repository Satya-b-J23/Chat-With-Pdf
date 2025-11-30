DocuChat AI: Intelligent PDF Conversational Assistant 📄🤖
DocuChat AI is a robust Retrieval-Augmented Generation (RAG) application designed to bridge the gap between static documents and dynamic conversation. By leveraging Google's Gemini Flash models and LangChain, this tool allows users to upload multiple PDF documents and ask complex questions, receiving accurate, context-aware answers in seconds.

🚀 Key Features
Multi-Document RAG Engine: Upload and process multiple PDF files simultaneously to create a unified knowledge base.

High-Performance Vector Search: Utilizes FAISS (Facebook AI Similarity Search) for millisecond-latency retrieval of relevant context, with a robust NumPy fallback.

State-of-the-Art Embeddings: Powered by sentence-transformers/all-MiniLM-L6-v2 to convert text into semantic vector representations.

Intelligent Generation: Integrated with Google Gemini (Flash) for high-speed, cost-effective natural language generation.

Professional UI: A clean, responsive interface built with Streamlit, featuring session state management and a chat-like experience.

🛠️ Tech Stack
Frontend: Streamlit

LLM Orchestration: LangChain

Generative AI: Google Gemini (via langchain-google-genai)

Vector Store: FAISS / NumPy

Embeddings: HuggingFace Sentence Transformers

PDF Parsing: PyPDF2

⚙️ Installation & Setup
Clone the Repository

Bash

git clone https://github.com/your-username/docuchat-ai.git
cd docuchat-ai
Install Dependencies

Bash

pip install -r requirements.txt
Configure Environment Create a .env file in the root directory and add your Google API key:

Code snippet

GOOGLE_API_KEY=your_api_key_here
Run the Application

Bash

streamlit run app.py
🧠 How It Works
Ingestion: The app reads uploaded PDFs and extracts raw text.

Chunking: Text is split into smaller, manageable segments (1000 chars) to optimize context retention.

Embedding: Each chunk is converted into a vector using the embedding model.

Retrieval: When you ask a question, the system finds the most similar text chunks using cosine similarity.

Generation: The relevant chunks + your question are sent to Gemini, which generates a human-like answer based only on your documents.
