from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from src.helper import download_embeddings_model
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables FIRST
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_this_in_production'

# Get environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")
index_name = os.getenv('PINECONE_INDEX_NAME')

# Set environment variables (for libraries that need them)
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ['GROQ_API_KEY'] = groq_api_key

# Initialize embeddings and vector store
embeddings_model = download_embeddings_model()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def generation(query, retriever):
    """Generate a response using the medical chatbot"""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    # Create prompt templates
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful medical assistant. "
        "Based only on the context below, answer the question concisely and clearly. "
        "If you do not have enough information, say so. Use three sentence maximum "
        "and keep the answer concise.\n\n"
        "{context}"
    )

    user_prompt = HumanMessagePromptTemplate.from_template(
        "Question: {query}"
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Format and invoke
    final_messages = prompt.format_messages(
        context=context,
        query=query
    )

    response = llm.invoke(final_messages)
    return response.content


@app.route('/get', methods=['POST'])
def chat():
    """Handle chat requests"""
    query = request.form['msg']
    response = generation(query, retriever)
    return str(response)


@app.route('/')
def index():
    """Render the main chat page"""
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)