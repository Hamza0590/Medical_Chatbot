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


def generation(query, retriever, chat_history=None):
    """Generate a response using the medical chatbot with conversation context"""
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )

    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    # Build chat history string
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 3 exchanges (6 messages)
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    
    # Create prompt templates with chat history
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful medical assistant. "
        "Based on the context below and the conversation history, answer the question concisely and clearly. "
        "If you do not have enough information, say 'Please ask a different question'. Use three sentence maximum "
        "and keep the answer concise.\n\n"
        "Context from medical documents:\n{context}\n\n"
        "Conversation History:\n{history}"
    )

    user_prompt = HumanMessagePromptTemplate.from_template(
        "Question: {query}"
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    # Format and invoke
    final_messages = prompt.format_messages(
        context=context,
        history=history_text if history_text else "No previous conversation.",
        query=query
    )

    response = llm.invoke(final_messages)
    return response.content


@app.route('/get', methods=['POST'])
def chat():
    """Handle chat requests with conversation history"""
    query = request.form['msg']
    
    # Initialize chat history in session if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Get response with chat history
    response = generation(query, retriever, session['chat_history'])
    
    # Store user message and bot response in session
    session['chat_history'].append({'role': 'user', 'content': query})
    session['chat_history'].append({'role': 'assistant', 'content': response})
    
    # Keep only last 20 messages to prevent session from growing too large
    if len(session['chat_history']) > 20:
        session['chat_history'] = session['chat_history'][-20:]
    
    session.modified = True
    
    return str(response)


@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'success', 'message': 'Chat history cleared'})


@app.route('/')
def index():
    """Render the main chat page"""
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)