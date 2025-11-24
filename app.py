from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from src.helper import load_pdf, filter_to_minimal_docs, text_split, download_embeddings_model
import os
import traceback
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import List, Dict, Tuple
from collections import defaultdict
import threading
import time

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_this_in_production'


pinecone_api_key = os.getenv('PINECONE_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")
index_name = os.getenv('PINECONE_INDEX_NAME')

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ['GROQ_API_KEY'] = groq_api_key


embeddings_model = None
docsearch = None
retriever = None
initialization_complete = False
initialization_lock = threading.Lock()


def initialize_models():
    global embeddings_model, docsearch, retriever, initialization_complete


    try:

        embeddings_model = download_embeddings_model()
        pc = Pinecone(api_key=pinecone_api_key)
        if not pc.has_index(index_name):

            documents = load_pdf()
            minimal_docs = filter_to_minimal_docs(documents)
            text_chunks = text_split(minimal_docs)

            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

            docsearch = PineconeVectorStore.from_documents(
                documents=text_chunks,
                embedding=embeddings_model,
                index_name=index_name
            )           
        else:
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=embeddings_model
            )

        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        with initialization_lock:
            initialization_complete = True
        
    except Exception as e:
        traceback.print_exc()
        
        with initialization_lock:
            initialization_complete = False



def wait_for_initialization(timeout=10):

    start_time = time.time()
    while not initialization_complete:
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.1)
    return True


def generate_query_variations(original_query: str, num_variations: int = 5) -> List[str]:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7  
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a medical query expansion expert. Given a medical question, "
            "generate {num_variations} alternative ways to ask the same question. "
            "Each variation should capture different aspects or phrasings while maintaining the core intent.\n\n"
            "Rules:\n"
            "1. Keep variations medically accurate\n"
            "2. Use different medical terminology where appropriate\n"
            "3. Vary the question structure\n"
            "4. Each variation on a new line\n"
            "5. Do not number the variations\n\n"
            "Original question: {query}\n\n"
            "Generate {num_variations} variations:"
        )
    ])
    
    messages = prompt.format_messages(query=original_query, num_variations=num_variations)
    response = llm.invoke(messages)
    
    variations = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
    
    all_queries = [original_query] + variations[:num_variations]
    
    return all_queries


def reciprocal_rank_fusion(doc_lists: List[List[any]], k: int = 60) -> List[Tuple[any, float]]:
    """
    Re-rank documents using Reciprocal Rank Fusion (RRF).
    RRF score = sum(1 / (k + rank)) for each document across all queries.
    
    Args:
        doc_lists: List of document lists from different queries
        k: Constant for RRF (typically 60)
    
    Returns:
        List of (document, score) tuples sorted by RRF score
    """
    doc_scores = defaultdict(float)
    doc_objects = {}
    
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):

            doc_id = doc.page_content
            doc_scores[doc_id] += 1.0 / (k + rank)
            doc_objects[doc_id] = doc
    
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_objects[doc_id], score) for doc_id, score in ranked_docs]


def advanced_retrieval(query: str, retriever, num_variations: int = 3, top_k: int = 6) -> str:
    print("top_k", top_k, "num_variations", num_variations)
    query_variations = generate_query_variations(query, num_variations)
    doc_lists = []
    for q in query_variations:
        docs = retriever.invoke(q)
        doc_lists.append(docs)
    
    ranked_docs = reciprocal_rank_fusion(doc_lists)
    
    top_docs = ranked_docs[:top_k]
    
    context = "\n\n".join([doc.page_content for doc, score in top_docs]) 
    return context


def generation(query, retriever, chat_history=None, use_multi_query=True):

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )


    if use_multi_query:
        start = time.time()
        context = advanced_retrieval(query, retriever, num_variations=3, top_k=3)
        end = time.time()
        print(f"Advanced retrieval took {end - start:.2f} seconds")
    else:
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)


    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    
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

    final_messages = prompt.format_messages(
        context=context,
        history=history_text if history_text else "No previous conversation.",
        query=query
    )

    response = llm.invoke(final_messages)
    return response.content


@app.route('/status', methods=['GET'])
def status():
    """Check if initialization is complete"""
    return jsonify({
        'initialized': initialization_complete,
        'message': 'Ready' if initialization_complete else 'Initializing...'
    })


@app.route('/get', methods=['POST'])
def chat():
    """Handle chat requests with conversation history"""

    if not initialization_complete:
        if not wait_for_initialization(timeout=30):
            return jsonify({
                'error': 'System is still initializing. Please wait a moment and try again.'
            }), 503
    
    query = request.form['msg']
    

    if 'chat_history' not in session:
        session['chat_history'] = []
    
    try:
        response = generation(query, retriever, session['chat_history'], use_multi_query=True)
        
        session['chat_history'].append({'role': 'user', 'content': query})
        session['chat_history'].append({'role': 'assistant', 'content': response})
        
        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
        
        session.modified = True
        
        return str(response)
    
    except Exception as e:
        print(f"Error in chat: {e}")
        return str(f"Sorry, I encountered an error: {str(e)}")


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
    init_thread = threading.Thread(target=initialize_models, daemon=True)
    init_thread.start()
    app.run(debug=True, use_reloader=False)  