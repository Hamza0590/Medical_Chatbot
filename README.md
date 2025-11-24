# Medical Chatbot 🏥🤖

A medical chatbot powered by LangChain, Pinecone, and Groq's LLaMA 3.3 70B model. This chatbot can answer medical questions based on a medical book PDF using RAG (Retrieval Augmented Generation).

## Features

- 📚 PDF document processing and chunking
- 🔍 Semantic search using Pinecone vector database
- 🤖 AI-powered responses using Groq's LLaMA 3.3 70B
- 💬 Clean web interface built with Flask
- ⚡ Fast and accurate medical information retrieval
- 🧠 **Conversation context** - Remembers chat history for contextual responses
- 🗑️ Clear chat functionality to reset conversations

## Tech Stack

- **Backend**: Flask
- **LLM**: Groq (LLaMA 3.3 70B Versatile)
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Framework**: LangChain

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Hamza0590/Medical_Chatbot.git
cd Medical_Chatbot
```

### 2. Create Virtual Environment

```bash
conda create -n chatbot python=3.12 -y
conda activate chatbot
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory (use `.env.example` as template):

```bash
PINECONE_API_KEY="your_pinecone_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
PINECONE_INDEX_NAME="medical-chatbot"
```

**Get your API keys:**
- Pinecone: [https://www.pinecone.io/](https://www.pinecone.io/)
- Groq: [https://console.groq.com/](https://console.groq.com/)

### 5. Add Your Medical PDF

Place your medical book PDF in the `data/` folder:

```
data/
  └── Medical_book.pdf
```

### 6. Create Vector Store Index

Run the indexing script to process the PDF and create embeddings:

```bash
python store_index.py
```

This will:
- Load the PDF from `data/` folder
- Split it into chunks
- Generate embeddings
- Store them in Pinecone

### 7. Run the Application

```bash
python app.py
```

The app will be available at: `http://localhost:5000`

## Project Structure

```
Medical_Chatbot/
├── data/                      # PDF files
│   └── Medical_book.pdf
├── src/                       # Source code
│   ├── __init__.py
│   ├── helper.py             # Helper functions
│   └── prompt.py             # Prompt templates
├── templates/                 # HTML templates
│   └── chat.html
├── static/                    # CSS and static files
│   └── style.css
├── app.py                    # Flask application
├── store_index.py            # Script to create vector store
├── requirement.txt           # Dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Type your medical question in the chat interface
3. The chatbot will retrieve relevant information from the medical book and provide an answer
4. Ask follow-up questions - the bot remembers your conversation context
5. Click the "Clear" button to reset the conversation and start fresh

## Example Questions

- "What is Anthrax?"
- "What are the symptoms of diabetes?"
- "How is pneumonia treated?"
- "What causes hypertension?"

## How It Works

1. **Document Processing**: The PDF is loaded and split into smaller chunks
2. **Embedding Generation**: Each chunk is converted into vector embeddings
3. **Vector Storage**: Embeddings are stored in Pinecone for fast retrieval
4. **Query Processing**: User questions are embedded and similar chunks are retrieved
5. **Context Building**: Combines retrieved documents with conversation history (last 3 exchanges)
6. **Response Generation**: Retrieved context and chat history are sent to LLaMA 3.3 70B to generate contextual answers

## Important Notes

⚠️ **Security**: Never commit your `.env` file to version control. It contains sensitive API keys.

⚠️ **Disclaimer**: This chatbot is for educational purposes only. Always consult qualified healthcare professionals for medical advice.

## Limitations

### Technical Limitations

- **Knowledge Base Scope**: Responses are limited to the content of the uploaded medical PDF. The chatbot cannot access information beyond this document.
- **Context Window**: Only the last 3 conversation exchanges (6 messages) are included in the context to avoid token limits.
- **Session-Based Memory**: Chat history is stored in browser sessions and will be lost when the session expires or browser is closed.
- **Single User Sessions**: Each browser session maintains separate conversation history; no cross-session memory.
- **No Persistent Storage**: Conversations are not saved to a database and cannot be retrieved after clearing or session expiration.
- **Embedding Model Constraints**: Uses a lightweight embedding model (all-MiniLM-L6-v2) which may not capture all semantic nuances.
- **Retrieval Accuracy**: Returns top 5 most similar chunks, which may not always include all relevant information.

### Medical & Content Limitations

- **Not a Medical Professional**: This is an AI chatbot, not a licensed healthcare provider. It cannot diagnose, treat, or provide personalized medical advice.
- **Information Currency**: The chatbot's knowledge is limited to the publication date of the source PDF and does not include recent medical research or updates.
- **No Emergency Response**: Cannot handle medical emergencies. Always call emergency services (911 or local equivalent) for urgent medical situations.
- **Lack of Context**: Cannot consider individual patient history, symptoms, medications, or personal health factors.
- **Potential Inaccuracies**: AI-generated responses may contain errors or misinterpretations. Always verify information with qualified healthcare professionals.
- **No Visual Diagnosis**: Cannot analyze images, lab results, X-rays, or other diagnostic materials.
- **Limited Scope**: May not cover all medical conditions, treatments, or specialties depending on the source document.

### Usage Limitations

- **Response Length**: Answers are limited to 3 sentences maximum for conciseness, which may oversimplify complex topics.
- **No Multi-Language Support**: Currently operates in English only.
- **Internet Required**: Requires active internet connection for API calls to Groq and Pinecone.
- **API Rate Limits**: Subject to rate limits from Groq and Pinecone APIs.
- **No Authentication**: No user accounts or personalized experiences.
- **Single Document**: Can only query one PDF at a time; switching documents requires re-indexing.
- **Processing Time**: Initial setup (indexing) can take several minutes depending on PDF size.

### Privacy & Security Limitations

- **Data Transmission**: User queries are sent to third-party APIs (Groq, Pinecone).
- **No Encryption**: Chat history is stored in Flask sessions without additional encryption.
- **Session Security**: Uses a placeholder secret key that should be changed in production.
- **No HIPAA Compliance**: Not designed for handling protected health information (PHI).

### Recommendations

✅ **DO USE** for educational purposes, general medical information, and learning about medical concepts.  
❌ **DO NOT USE** for diagnosing conditions, making treatment decisions, or replacing professional medical consultation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Hamza**
- GitHub: [@Hamza0590](https://github.com/Hamza0590)

## Acknowledgments

- LangChain for the RAG framework
- Groq for providing fast LLM inference
- Pinecone for vector database
- HuggingFace for embedding models