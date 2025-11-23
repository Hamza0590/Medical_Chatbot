# Medical Chatbot 🏥🤖

A medical chatbot powered by LangChain, Pinecone, and Groq's LLaMA 3.3 70B model. This chatbot can answer medical questions based on a medical book PDF using RAG (Retrieval Augmented Generation).

## Features

- 📚 PDF document processing and chunking
- 🔍 Semantic search using Pinecone vector database
- 🤖 AI-powered responses using Groq's LLaMA 3.3 70B
- 💬 Clean web interface built with Flask
- ⚡ Fast and accurate medical information retrieval

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
5. **Response Generation**: Retrieved context is sent to LLaMA 3.3 70B to generate accurate answers

## Important Notes

⚠️ **Security**: Never commit your `.env` file to version control. It contains sensitive API keys.

⚠️ **Disclaimer**: This chatbot is for educational purposes only. Always consult qualified healthcare professionals for medical advice.

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