# 🤖 RAG Application - Multi-Document Chat

A production-ready **Retrieval-Augmented Generation (RAG)** application that enables intelligent conversations with multiple documents simultaneously. Built with FastAPI, LangChain, and FAISS, featuring a clean Apple-inspired UI.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

- **Multi-Document Support**: Upload and chat with multiple documents simultaneously
- **Isolated Chat Sessions**: Each document maintains its own conversation history
- **Smart Document Switching**: Seamlessly switch between different document chats
- **Vector Search**: FAISS-powered semantic search for accurate retrieval
- **Modern UI**: Clean, minimal Apple-inspired interface with 20/80 split layout
- **Conversation Memory**: Maintains context-aware conversations per document
- **Supported Formats**: PDF, TXT, DOCX, and more
- **Production Ready**: Dockerized with CI/CD pipeline support

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
├─────────────────────────────────────────────────────────┤
│  Document Ingestion → Text Splitting → Embeddings       │
│         ↓                                                │
│  FAISS Vector Store → Retrieval → LLM Response          │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
- **Frontend**: Vanilla JavaScript with responsive design
- **Backend**: FastAPI for REST API
- **Vector DB**: FAISS for efficient similarity search
- **LLM Integration**: LangChain with multiple provider support (OpenAI, Google, Groq)
- **Data Versioning**: DVC for tracking large files and indexes

---

## 🚀 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI, Uvicorn |
| **LLM Framework** | LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | HuggingFace, OpenAI |
| **LLM Providers** | OpenAI, Google Gemini, Groq |
| **Document Processing** | PyPDF, python-docx, docx2txt |
| **Data Versioning** | DVC with Google Cloud Storage |
| **Containerization** | Docker |
| **CI/CD** | Jenkins, GCP Cloud Run |

---

## 📋 Prerequisites

- Python 3.11+
- pip or conda
- Git
- DVC (for data versioning)
- Docker (for containerization)
- API keys for LLM providers (OpenAI, Google, or Groq)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-application.git
cd rag-application
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up DVC (Optional - for accessing shared data)

```bash
# Initialize DVC (already done in repo)
dvc pull  # Pull data tracked by DVC from remote storage
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Provider API Keys (choose one or more)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Application Settings
ENV=development
PORT=8000

# Google Cloud (for DVC remote storage)
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

> ⚠️ **Never commit `.env` files to version control!**

---

## 🎯 Usage

### Running Locally

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` in your browser.

### Using the Application

1. **Upload Documents**: Click "Upload Document" in the left sidebar
2. **Add Files**: Drag & drop or select files (PDF, DOCX, TXT)
3. **Start Chatting**: Documents are automatically indexed and ready for chat
4. **Switch Documents**: Click any document in the sidebar to switch chats
5. **Ask Questions**: Each document maintains its own conversation context

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t rag-application:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/faiss_index:/app/faiss_index \
  rag-application:latest
```

---

## 🔧 Project Structure

```
RAG/
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup configuration
├── Dockerfile                   # Container configuration
├── Jenkinsfile                  # CI/CD pipeline
├── .dvcignore                   # DVC ignore rules
├── .gitignore                   # Git ignore rules
├── data.dvc                     # DVC tracking for data/
├── faiss_index.dvc              # DVC tracking for vector indexes
│
├── multi_doc_chat/              # Main application package
│   ├── config/                  # Configuration modules
│   ├── exception/               # Custom exceptions
│   ├── logger/                  # Logging utilities
│   ├── model/                   # Pydantic models
│   ├── prompts/                 # LLM prompt templates
│   ├── src/                     # Core business logic
│   │   ├── document_ingestion/  # Document processing & indexing
│   │   └── document_chat/       # RAG retrieval & chat logic
│   └── utils/                   # Helper functions
│
├── static/                      # Static assets (CSS, JS, images)
├── templates/                   # HTML templates
│   └── index.html               # Main frontend interface
│
├── tests/                       # Unit and integration tests
├── notebook/                    # Jupyter notebooks for experimentation
├── evaluation/                  # Model evaluation scripts
│
├── data/                        # Documents (DVC-tracked, not in Git)
├── faiss_index/                 # Vector indexes (DVC-tracked, not in Git)
└── logs/                        # Application logs (not in Git)
```

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Optional* |
| `GOOGLE_API_KEY` | Google API key for Gemini | Optional* |
| `GROQ_API_KEY` | Groq API key for fast inference | Optional* |
| `ENV` | Environment (development/production) | No |
| `PORT` | Server port (default: 8000) | No |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON | For DVC |

*At least one LLM provider API key is required

---

## 📦 DVC - Data Version Control

This project uses DVC to version control large files (documents, FAISS indexes).

### Setup DVC Remote (First Time)

```bash
# Configure Google Cloud Storage as remote
dvc remote add -d storage gs://your-bucket-name/rag-data

# Push data to remote
dvc push
```

### Working with DVC

```bash
# Pull latest data from remote
dvc pull

# Add new data to DVC tracking
dvc add data/
dvc add faiss_index/

# Commit DVC files to Git
git add data.dvc faiss_index.dvc .dvc/
git commit -m "Update data version"

# Push to DVC remote
dvc push
```

---

## 🚢 CI/CD & Deployment

### Jenkins Pipeline

The `Jenkinsfile` automates:
1. Code checkout
2. Dependency installation
3. Testing
4. Docker image build
5. Push to container registry
6. Deploy to GCP Cloud Run

### Deploy to GCP Cloud Run

```bash
# Build and push to GCP Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-application

# Deploy to Cloud Run
gcloud run deploy rag-application \
  --image gcr.io/PROJECT_ID/rag-application \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=multi_doc_chat tests/

# Run specific test file
pytest tests/test_ingestion.py
```

---

## 📊 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

```http
GET  /health              # Health check
POST /upload              # Upload and index documents
POST /chat                # Send message to document
```

---

## 🎨 UI Features

- **20/80 Split Layout**: Sidebar for documents, main area for chat
- **Multi-Document Management**: Upload and switch between multiple documents
- **Isolated Conversations**: Each document maintains separate chat history
- **Real-time Responses**: Streaming responses with loading indicators
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Persistent Storage**: LocalStorage for session management

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt  # if you have one

# Run tests before committing
pytest tests/

# Format code
black multi_doc_chat/
isort multi_doc_chat/
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Kaustubh Kamble**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- OpenAI, Google, and Groq for LLM APIs

---

## 📮 Support

If you encounter any issues or have questions:
- Open an [Issue](https://github.com/yourusername/rag-application/issues)
- Check the [Discussions](https://github.com/yourusername/rag-application/discussions)
- Read the [Wiki](https://github.com/yourusername/rag-application/wiki)

---

## 🗺️ Roadmap

- [ ] Add support for more document formats (Markdown, HTML, CSV)
- [ ] Implement user authentication
- [ ] Add document search across all uploaded files
- [ ] Support for YouTube video transcripts
- [ ] Web scraping capabilities
- [ ] Advanced RAG techniques (HyDE, Self-Query)
- [ ] Multi-language support
- [ ] Export chat history
- [ ] Document comparison feature
- [ ] Cloud storage integration (S3, GCS, Azure)

---

<div align="center">

**Built with ❤️ using FastAPI, LangChain, and FAISS**

⭐ Star this repo if you find it helpful!

</div>
