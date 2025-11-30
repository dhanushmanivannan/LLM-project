# 📚 Semantic Book Recommendation System

An intelligent book recommendation system that combines Natural Language Processing (NLP), sentiment analysis, and semantic search to provide personalized book recommendations. Built with LangChain, Hugging Face embeddings, ChromaDB vector database, and an interactive Gradio interface.

## 🌟 Features

- **Semantic Search**: Uses sentence transformers and vector embeddings to understand the meaning behind your queries
- **Emotion-Based Filtering**: Find books by emotional tone (Happy, Surprising, Angry, Suspenseful, Sad)
- **Category Filtering**: Browse recommendations by book categories
- **Interactive Web Interface**: User-friendly Gradio dashboard with visual book gallery
- **Advanced NLP Pipeline**: Includes sentiment analysis and text classification
- **Vector Database**: Powered by ChromaDB for efficient similarity search

## 🛠️ Technologies Used

- **LangChain**: Document processing and text splitting
- **Hugging Face Transformers**: Sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **ChromaDB**: Vector database for semantic search
- **Gradio**: Interactive web dashboard
- **Pandas & NumPy**: Data manipulation and analysis
- **Python 3.8+**: Core programming language

## 📋 Prerequisites

```bash
Python 3.8 or higher
pip (Python package manager)
```

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/dhanushmanivannan/LLM-project.git
cd LLM-project
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the project root (if needed for API keys)

## 📦 Dependencies

```
pandas
numpy
python-dotenv
langchain
langchain-core
langchain-huggingface
langchain-chroma
chromadb
sentence-transformers
gradio
jupyter
notebook
```

## 💻 Usage

### Running the Main Application

```bash
python gradio-dashboard.py
```

The application will launch a Gradio interface accessible at `http://localhost:7860` (or a public URL if share=True is enabled).

### Using the Interface

1. **Describe your book**: Enter a natural language query (e.g., "A story about forgiveness")
2. **Select a category**: Choose a specific genre or "All" for broader results
3. **Choose emotional tone**: Filter by mood (Happy, Surprising, Angry, Suspenseful, Sad, or All)
4. **Get recommendations**: Click "Find Recommendations" to see personalized results with book covers

### Exploring the Notebooks

The project includes several Jupyter notebooks for different stages of the pipeline:

- `data-exploration.ipynb` - Initial data analysis and exploration
- `sentiment-analysis.ipynb` - Emotion detection and classification
- `text-classification.ipynb` - Book categorization
- `vector-search.ipynb` - Semantic search implementation
- `nlp_project.ipynb` - Main NLP pipeline

## 🏗️ Project Structure

```
LLM-project/
│
├── gradio-dashboard.py           # Main application with Gradio UI
├── gradio-sample.py              # Sample Gradio implementation
├── data-exploration.ipynb        # Data analysis notebook
├── sentiment-analysis.ipynb      # Sentiment analysis pipeline
├── text-classification.ipynb     # Text classification models
├── vector-search.ipynb           # Vector search implementation
├── nlp_project.ipynb            # Complete NLP pipeline
│
├── tagged_description.txt        # Processed book descriptions
├── books_cleaned.csv            # Cleaned book dataset
├── books_with_categories.csv    # Books with category labels
├── books_with_emotions.csv      # Books with emotion scores
│
├── .env                         # Environment variables
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## 🎯 How It Works

### 1. Data Processing
- Book data is loaded from CSV files containing titles, authors, descriptions, categories, and emotion scores
- Descriptions are processed and split into chunks for efficient retrieval

### 2. Embedding Generation
- Uses Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model
- Converts text into dense vector representations capturing semantic meaning

### 3. Vector Database
- ChromaDB stores and indexes book embeddings
- Enables fast similarity search for relevant recommendations

### 4. Semantic Retrieval
- User queries are embedded using the same model
- Similarity search finds the most relevant books (top 50 initially)
- Results are filtered by category and emotional tone
- Final top 16 recommendations are returned

### 5. Emotion Analysis
The system analyzes books across five emotional dimensions:
- **Joy** (Happy books)
- **Surprise** (Surprising narratives)
- **Anger** (Intense, provocative content)
- **Fear** (Suspenseful, thrilling stories)
- **Sadness** (Emotional, melancholic themes)

## 📊 Features Deep Dive

### Semantic Search
Unlike keyword matching, semantic search understands the **meaning** behind your query. For example, "a story about redemption" will match books about forgiveness, second chances, and personal transformation.

### Multi-Stage Filtering
1. Initial semantic search retrieves top 50 matches
2. Category filter narrows results to selected genre
3. Emotion sorting prioritizes books matching desired tone
4. Final 16 recommendations displayed with covers and descriptions

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more emotion categories
- Integrate with external book APIs (Goodreads, Open Library)
- Implement user rating system
- Add book similarity comparisons
- Enhance UI/UX design

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Dhanush Manivannan**

- GitHub: [@dhanushmanivannan](https://github.com/dhanushmanivannan)

## 🙏 Acknowledgments

- Hugging Face for sentence-transformers models
- LangChain for the powerful document processing framework
- ChromaDB team for the vector database
- Gradio for the intuitive web interface framework
- The open-source NLP community

## 🔮 Future Enhancements

- [ ] User authentication and personalized recommendation history
- [ ] Multi-language support
- [ ] Integration with Goodreads API for ratings and reviews
- [ ] Advanced filtering (publication year, page count, ratings)
- [ ] Book comparison feature
- [ ] Export recommendations as PDF or CSV
- [ ] Collaborative filtering for "users like you" recommendations
- [ ] Fine-tune embeddings on book-specific corpus

## 📧 Contact

For questions, suggestions, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/dhanushmanivannan/LLM-project/issues)

## 🎓 Learn More

This project demonstrates practical applications of:
- Vector embeddings and semantic search
- Sentiment analysis and emotion detection
- RAG (Retrieval Augmented Generation) concepts
- Building production-ready ML applications

---

⭐ **If you find this project helpful, please consider giving it a star!**
