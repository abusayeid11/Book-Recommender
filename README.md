# Semantic Book Recommender

## Overview
The **Semantic Book Recommender** is a book recommendation system that leverages **vector search**, **text classification**, and **emotion analysis** to provide personalized book suggestions. It uses **semantic similarity** to match user queries with relevant books and refines recommendations based on **categories** and **emotional tone**. The system is built with **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and features a **Gradio** interface for user interaction.

## Features
- **Semantic Search with Vector Embeddings**: Uses **HuggingFace's all-MiniLM-L6-v2** embeddings to perform similarity-based book retrieval.
- **Emotion-Based Filtering**: Books are ranked based on detected emotional tones like *Happy, Suspense, Surprising, Sad, and Angry*.
- **Text Classification**: Books are categorized into different genres for more relevant recommendations.
- **Gradio Interface**: Provides an easy-to-use **dashboard** for users to input queries and receive book recommendations.

## Tech Stack
- **Python** (Core language)
- **LangChain** (Document processing and vector search)
- **ChromaDB** (Vector database for storing book embeddings)
- **HuggingFace Embeddings** (Text embeddings for semantic search)
- **Pandas & NumPy** (Data processing)
- **Gradio** (User interface)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed. Then, install the required dependencies:

```sh
pip install pandas numpy langchain langchain_chroma langchain_openai gradio chromadb dotenv
```

## Usage
### 1. Prepare the Data
- **Books Dataset**: Load `books_with_emotions.csv`, which includes book metadata, thumbnails, and emotion scores.
- **Document Processing**: Load `tagged_description.txt` to create a searchable database using LangChain's `TextLoader`.

### 2. Start the Recommender System
Run the following command to launch the **Gradio dashboard**:

```sh
python app.py
```

### 3. Use the Interface
- **Enter a book description** (e.g., *A mystery thriller with unexpected twists*).
- **Select a category** (or choose "All" for broader results).
- **Pick an emotional tone** (or choose "All").
- Click **Find Recommendations** to see results.

## Code Explanation
### **Key Functions**
#### `retrieve_semantic_recommendations(query, category, tone)`
- Performs a **vector similarity search** using ChromaDB.
- Filters books by **category** and sorts them based on the chosen **emotional tone**.

#### `recommend_book(query, category, tone)`
- Calls `retrieve_semantic_recommendations` and formats results for display.
- Truncates long descriptions and properly formats author names.

## Future Improvements
- **Fine-tune embeddings** for better semantic understanding.
- **Enhance emotion analysis** using a more robust sentiment model.
- **Improve UI/UX** with better visuals and more filtering options.


