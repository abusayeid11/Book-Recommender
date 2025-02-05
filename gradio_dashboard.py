import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


books = pd.read_csv('books_with_emotions.csv')

books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail']
)

raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(raw_documents)

db_books = Chroma.from_documents(documents, embedding=embedding_model)

def retrieve_semantic_recommendations(
        query: str,
        category: str=None,
        tone: str=None,
        initial_top_k: int=50,
        final_top_k: int=16
) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    book_list = [
    int(first_token) for rec, _ in recs  # Unpack the tuple (document, score)
    if (first_token := rec.page_content.strip('"').split()[0]).isdigit()
]
    book_recs = books[books['isbn13'].isin(book_list)]

    if category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category]

    book_recs = book_recs.head(final_top_k)

    if tone == 'Happy':
        book_recs = book_recs.sort_values(by='joy', ascending=False)
    elif tone == 'Surprising':
        book_recs = book_recs.sort_values(by='surprise', ascending=False)
    elif tone == 'Angry':
        book_recs = book_recs.sort_values(by='angry', ascending=False)
    elif tone == 'Suspense':
        book_recs = book_recs.sort_values(by='fear', ascending=False)
    elif tone == 'Sad':
        book_recs = book_recs.sort_values(by='sadness', ascending=False)

    return book_recs

def recommend_book(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row['authors'].split(';')

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row['large_thumbnail'], caption))  # Fix here: tuple instead of multiple args

    return results

tones = ["All", "Happy", "Suspense", "Surprising", "Sad", "Angry"]
categories = ["All"] + sorted(books['simple_categories'].unique())

with gr.Blocks(gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label='Please enter description of a book',
                                placeholder='e.g., A story about forgiveness')
        category_dropdown = gr.Dropdown(choices=categories, label='Select a category:', value='All')
        tone_dropdown = gr.Dropdown(choices=tones, label='Select an emotional tone:', value='All')
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", rows=2, columns=8)

    submit_button.click(recommend_book, inputs=[user_query, category_dropdown, tone_dropdown], outputs=[output])

if __name__ == "__main__":
    dashboard.launch()
