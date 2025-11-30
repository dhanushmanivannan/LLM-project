import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# ---------------------------------------------
# LOAD BOOKS CSV
# ---------------------------------------------
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ---------------------------------------------
# LOAD TAGGED_DESCRIPTION.TXT WITH SAFE DECODING
# ---------------------------------------------
try:
    with open("tagged_description.txt", "r", encoding="utf-8") as f:
        text = f.read()
except UnicodeDecodeError:
    with open("tagged_description.txt", "r", encoding="latin-1", errors="ignore") as f:
        text = f.read()

raw_documents = [Document(page_content=text)]

# ---------------------------------------------
# CREATE DOCUMENT CHUNKS
# ---------------------------------------------
text_splitter = CharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
    separator="\n"
)

documents = text_splitter.split_documents(raw_documents)

# ---------------------------------------------
# EMBEDDINGS + CHROMA DB
# ---------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_books = Chroma.from_documents(documents, embeddings)

# ---------------------------------------------
# SEMANTIC RETRIEVAL FUNCTION
# ---------------------------------------------
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)

    # safely extract ISBN from each chunk
    books_list = []
    for rec in recs:
        tokens = rec.page_content.replace('"', '').split()
        if tokens and tokens[0].isdigit():
            books_list.append(int(tokens[0]))

    # filter original DF
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # filter by category
    if category.lower() != "all":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # tone sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


# ---------------------------------------------
# OUTPUT FORMAT FOR GRADIO
# ---------------------------------------------
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"] or ""
        desc_split = description.split()
        truncated_description = " ".join(desc_split[:30]) + "..."

        # clean authors formatting
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        results.append((row["large_thumbnail"], caption))

    return results


# ---------------------------------------------
# GRADIO UI
# ---------------------------------------------
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe the book you want:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Category",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Emotional Tone",
            value="All"
        )
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## 🔍 Recommended Books")
    output = gr.Gallery(label="Results", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# ---------------------------------------------
# RUN APP
# ---------------------------------------------
if __name__ == "__main__":
    dashboard.launch(share=True)
