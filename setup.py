#!/usr/bin/env python3
"""
setup.py — First-time setup: ingest sample documents and build the FAISS index.
Run once before launching the Streamlit app:
    python setup.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("  RAG Chatbot — First-Time Setup")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1/4] Loading documents from data/ …")
    from src.ingest import load_documents, clean_documents, chunk_documents
    docs = load_documents()

    if not docs:
        print("\n⚠️  No documents found in data/.")
        print("    Place files in data/pdfs/, data/txt/, data/md/, etc.")
        print("    Sample documents are included — re-run after adding your own.")
        # Fall through to index what we have (may be nothing)

    # Step 2: Clean
    print("\n[2/4] Cleaning documents …")
    cleaned = clean_documents(docs)

    # Step 3: Chunk
    print("\n[3/4] Chunking …")
    chunks = chunk_documents(cleaned)

    if not chunks:
        print("\n❌  No chunks produced. Add documents to data/ and try again.")
        sys.exit(1)

    # Step 4: Embed and save
    print("\n[4/4] Building embeddings and saving FAISS index …")
    from src.embeddings import get_embedding_model, create_vector_store, save_vector_store
    embed_model = get_embedding_model()
    vector_store = create_vector_store(chunks, embed_model)
    save_vector_store(vector_store)

    print("\n" + "=" * 60)
    print(f"  ✅  Setup complete! Indexed {len(chunks)} chunks.")
    print("  Launch the app with:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()