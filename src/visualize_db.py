import os
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pdf_helpers.helper_vsdb import load_split_pdf, reset_vector_store_db
from pdf_helpers.helper_llm import embeddings
from langchain_chroma import Chroma

# Configure logging to overwrite the log file each run
logging.basicConfig(filename='visualization_log.txt', filemode='w', level=logging.INFO, format='%(message)s')

def create_db_content(file_path):
    """
    Loads and splits the PDF document into chunks and prepares data for the vector store.
    Returns:
        documents (list): List of document texts.
        metadatas (list): List of metadata dictionaries for each document.
        ids (list): List of unique IDs for each document.
    """
    doc = load_split_pdf(file_path)

    documents = []
    metadatas = []
    ids = []

    for idx, doc_chunk in enumerate(doc):
        doc_chunk.metadata['chunk_id'] = idx
        documents.append(doc_chunk.page_content)
        metadatas.append(doc_chunk.metadata)
        ids.append(str(idx))

    return documents, metadatas, ids

def create_vector_store(db_directory, embedding_function, documents, metadatas, ids):
    """
    Creates a Chroma vector store and adds the provided texts.
    Args:
        db_directory (str): Directory to persist the database.
        embedding_function (callable): Function to compute embeddings.
        documents (list): List of document texts.
        metadatas (list): List of metadata dictionaries.
        ids (list): List of document IDs.
    Returns:
        db (Chroma): The created Chroma vector store database.
    """
    db = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_function,
    )
    db.add_texts(texts=documents, metadatas=metadatas, ids=ids)
    return db


def inspect_vector_store(db):
    """
    Retrieves data from the vector store database and returns embeddings array, documents, id_to_index mapping.
    Args:
        db (Chroma): The Chroma vector store database.
    Returns:
        embeddings_array (np.ndarray): Array of embeddings.
        documents (list): List of document texts.
        id_to_index (dict): Mapping from document IDs to their indices.
    """
    collection = db._collection
    logging.info("Collection Name: %s", collection.name)
    logging.info("Number of Documents: %d", collection.count())
    logging.info("Collection Metadata: %s", collection.metadata)

    results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    embeddings_list = results['embeddings']
    metadatas = results['metadatas']
    documents = results['documents']
    ids = results['ids']

    logging.info("\nDocuments in the Collection:")
    for idx in range(len(ids)):
        logging.info(f"Document ID: {ids[idx]}")
        logging.info(f"Embedding: {embeddings_list[idx][:5]}...")
        logging.info(f"Metadata: {metadatas[idx]}")
        logging.info(f"Document Content: {documents[idx]}")
        logging.info("-----\n")

    id_to_index = {ids[idx]: idx for idx in range(len(ids))}
    embeddings_array = np.array(embeddings_list)

    return embeddings_array, documents, id_to_index


def visualize_embeddings_with_query(embeddings, documents, query_embedding=None, retrieved_indices=None):
    """
    Visualizes embeddings in 2D space using PCA.
    Optionally includes a query embedding and highlights retrieved documents.
    Args:
        embeddings (np.ndarray): Array of document embeddings.
        documents (list): List of document texts.
        query_embedding (np.ndarray, optional): The query embedding.
        retrieved_indices (list, optional): Indices of retrieved documents.
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], label='Documents', c='blue')

    for i, doc in enumerate(documents):
        plt.annotate(f'Doc {i}', (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    if query_embedding is not None:
        query_embedding_2d = pca.transform(query_embedding)
        plt.scatter(query_embedding_2d[0, 0], query_embedding_2d[0, 1],
                    label='Query', c='red', marker='X', s=100)
        plt.annotate('Query', (query_embedding_2d[0, 0], query_embedding_2d[0, 1]))

    if retrieved_indices:
        plt.scatter(embeddings_2d[retrieved_indices, 0], embeddings_2d[retrieved_indices, 1],
                    edgecolors='green', facecolors='none', s=200, linewidths=2, label='Retrieved Docs')

    plt.title("Embeddings Visualization with Query")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


def create_retriever(db):
    """
    Creates a retriever from the database.
    Args:
        db (Chroma): The Chroma vector store database.
    Returns:
        retriever: The retriever object.
    """
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever


def main():
    # Specify the persist directory and file path
    file_path = "data/pdfs/Animal_facts.pdf"
    persist_directory = "./data/chroma"

    # Reset the vector store database
    reset_vector_store_db(persist_directory)

    # Create database content from the PDF file
    documents, metadatas, ids = create_db_content(file_path)

    # Create the vector store database and add the content
    db = create_vector_store(persist_directory, embeddings, documents, metadatas, ids)

    # Inspect the database and get embeddings and mappings
    embeddings_array, documents, id_to_index = inspect_vector_store(db)

    # Visualize the embeddings without any query
    visualize_embeddings_with_query(embeddings_array, documents)

    # Create the retriever
    retriever = create_retriever(db)

    # Define a query
    query = "How long is a giraffe's tongue?"

    # Compute the query embedding and convert to NumPy array
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Use the retriever to get top-k documents
    retrieved_docs = retriever.invoke(query)

    # Get indices of retrieved documents in the embeddings array
    retrieved_indices = []
    for doc in retrieved_docs:
        doc_id = doc.metadata.get('chunk_id')
        if doc_id is not None:
            index = id_to_index.get(str(doc_id))
            if index is not None:
                retrieved_indices.append(index)

    # Visualize the embeddings with the query and retrieved documents
    visualize_embeddings_with_query(
        embeddings_array, documents,
        query_embedding=query_embedding,
        retrieved_indices=retrieved_indices
    )


if __name__ == "__main__":
    main()
