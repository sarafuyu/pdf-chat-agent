import os
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pdf_helpers.helper_vsdb import load_split_pdf
from pdf_helpers.helper_llm import embeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(filename='output_log.txt', level=logging.INFO, format='%(message)s')

# Specify the persist directory
file_path = "data/pdfs/Animal_facts.pdf"
persist_directory = "./data/chroma"

# Function to handle errors during rmtree
def remove_readonly(func, path, excinfo):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Remove existing data in the persist directory
if os.path.exists(persist_directory):
    try:
        # Close any open connections to the Chroma database
        db_temp = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        db_temp._client.close()
        del db_temp

        # Now, remove the directory
        shutil.rmtree(persist_directory, onerror=remove_readonly)
    except Exception as e:
        print(f"Error removing persist directory: {e}")

# Recreate the persist directory
os.makedirs(persist_directory, exist_ok=True)

# Split the PDF document into chunks
doc = load_split_pdf(file_path)

# Remove duplicate chunks
def remove_duplicate_chunks(docs):
    unique_docs = []
    seen_contents = set()
    for d in docs:
        content = d.page_content.strip()
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append(d)
    return unique_docs

doc = remove_duplicate_chunks(doc)

# Prepare data for the vector store
documents = []
metadatas = []
ids = []

for idx, doc_chunk in enumerate(doc):
    # Add a unique ID to each document's metadata
    doc_chunk.metadata['chunk_id'] = idx
    documents.append(doc_chunk.page_content)
    metadatas.append(doc_chunk.metadata)
    ids.append(str(idx))

# Create the Chroma vector store
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

db.add_texts(texts=documents, metadatas=metadatas, ids=ids)

# Define the combined inspection and visualization function
def inspect_and_visualize_chroma(db):
    # Get collection information
    collection = db._collection
    logging.info("Collection Name: %s", collection.name)
    logging.info("Number of Documents: %d", collection.count())
    logging.info("Collection Metadata: %s", collection.metadata)

    # Retrieve embeddings, metadatas, documents, and ids
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
        logging.info(f"Document Content: {documents[idx][:100]}...")
        logging.info("-----\n")

    # Build a mapping from ids to indices
    id_to_index = {ids[idx]: idx for idx in range(len(ids))}

    # Convert embeddings to a NumPy array
    embeddings_array = np.array(embeddings_list)

    # Visualize the embeddings without any query
    visualize_embeddings_with_query(embeddings_array, documents)

    return embeddings_array, documents, id_to_index

# Visualization function
def visualize_embeddings_with_query(embeddings, documents, query_embedding=None, retrieved_indices=None):
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))

    # Plot all document embeddings
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], label='Documents', c='blue')

    # Annotate document points
    for i, doc in enumerate(documents):
        plt.annotate(f'Doc {i}', (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    # Plot query embedding if provided
    if query_embedding is not None:
        query_embedding_2d = pca.transform(query_embedding)
        plt.scatter(query_embedding_2d[0, 0], query_embedding_2d[0, 1], label='Query', c='red', marker='X', s=100)
        plt.annotate('Query', (query_embedding_2d[0, 0], query_embedding_2d[0, 1]))

    # Highlight retrieved documents if provided
    if retrieved_indices:
        plt.scatter(embeddings_2d[retrieved_indices, 0], embeddings_2d[retrieved_indices, 1],
                    edgecolors='green', facecolors='none', s=200, linewidths=2, label='Retrieved Docs')

    plt.title("Embeddings Visualization with Query")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

# Call the combined inspection and visualization function
embeddings_array, documents, id_to_index = inspect_and_visualize_chroma(db)

# Create the retriever
def create_retriever(db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

retriever = create_retriever(db)

# Define a query
query = "How long is a giraffe's tongue?"

# Compute the query embedding and convert to NumPy array
query_embedding = embeddings.embed_query(query)
query_embedding = np.array(query_embedding).reshape(1, -1)

# Use the retriever to get top-k documents
retrieved_docs = retriever.get_relevant_documents(query)

# Get indices of retrieved documents in the embeddings array
retrieved_indices = []
for doc in retrieved_docs:
    doc_id = doc.metadata.get('chunk_id')
    if doc_id is not None:
        index = id_to_index.get(str(doc_id))
        if index is not None:
            retrieved_indices.append(index)

# Visualize the embeddings with the query and retrieved documents
visualize_embeddings_with_query(embeddings_array, documents, query_embedding=query_embedding, retrieved_indices=retrieved_indices)
