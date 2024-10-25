from langchain_community.vectorstores import Chroma
from pdf_helpers.helper_vsdb import load_split_pdf
from pdf_helpers.helper_llm import embeddings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Specify the persist directory
file_path = "data/pdfs/Animal_facts.pdf"
# Split document into chunks
doc = load_split_pdf(file_path)

# Vector Store db directory
persist_directory = "./data/chroma"
# Load the Chroma vector store
db = Chroma.from_documents(documents=doc, embedding=embeddings, persist_directory=persist_directory)

# Define the combined inspection function
def inspect_chroma_collection(db):
    # Get collection information
    collection = db._collection
    print("Collection Name:", collection.name)
    print("Number of Documents:", collection.count())
    print("Collection Metadata:", collection.metadata)
    
    # Retrieve embeddings, metadatas, and documents
    results = collection.get(include=['embeddings', 'metadatas', 'documents'])
    embeddings_list = results['embeddings']
    metadatas = results['metadatas']
    documents = results['documents']
    ids = results['ids']  # 'ids' are always included by default
    
    print("\nDocuments in the Collection:")
    for idx in range(len(ids)):
        print(f"Document ID: {ids[idx]}")
        print(f"Embedding: {embeddings_list[idx]}")
        print(f"Metadata: {metadatas[idx]}")
        print(f"Document Content: {documents[idx]}")
        print("-----\n")
    
    return np.array(embeddings_list), documents

# Visualization function
def visualize_embeddings(embeddings, documents):
    # Reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    # Annotate the points with document IDs
    for i, doc in enumerate(documents):
        plt.annotate(f"Doc {i}", (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title("Embeddings Visualization using t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# Get embedding metadata from database
embedings, documents = inspect_chroma_collection(db)

# Plot reduced 2D embeddings
visualize_embeddings(embedings, documents)