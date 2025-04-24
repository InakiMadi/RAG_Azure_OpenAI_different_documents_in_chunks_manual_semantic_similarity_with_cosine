import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import json
from src.azure_openai_client import AzureOpenAIClient
from typing import List, Dict, Tuple, Any


class CosineRAG:
    def __init__(self, client: AzureOpenAIClient):
        self.client_AzureOpenAIClient = client

        # In-memory knowledge base (simulating a vector database)
        self.knowledge_base = []
        self.embeddings = []

    @staticmethod
    def get_documents(input_files: List[str]):
        documents = []
        for file in input_files:
            text = CosineRAG.extract_cv(file)
            chunks = CosineRAG.chunk_text(text)

            for index, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {"source": f"{file}, chunk nº: {index}."}
                })
        return documents

    def add_to_knowledge_base_and_embeddings(self, input_files: List[str]):
        documents = CosineRAG.get_documents(input_files)
        for doc in documents:
            embedding = self.client_AzureOpenAIClient.get_embedding(doc["text"])
            new_knowledge = {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "embedding": embedding
            }
            self.knowledge_base.append(new_knowledge)
            self.embeddings.append(embedding)

    # Sorts and returns indices in descendant.
    @staticmethod
    def get_top_indices_and_values(similarities, top_k: int = 5):
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similar_values = [similarities[index] for index in top_indices]
        return top_indices, top_similar_values

    # Query the RAG system with a question. Returns tuple of (retrieved documents, generated answer).
    def query(self, question: str, top_k: int = 5) -> tuple[str, List[List], List[float]]:
        # Get embedding for the query
        question_embedding_as_column = np.array(self.client_AzureOpenAIClient.get_embedding(question))
        question_embedding = question_embedding_as_column.reshape(1, -1)

        # Calculate similarities (simple cosine similarity) between question and documents.
        doc_embeddings = np.array(self.embeddings)
        similarities = cosine_similarity(question_embedding, doc_embeddings)[0]

        # Get top_k most similar documents
        top_indices, top_similar_values = CosineRAG.get_top_indices_and_values(similarities, top_k)
        retrieved_docs = [self.knowledge_base[index] for index in top_indices]

        # Generate answer using OpenAI
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])
        prompt = f"""Answer the question based on the context below. If you don't know the answer, say you don't know.

        Context: {context}

        Question: {question}

        Answer:"""

        answer = self.client_AzureOpenAIClient.query(prompt)

        return answer, retrieved_docs, top_similar_values

    # Save knowledge base to a JSON file.
    def save_knowledge_base(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.knowledge_base, f)

    # Load knowledge base from a JSON file.
    def load_knowledge_base(self, file_path: str):
        with open(file_path, 'r') as f:
            self.knowledge_base = json.load(f)
            self.embeddings = [doc["embedding"] for doc in self.knowledge_base]

    @staticmethod
    def extract_cv(cv_path: str) -> str:
        reader = PdfReader(cv_path)
        return " ".join(page.extract_text() for page in reader.pages)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 100) -> List[
        str]:  # Short chunk_size to simulate bigger texts (CVs are usually short).
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
