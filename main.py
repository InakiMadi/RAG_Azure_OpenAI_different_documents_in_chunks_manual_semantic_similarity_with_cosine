from src.azure_openai_client import AzureOpenAIClient
from src.cosine_rag import CosineRAG
from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    client_context = ("Can only speak in English. Do NOT ask for more input at the end. "
                      "Explain your answer thoroughly. "
                      "Answer every question. "
                      "No more than 100 words.")

    ai = AzureOpenAIClient(client_context=client_context)

    # print(ai.query("Hola qué tal"))

    rag = CosineRAG(ai)

    input_path = "input"
    input_files = [input_path + "/" + file for file in listdir(input_path) if isfile(join(input_path, file))]

    # Add input documents to the knowledge base
    rag.add_to_knowledge_base_and_embeddings(input_files)

    # Query the system
    answer, retrieved_docs, similar_values = rag.query(
        # "Does he know Python?"
        # "What is supervised learning?"
        # "What is your opinion on unsupervised learning?"
        "Is he a good candidate for a Junior AI Engineer?"
    )

    print("Generated Answer:")
    print(answer)

    print("\n\nRetrieved Documents:")
    for index, doc in enumerate(retrieved_docs):
        doc_text = doc['text'].replace('\n', ' ')
        print(
            f"- \"{doc_text}\"\n\t-- Source: {doc['metadata']['source']}\n\t-- Similarity value: {similar_values[index]}.")
    print(f"\nDifference of similarity between top 2: {similar_values[0] - similar_values[1]}.")

    # Save knowledge base for later use
    rag.save_knowledge_base("knowledge_base.json")
