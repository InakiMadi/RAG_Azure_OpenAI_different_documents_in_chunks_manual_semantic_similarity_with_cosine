## RAG Azure OpenAI API with different documents separated in chunks, with a manual knowledge basis in JSON files, and semantic similarity using cosine

First of all, install all requirements.

> pip install -r requirements.txt

This is a simple RAG example using Azure OpenAI API.

### Fixed issues and what I learnt:

- There are Resources (usually in portal.azure.com) and Deployments (aoi.azure.com or Azure AI Foundry).
- In a given Resource you can deploy several models. In this case, we need a GPT and an Embedding model.
- The default Resource for me was Spain Central. Issue: you can't deploy gpt-35-turbo in Spain Central.
- I tried Sweden Central as Resource as well, but they told me for gpt-35-turbo that I didn't have more quota.
- In the end, the solution I came up with was to create a EastUS Resource.
- In that resource, I deployed gpt-35-turbo and text-embedding-3-small.
- Running the code, I had the error 400 - chatCompletions isn't a function from the model I deployed.
- I tried different versions of gpt-35-turbo, which has a chatCompletions in OpenAI, with no luck.
- Fix: Deploy gpt-4.1 instead of gpt-35-turbo.
- Note: Although going to portal.azure.com is the initial idea, Azure AI Foundry is really the place to select and deploy models.

### Details of the project:

1. An Azure OpenAI client (API key needed for the user, not uploaded for safety reasons), refactored in a clean OpenAIClient class.
2. RAG. Retrieves information from different documents and answers queries with respect to them.
    - Texts: Extract texts from different PDFs using PdfReader.
    - Chunks: Divide each text in chunks to simulate larger documents for knowledge basis.
    - chatCompletions: Azure OpenAi with gpt-4.1.
    - Embeddings: Azure OpenAI with text-embedding-3-small.
    - Semantic similarity: Cosine between embeddings from knowledge basis and query.