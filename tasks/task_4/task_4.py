from google.auth import credentials, default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.
    The EmbeddingClient class should be capable of initializing an embedding client with specific configurations
    for model name, project, and location. Your task is to implement the __init__ method based on the provided
    parameters. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Steps:
    1. Implement the __init__ method to accept 'model_name', 'project', 'location', and 'key_file_path' parameters.
       These parameters are crucial for setting up the connection to the VertexAIEmbeddings service.
    2. Within the __init__ method, initialize the 'self.client' attribute as an instance of VertexAIEmbeddings
       using the provided parameters and credentials from the service account key file.
    3. Implement the embed_query method to use the embedding client to retrieve embeddings for the given query.

    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.
    - key_file_path: The path to the service account key file for authentication.

    Instructions:
    - Carefully initialize the 'self.client' with VertexAIEmbeddings in the __init__ method using the parameters and credentials.
    - Implement the embed_query method to use the embedding client to retrieve embeddings for the given query.

    Note: The 'embed_documents' method has been provided for you. Focus on correctly initializing the class and implementing the embed_query method.
    """

    def __init__(self, model_name, project, location, key_file_path):
        # Initialize the VertexAIEmbeddings client with the given parameters and service account key file
        self.client = self._initialize_client(model_name, project, location, key_file_path)

    def _initialize_client(self, model_name, project, location, key_file_path):
        # Load the service account key file
        credentials = service_account.Credentials.from_service_account_file(key_file_path)

        # Initialize the VertexAIEmbeddings client with the provided parameters and credentials
        client = VertexAIEmbeddings(model_name=model_name, project=project, location=location, credentials=credentials)
        return client

    def embed_query(self, query):
        """Uses the embedding client to retrieve embeddings for the given query."""
        vectors = self.client.embed_query(query)
        return vectors

    def embed_documents(self, documents):
        """Retrieve embeddings for multiple documents."""
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "radica-ai"
    location = "us-central1"
    key_file_path = "/Users/adigweleo/Downloads/radica-ai-22cfc1454dfc.json"  # Replace with the actual path to your service account key file

    embedding_client = EmbeddingClient(model_name, project, location, key_file_path)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        print(vectors)
        print("Successfully used the embedding client!")
    
