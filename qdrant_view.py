from qdrant_client import QdrantClient
from qdrant_client.models import FilterSelector, Filter
# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Specify your collection name
collection_name = "ps04-merged"

# Retrieve points (vectors) from the collection
response = client.scroll(
    collection_name=collection_name,
    limit=10,  # Number of points to retrieve
)
# client.delete_collection(collection_name=collection_name)

# Print the retrieved points
for point in response[0]:
    print(f"ID: {point.id}, Vector: {point.vector}, Payload: {point.payload}")
