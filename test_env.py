from dotenv import load_dotenv
import os

# This simple script shows the content of the .env file to see the values of the environment variables
load_dotenv()
print("Loaded API KEY:", os.getenv("GRAPH_API_KEY"))

load_dotenv(dotenv_path=".env.costs") 

print("CONST:", os.getenv("SMALL_INDEXER"))
print("CONST:", os.getenv("MEDIUM_INDEXER"))