import openai
from dotenv import load_dotenv
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import csv
import csv

# Load .env file
load_dotenv()

openai.log = False  # Set to "debug" if needed - and include quote marks

openai.api_type = "azure"
openai.api_base = 'https://bc-api-management-uksouth.azure-api.net'
openai.api_version = "2023-03-15-preview"

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Pinecone key
if "pinecone_api_key" in os.environ:
    pinecone_key = os.getenv("pinecone_api_key")
    pinecone_env = os.getenv("Pinecone_API_ENV")
    pass

# Connect to Pinecone
pinecone.init(api_key=pinecone_key, environment=pinecone_env)

# Initialize the embeddings from OpenAI for embedding the document chunks
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key, chunk_size=1, engine="text-embedding-ada-002")

# Check active Pinecone indexes
active_indexes = pinecone.list_indexes()

# Put in the name of your Pinecone index here
index_name = "example-index"
print(active_indexes)

# Initialize the document search using the existing index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Invoke ChatOpenAI --turbo model
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(streaming=False, temperature=0.5, max_tokens=512, openai_api_key=openai.api_key, engine ='gpt-35-turbo')

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=retriever, return_source_documents=True)

def result_query(query):
    embeddings =  OpenAIEmbeddings(openai_api_key=openai.api_key,engine="text-embedding-ada-002")
    query_result = embeddings.embed_query(query)
    index = pinecone.Index("example-index")
    answer = index.query(
      vector = list(query_result),
      top_k=1,
      include_metadata=True
        )
    #filtering to the specific chunk
    for i in range(0,len(answer['matches'])):
        
        ans_text = answer['matches'][0]['metadata']['chunk_text']
        
    return ans_text

# Create or open a CSV file for storing queries and answers
csv_file = "queries_and_answers.csv"
with open(csv_file, mode="w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Query","Detailed Answer"])  # Write header row

while True:
    # Prompt the user for a query
    user_input = input("Enter your query (type 'quit' to exit): ")
    
    if user_input.lower() == "quit":
        break  # Exit the loop if the user types 'quit'
    
    # Save the query to the CSV file
    with open(csv_file, mode="a", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([user_input, "", ""])  # Save the query with empty answers for now

    # Get the answer
    answer = result_query(user_input)

    # Integrate the provided code for generating a detailed response
    sudo_answer_response = openai.ChatCompletion.create(
        engine="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"You are CareerDevBot. From the data given, fetch the answer point to point to the competencies associated with the user's role. Provide a detailed response for each competency. Do not add any content on your own, apart from the data given below.\n\n{answer}.\n\nGive the answer in a paragraph in a maximum of 10-12 sentences"
            },
            {
                "role": "user", 
                "content": user_input  # Use the user's input as the query
            }
        ]
    )

    # Extract the detailed answer from the response
    detailed_answer = sudo_answer_response['choices'][0]['message']['content']

    # Save both answers to the CSV file
    with open(csv_file, mode="a", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["", detailed_answer])

    # Print the detailed answer to the user
    print(detailed_answer)

print("Goodbye!")
