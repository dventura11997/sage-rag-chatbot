import os
import json
import io
import pandas as pd
from PyPDF2 import PdfReader
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from io import StringIO
import faiss
import openai
import pickle

class ProcessPDFs:
    def ConnectAzure():
        connect_str = "DefaultEndpointsProtocol=https;AccountName=devprojectsdb;AccountKey=vl7x6XrnS8Esycm9fFsXO/biKfHRyKWRXYuI9WcRb1r1xiMlRUQcipmsvUruJu3K5VHY1NjMbdyi+ASt1FaEhA==;EndpointSuffix=core.windows.net"
        #print(f"Connection String: {connect_str}")  # Debugging line

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # Specify the container name
        container_name = 'sage-pdf-docs'  # Replace with your actual container name
        container_client = blob_service_client.get_container_client(container_name)

        # List the blobs in the container
        blob_list = container_client.list_blobs()

        # List the blobs in the container
        blob_list = container_client.list_blobs()
        for blob in blob_list:
             print("\t" + blob.name)
    
    # Function to read PDF files - AZURE VERSION
    def extract_pdf_data(pdf_file):
        pdf_reader = PdfReader(pdf_file)
        metadata = pdf_reader.metadata
        num_pages = len(pdf_reader.pages)
        
        #print(metadata.title)  # Print the title for debugging

        # Extract text from all pages
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()  # Call extract_text() as a function

        return {
            "title": metadata.title,
            "text": text,
            "author": metadata.author,
            "creator": metadata.creator,
            "producer": metadata.producer,
            "subject": metadata.subject,
            "creation_date": metadata.get('/CreationDate'),
            "modification_date": metadata.get('/ModDate'),
            "keywords": metadata.get('/Keywords'),
            "num_pages": num_pages,
        }
    
    def extract_data_multiple_pdfs():
        # Initialize a list to hold the metadata
        pdf_metadata_list = []

        # Connect to Azure Blob Storage
        connect_str = "DefaultEndpointsProtocol=https;AccountName=devprojectsdb;AccountKey=vl7x6XrnS8Esycm9fFsXO/biKfHRyKWRXYuI9WcRb1r1xiMlRUQcipmsvUruJu3K5VHY1NjMbdyi+ASt1FaEhA==;EndpointSuffix=core.windows.net"
        print(connect_str)
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Get the container client
        container_name = 'sage-pdf-docs'  # Replace with your actual container name
        container_client = blob_service_client.get_container_client(container_name)

        # List blobs in the container
        blob_list = container_client.list_blobs()
        #print(blob_list)

        for blob in blob_list:
            if blob.name.endswith('.pdf'):
                print(f"Processing metadata for blob: {blob.name}")  # Print the blob name for debugging
                # Download the blob content
                blob_client = container_client.get_blob_client(blob)
                pdf_data = blob_client.download_blob().readall()

                # Use io.BytesIO to read the PDF data
                pdf_metadata = ProcessPDFs.extract_pdf_data(io.BytesIO(pdf_data))
                pdf_metadata_list.append(pdf_metadata)

        # Create a DataFrame from the metadata list
        pdf_metadata_df = pd.DataFrame(pdf_metadata_list)
        

        return pdf_metadata_df


    def generate_chunks(text: str, chunk_size: int = 10000, chunk_overlap: int = 200):
        """Chunk the text to stay within the token limit for OpenAI models."""

        # Initialize the tiktoken encoder for the correct model
        encoder = tiktoken.get_encoding("cl100k_base")  # Use the same encoding as OpenAI's model
        # Define the separators and chunking behavior
        separators = ["\n\n", "\n", " ", ""]
        
        # Initialize the RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=lambda text: len(encoder.encode(text))  # Token count function using tiktoken
        )
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        return chunks
    
    # Function to generate PDF summaries
    def generate_pdf_summaries(pdf_metadata_df):
        # Initialize the OpenAI client (you can also set OPENAI_API_KEY in env vars)
        client = openai.OpenAI(api_key="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")

        for idx, row in pdf_metadata_df.iterrows():
            

            # Generate chunks for the current row's text
            chunks = ProcessPDFs.generate_chunks(row['text'])
        
            # Initialize a list to collect summaries for all chunks
            summaries = []

            for chunk in chunks:
                # Prepare the chat prompt
                chat_prompt = [
                    {"role": "user", "content": f"Can you please provide a concise summary of the following text:\n\n{chunk}"}
                ]

                print(f"Processing summary with unchunked prompt for {row['title']}")
                # Include speech result if speech is enabled  
                messages = chat_prompt  
                
                # Generate the completion  
                completion = client.chat.completions.create(  
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=800,  
                    temperature=0.7,  
                    top_p=0.95,  
                    frequency_penalty=0,  
                    presence_penalty=0,
                    stop=None,  
                    stream=False
                )
                
                #print(completion.to_json())
                response = json.loads(completion.to_json())
                #print(response)
                # Extract the summary from the completion response
                summary = response['choices'][0]['message']['content']
                summaries.append(summary)
            
            # Combine all chunk summaries
            full_summary = " ".join(summaries)
            print(full_summary)
        
            # Store the final summary in the DataFrame
            pdf_metadata_df.at[idx, 'summary'] = full_summary

        print(pdf_metadata_df.head())
        pdf_metadata_df.drop(['text'], axis=1, inplace=True)

        # Save the file to a storage container
        container_name = 'sage-pdf-docs'  
        storage_acct_name = 'devprojectsdb'
        file_name = 'pdf_metadata_df.csv'
        connect_str = "DefaultEndpointsProtocol=https;AccountName=devprojectsdb;AccountKey=vl7x6XrnS8Esycm9fFsXO/biKfHRyKWRXYuI9WcRb1r1xiMlRUQcipmsvUruJu3K5VHY1NjMbdyi+ASt1FaEhA==;EndpointSuffix=core.windows.net"

        # Convert DataFrame to CSV in memory
        csv_buffer = StringIO()
        pdf_metadata_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Connect and upload
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)

        blob_client.upload_blob(csv_data, overwrite=True)
    
    
        return pdf_metadata_df
    
    def store_to_vector_store(df, index_path="faiss_index", metadata_path="faiss_metadata.pkl"):
        # Load model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Get text column and encode
        texts = df["text"].fillna("").tolist()
        embeddings = model.encode(texts, convert_to_numpy=True)

        # Create FAISS index
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Save index to disk
        faiss.write_index(index, index_path)

        # Save metadata separately (e.g., title, author)
        metadata = df.drop(columns=["text"]).to_dict(orient="records")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"Stored {len(texts)} PDFs in vector store.")