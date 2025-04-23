#Start-up: 
# cd "C:\Users\BD335SR\OneDrive - EY\Documents\DanielVentura\Engagements\MSFT GEN AI\genai-chatbot-asset"
# env/scripts/activate
# Python rag_model_msft.py
# Document set source: https://www.australiansuper.com/tools-and-advice/forms-and-fact-sheets
# Langchain dependencies: tiktoken, langchain-community

import json
import os
from pypdf import PdfReader
from openai import AzureOpenAI  
import pandas as pd
from azure.storage.blob import BlobServiceClient
import io




def select_relevant_pdfs(query):
    # Connect OpenAI client:
    client = openai.OpenAI(api_key="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")

    # Read metadata file in blob storage:
    connect_str = "DefaultEndpointsProtocol=https;AccountName=devprojectsdb;AccountKey=vl7x6XrnS8Esycm9fFsXO/biKfHRyKWRXYuI9WcRb1r1xiMlRUQcipmsvUruJu3K5VHY1NjMbdyi+ASt1FaEhA==;EndpointSuffix=core.windows.net"
    container_name = "sage-pdf-docs"
    blob_name = "pdf_metadata_df.csv"

    # Connect to blob
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Download and read into DataFrame
    download_stream = blob_client.download_blob()
    csv_data = download_stream.content_as_text()
    pdf_metadata_df = pd.read_csv(StringIO(csv_data))

    # Prepare the chat prompt
    chat_prompt = []

    # Iterate over each row in the dataframe to get the specific summary and title
    for index, row in pdf_metadata_df.iterrows():
        summary = row['summary']
        title = row['title']
        chat_prompt.append({"role": "user", "content": f"Summary: {summary}\nTitle: {title}"})

    # Add the user query to the prompt
    chat_prompt.append({"role": "user", "content": 
                        f"""Based on the above summaries, return any titles which may be relevant to this user query: "{query}". 

                            Strictly respond ONLY with a valid array object in this format:
                                ["Title 1", "Title 2", ...]
                            Do not include any commentary, explanation, or markdown — only the raw array.
                            Limit the response to between 1-3 of the three most relevant titles.
                    """})
                                                                                                                        
    messages = chat_prompt 
    print(f"Selecting the relevant documents based on the summary")
        
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
    print(response)
    # Extract the summary from the completion response
    relevant_pdfs = response['choices'][0]['message']['content'].strip()
    print(relevant_pdfs)
        
    return relevant_pdfs

