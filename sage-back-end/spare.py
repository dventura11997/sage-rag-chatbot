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




def query_response(query, company, pdf_meta_sum, pdf_metadata_df):
    #pdf_folder_path = r"C:\Users\BD335SR\OneDrive - EY\Documents\DanielVentura\Engagements\MSFT GEN AI\genai-chatbot-asset\rag-backend\pdf_docs"
    # workspaces folder path
    # pdf_folder_path = "/workspaces/genai-chatbot-asset/rag-backend/pdf_docs"

    # Call the functions to generate dataframes
    #df_metadata_df = extract_data_multiple_pdfs(pdf_folder_path)
    #pdf_meta_sum = pd.read_csv("C:\Users\BD335SR\OneDrive - EY\Documents\DanielVentura\Engagements\MSFT GEN AI\genai-chatbot-asset\pdf_metadata_df.csv")
    
    context = generate_contextual_paragraph(company)
    print(context)
    #select_relevant_pdfs(query, pdf_meta_sum)

    json_response = select_relevant_pdfs(query, pdf_meta_sum)
    print(json_response)
    
    # Ensure json_response is a dictionary
    if isinstance(json_response, str):
        json_response = json.loads(json_response)
    #print(json_response)

    # Extract the list of relevant PDF titles from the json_response
    relevant_pdf_titles = json_response.get("relevant_pdfs", [])
    print(f"Relevant PDFs: {relevant_pdf_titles}")

    # Merge the two DataFrames on the 'title' column
    merged_df = pd.merge(pdf_metadata_df, pdf_meta_sum[['title', 'summary']], on='title', how='inner')

    # Filter the merged DataFrame to find relevant entries
    relevant_df = merged_df[merged_df['title'].isin(relevant_pdf_titles)]

    # Prepare the chat prompt
    chat_prompt = [
        {"role": "user", "content": f"""Please create a concise and relevant 3-4 sentence response to the user query: "{query}". 
         Use the following context about the company: {context} and the relevant documents.
         Ensure the response is directly related to the query and does not include unnecessary information.
         Ensure the response is textual with full sentences.
         Do not respond to the query as if its an email with any sign-off, title or email signature.
         Please ensure there are no forward or backslashes in the response. Or any new line syntax (such as "\\n" or "\\n1")."""}
    ]

    for index, row in relevant_df.iterrows():
        summary = row['summary']
        title = row['title']
        text = row['text']
        chat_prompt.append({"role": "user", "content": f"Summary: {summary}\nTitle: {title}, Text: {text}"})



    print(f"Responding to user query using: {relevant_pdf_titles} and {context}")
    # Include speech result if speech is enabled  
    messages = chat_prompt  
            
    # Generate the completion  
    completion = client.chat.completions.create(  
        model=deployment,
        messages=messages,
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False
    )

    json_response = json.loads(completion.to_json())
    query_response = json_response['choices'][0]['message']['content'] 

    query_response = query_response.replace('\n', ' ').replace('\\n', ' ').strip()
    # Convert the list to a comma-separated string
    formatted_relevant_pdf_titles = ', '.join(relevant_pdf_titles)

    print(query_response)
    return query_response, formatted_relevant_pdf_titles

#Function here for vector store set-up, leverage it to search
# Function to store DataFrame in ChromaDB
# def store_dataframe_in_chroma(pdf_metadata_df):
#     # Initialize the embeddings model
#     embeddings = OpenAIEmbeddings(openai_api_key=subscription_key)

#     # Create a Chroma vector store
#     vector_store = Chroma(embedding_function=embeddings)

#     # Iterate through the DataFrame and add each summary to the vector store
#     for index, row in pdf_metadata_df.iterrows():
#         title = row['title']
#         summary = row['summary']
        
#         # Add the title and summary to the vector store
#         vector_store.add_texts(texts=[summary], metadatas=[{"title": title}])

#     print("DataFrame stored in ChromaDB successfully.")

# Define variables:

#pdf_folder_path = r"C:\Users\BD335SR\OneDrive - EY\Documents\DanielVentura\Engagements\MSFT GEN AI\genai-chatbot-asset\rag-backend\pdf_docs"
# workspaces folder path
#pdf_folder_path = "/workspaces/genai-chatbot-asset/rag-backend/pdf_docs"

#open_ai_api_key = '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'
# query = 'How do I change my Super to AusSuper?'
# company = 'Australian Super'
#generate_pdf_summaries()
# pdf_metadata_df = extract_data_multiple_pdfs(pdf_folder_path)
# pdf_meta_sum = pd.read_csv("pdf_metadata_df.csv")
# query_response(query, company, pdf_meta_sum, pdf_metadata_df)
#generate_contextual_paragraph(company)