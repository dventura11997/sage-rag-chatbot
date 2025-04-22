# Start-up
# cd rag-backend
# env/scripts/activate
# Python server.py

from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from functions import *

app = Flask(__name__)
app.secret_key = "hello"

CORS(app)

@app.route('/test', methods=['GET'])
def send_email():
    test = 'Hello world'

    return jsonify({"message": f"Test: {test}"}), 200

@app.route('/metadata', methods=['GET'])
def gen_pdf_metadata():
    pdf_metadata_df = ProcessPDFs.extract_data_multiple_pdfs()

    ProcessPDFs.generate_pdf_summaries(pdf_metadata_df)
    

    return jsonify({"message": f"PDF metadata generated in storage account"}), 200

@app.route('/azure', methods=['GET'])
def test_connection():
    ProcessPDFs.ConnectAzure() 
    message = "Returned blob list in console"

    return jsonify({"message": f"{message}"}), 200

# @app.route('/query_response', methods=['POST'])
# def gen_query_response():
#     try:
#         request_body = request.json

#         # Connect to Azure
#         connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
#         print(f"Connection String: {connect_str}")  # Debugging line

#         # Create the BlobServiceClient object
#         blob_service_client = BlobServiceClient.from_connection_string(connect_str)

#         # Get query parameters from JSON Body
#         query = request_body.get('query')
#         company = request_body.get('company')

#         # Get pdf_metadata_df with text from each pdf
#         container_name = 'kaipdfdocs'  
#         storage_acct_name = 'meistrgeus'
#         pdf_folder_path = f"https://{storage_acct_name}.blob.core.windows.net/{container_name}/"
#         pdf_metadata_df = extract_data_multiple_pdfs(pdf_folder_path)
#         print(f"Metadata spreadsheet name: {pdf_metadata_df.head}")

#         # Get pdf_metadata_df which has summaries from Azure storage account
#         #sas_token = "sv=2024-11-04&ss=bfqt&srt=c&sp=rwdlacupiytfx&se=2025-04-29T14:06:48Z&st=2025-04-10T06:06:48Z&spr=https&sig=fYEAWFcat7yZYDH1HaZxhGt60%2FvUDKr6YZrT0qx3bWY%3D"
#         #sas_token = "sv=2024-11-04&ss=bfqt&srt=c&sp=rwdlacupiytfx&se=2025-04-18T14:47:22Z&st=2025-04-11T06:47:22Z&spr=https&sig=sU5%2F3Lt%2BKv2z%2BTnYNYvVPPXkL6mGxj%2Fu5CRojKNCnE4%3D"
#         sas_token = "sp=racwdli&st=2025-04-11T07:09:42Z&se=2025-12-31T15:09:42Z&spr=https&sv=2024-11-04&sr=c&sig=WX%2FmgiTLtenkhAeGEXaaSU3doIE6uCmMILvsjZXPH3c%3D"
#         #pdf_folder_path = r"C:\Users\BD335SR\OneDrive - EY\Documents\DanielVentura\Engagements\MSFT GEN AI\genai-chatbot-asset\rag-backend\pdf_docs"
#         # workspaces folder path
#         #pdf_folder_path = "/workspaces/genai-chatbot-asset/rag-backend/pdf_docs"
#         metadata_sheet_name = "pdf_metadata_df.csv"
#         #print(metadata_sheet_name)
#         pdf_meta_sum_path = f"https://{storage_acct_name}.blob.core.windows.net/{container_name}/{metadata_sheet_name}?{sas_token}"
#         print(f"Path of metadata spreadsheet containing summaries: {pdf_meta_sum_path}")

#         # Read the CSV file from Azure Blob Storage into a DataFrame
#         #pdf_metadata_df = pd.read_csv(pdf_meta_sum)
#         pdf_meta_sum = pd.read_csv(pdf_meta_sum_path)
#         print(f"Reading PDF of metadata which contains summaries: {pdf_meta_sum.head}")

#         #pdf_meta_sum = pd.read_csv(r"C:\Users\BD335SR\OneDrive - EY\Documents\DanielVentura\Engagements\MSFT GEN AI\genai-chatbot-asset\pdf_metadata_df.csv")

#         response, formatted_relevant_pdf_titles = query_response(query, company, pdf_meta_sum, pdf_metadata_df)

#         return jsonify({"message": f"{response}", "relevant_documents": f"Referenced documents: {formatted_relevant_pdf_titles}"}), 200
    
#     except Exception as e:
#         # Catch all unexpected errors
#         return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False)  