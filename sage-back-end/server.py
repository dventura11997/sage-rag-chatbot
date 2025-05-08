# Start-up
# cd rag-backend
# env/scripts/activate
# Python server.py

from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import logging
import gc

# Configure logging so it shows up in my Render dashboard
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "hello"

CORS(app)

@app.route('/test', methods=['GET'])
def test():
    relevant_titles = "['What does Public mean?', 'TSIA April 2025', 'Financial Services Guide']"

    logger.info(f"Relevant documents: {relevant_titles}")
    relevant_titles = ', '.join(relevant_titles)
    logger.info(f"Relevant documents for message: {relevant_titles}")

    return jsonify({"message": f"Test: {test}"}), 200

@app.route('/azure', methods=['GET'])
def test_connection():
    try:
        # Import only when needed
        from functions import ProcessPDFs
        
        blobs = ProcessPDFs.ConnectAzure() 
        message = f"Successfully listed {len(blobs)} blobs"
        logger.info(message)
        
        # Force garbage collection
        gc.collect()

        return jsonify({"message": message}), 200

    except Exception as e:
        logger.error(f"Azure connection error: {str(e)}")
        return jsonify({"error": f"Connection error: {str(e)}"}), 500

@app.route('/gen_meta', methods=['GET'])
def gen_pdf_metadata():
    try:
        # Import only when needed
        from functions import ProcessPDFs
        
        logger.info("Starting PDF metadata extraction")
        pdf_metadata_df = ProcessPDFs.extract_data_multiple_pdfs()
        logger.info(f"Extracted metadata from {len(pdf_metadata_df)} PDFs")

        logger.info("Storing data to vector store")
        ProcessPDFs.store_to_vector_store(pdf_metadata_df, index_path="faiss_index", metadata_path="faiss_metadata.pkl")
        logger.info("Vector store creation complete")
        
        # Force garbage collection
        del pdf_metadata_df
        gc.collect()

        return jsonify({"message": "PDF metadata generated in storage account"}), 200
    except Exception as e:
        logger.error(f"Metadata generation error: {str(e)}")
        return jsonify({"error": f"Error generating metadata: {str(e)}"}), 500

@app.route('/generate_company_context', methods=['POST'])
def response_helpers():
    try:
        # Import only when needed
        from functions import ResponseHelpers
        
        request_body = request.json
        # Get query parameters from JSON Body
        company = request_body.get('company')
        
        logger.info(f"Generating contextual paragraph for company: {company}")
        contextual_paragraph = ResponseHelpers.generate_contextual_paragraph(company)
        logger.info(f"Contextual paragraph generated for {company}")   
        
        # Force garbage collection
        gc.collect()

        return jsonify(contextual_paragraph), 200
    except Exception as e:
        logger.error(f"Helper error: {str(e)}")
        return jsonify({"error": f"Error in helpers: {str(e)}"}), 500

@app.route('/chat_response', methods=['POST'])
def gen_query_response():
    try:
        # Import only when needed
        from functions import ChatResponse
        
        request_body = request.json

        # Get query parameters from JSON Body
        query = request_body.get('query')
        company = request_body.get('company')
        
        logger.info(f"Processing query for company {company}: {query}")
        response, relevant_titles = ChatResponse.query_response(query, company)
        logger.info(f"Generated response with relevant titles: {relevant_titles}")
        
        # Force garbage collection
        gc.collect()

        return jsonify({"message": f"{response}", "relevant_documents": f"Documents used for response: {relevant_titles}"}), 200
    
    except Exception as e:
        # Log the full traceback
        import traceback
        logger.error(f"Query response error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
@app.route('/chat_response_basic', methods=['POST'])
def gen_query_response_basic():
    try:
        from azure.storage.blob import BlobServiceClient
        import pandas as pd
        from io import StringIO
        from functions import ResponseHelpers, ProcessPDFs, ChatResponse

        request_body = request.json

        # Get query parameters from JSON Body
        query = request_body.get('query')

        # Azure Storage settings
        container_name = 'sage-pdf-docs'
        storage_acct_name = 'devprojectsdb'
        file_name = 'pdf_metadata_df.csv'
        connect_str = "DefaultEndpointsProtocol=https;AccountName=devprojectsdb;AccountKey=vl7x6XrnS8Esycm9fFsXO/biKfHRyKWRXYuI9WcRb1r1xiMlRUQcipmsvUruJu3K5VHY1NjMbdyi+ASt1FaEhA==;EndpointSuffix=core.windows.net"

        # Connect to the blob
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)

        pdf_folder_path = f"https://{storage_acct_name}.blob.core.windows.net/{container_name}/"

        # Download the blob content
        download_stream = blob_client.download_blob()
        csv_content = download_stream.readall().decode('utf-8')

        logger.info(f"Extracting data from PDFs")
        pdf_metadata_df = ProcessPDFs.extract_data_multiple_pdfs()
        logger.info(f"Extracted data from PDFs")
        # Convert CSV content to DataFrame
        logger.info(f"Reading summary CSV from storage account")
        pdf_metadata_sum = pd.read_csv(StringIO(csv_content))
        logger.info(f"Read summary CSV from storage account")

        logger.info(f"Selecting Relevant PDFs")
        relevant_pdfs = ResponseHelpers.select_relevant_pdfs(query, pdf_metadata_sum)
        logger.info(f"Selected Relevant PDFs: {relevant_pdfs}")

        logger.info(f"Generating Query Response")
        query_response, formatted_relevant_pdf_titles = ChatResponse.query_response_basic(query, relevant_pdfs, pdf_metadata_sum, pdf_metadata_df)
        logger.info(f"Query response generated with length: {len(query_response)}")

        return jsonify({"message": f"{query_response}", "relevant_documents": f"Documents used for response: {formatted_relevant_pdf_titles}"}), 200
    
    except Exception as e:
        # Log the full traceback
        import traceback
        logger.error(f"Query response error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    # Use threaded=False to reduce memory usage
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=False) 