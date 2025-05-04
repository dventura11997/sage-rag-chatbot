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

@app.route('/metadata', methods=['GET'])
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

@app.route('/helpers', methods=['POST'])
def response_helpers():
    try:
        # Import only when needed
        from functions import ResponseHelpers
        
        request_body = request.json
        # Get query parameters from JSON Body
        company = request_body.get('company')
        
        logger.info(f"Generating contextual paragraph for company: {company}")
        contextual_paragraph = ResponseHelpers.generate_contextual_paragraph(company)   
        
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
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    # Use threaded=False to reduce memory usage
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=False) 