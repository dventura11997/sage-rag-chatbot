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
def test():
    relevant_titles = "['What does Public mean?', 'TSIA April 2025', 'Financial Services Guide']"

    print(f"Relevant documents: {relevant_titles}")
    relevant_titles = ', '.join(relevant_titles)
    print(f"Relevant documents for message: {relevant_titles}")

    return jsonify({"message": f"Test: {test}"}), 200

@app.route('/azure', methods=['GET'])
def test_connection():
    ProcessPDFs.ConnectAzure() 
    message = "Returned blob list in console"

    return jsonify({"message": f"{message}"}), 200

@app.route('/metadata', methods=['GET'])
def gen_pdf_metadata():
    pdf_metadata_df = ProcessPDFs.extract_data_multiple_pdfs()

    #ProcessPDFs.generate_pdf_summaries(pdf_metadata_df)
    ProcessPDFs.store_to_vector_store(pdf_metadata_df, index_path="faiss_index", metadata_path="faiss_metadata.pkl")

    

    return jsonify({"message": f"PDF metadata generated in storage account"}), 200

@app.route('/helpers', methods=['POST'])
def response_helpers():
    request_body = request.json
    # Get query parameters from JSON Body
    company = request_body.get('company')

    contextual_paragraph = ResponseHelpers.generate_contextual_paragraph(company)   

    return jsonify(contextual_paragraph), 200

@app.route('/chat_response', methods=['POST'])
def gen_query_response():
    try:
        request_body = request.json

        # Get query parameters from JSON Body
        query = request_body.get('query')
        company = request_body.get('company')


        response, relevant_titles = ChatResponse.query_response(query, company)
        # response = "Dummy response"
        # relevant_titles = "Dummy titles"

        return jsonify({"message": f"{response}", "relevant_documents": f"Documents used for response: {relevant_titles}"}), 200
    
    except Exception as e:
        # Catch all unexpected errors
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=False)  