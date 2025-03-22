from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Ollama
Settings.llm = Ollama(model="deepseek-coder:1.3b", base_url="http://localhost:11434")
Settings.embed_model = OllamaEmbedding(model_name="deepseek-coder:1.3b")
Settings.context_window = 4096

index = None

# Function to extract text from image-based PDFs
def extract_text_from_image_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global index
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extract text from PDF (handles both normal and image-based PDFs)
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                extracted_text = page.get_text("text")
                if extracted_text:
                    text += extracted_text
                else:
                    text += extract_text_from_image_pdf(file_path)

        if not text.strip():
            return jsonify({'error': 'No readable text found in the PDF'}), 400

        # Create vector index
        index = VectorStoreIndex.from_documents([Document(text=text)], show_progress=True)
        
        return jsonify({'message': 'File uploaded successfully'})
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global index
    data = request.json
    question = data.get('question')

    if not index:
        return jsonify({'error': 'No PDF has been processed yet. Please upload a file first.'}), 400

    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        return jsonify({'answer': str(response)})
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
