# app.py
from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings,Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Ollama with Deepseek R1 1.5B
Settings.llm = Ollama(model="deepseek-coder:1.3b", base_url="http://localhost:11434")
Settings.embed_model = OllamaEmbedding(model_name="deepseek-coder:1.3b")
Settings.context_window = 4096 

index = None

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
        
        # Extract text from PDF
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        
        # NewCode
        from PIL import Image
        import pytesseract
        import fitz

        def extract_text_from_image_pdf(file_path):
            text = ""
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img)
            return text
        
      
                
        # Create vector index
        index = VectorStoreIndex.from_documents(
            [Document(text=text)],
            show_progress=True
        )
        
        return jsonify({'message': 'File uploaded successfully'})
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global index
    data = request.json
    question = data.get('question')
    
    if not index:
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    
    return jsonify({'answer': str(response)})

if __name__ == '__main__':
    app.run(debug=True)