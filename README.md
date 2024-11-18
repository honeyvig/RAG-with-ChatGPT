# RAG-with-ChatGPT
To build a small Retrieval-Augmented Generation (RAG) system using ChatGPT with your desired content (e.g., AI Act and ISO standards in [PDF](https://artificialintelligenceact.eu/fr/ai-act-explorer/)), we will break the project down into a few key components:

    Extracting Content from PDF: Use Python to extract text from the provided PDF files (e.g., AI Act and ISO standards).
    Building the RAG system: We will leverage ChatGPT to retrieve relevant information from the content and generate answers.
    Setting up a Web Interface: A simple web page where users can interact with the system.
    Adding Content Dynamically: Ensure that new content can be easily added to the system for future updates.

Here’s how we’ll proceed step by step.
Step 1: Extract Content from PDF

We'll use the PyMuPDF library (fitz) to extract text from the PDFs. This is an easy-to-use library that allows us to extract text and structure it for later use in the RAG system.

Install the necessary libraries:

pip install PyMuPDF flask openai

Here’s an example of extracting text from a PDF using PyMuPDF:

import fitz  # PyMuPDF

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Example usage:
pdf_path = 'path_to_your_iso_standard.pdf'
pdf_text = extract_pdf_text(pdf_path)
print(pdf_text[:500])  # Print the first 500 characters to check

This function will return the entire text of the PDF. Once you extract the content from both the AI Act and ISO Standards, you'll want to store that text for later querying.
Step 2: Setting Up the RAG System
Retrieval:

You can use a simple TF-IDF (Term Frequency-Inverse Document Frequency) or Embeddings based approach for retrieving relevant content. OpenAI embeddings and ChatGPT can be used to fetch the most relevant passage from the content.

For simplicity, we’ll start with a vector search approach. The openai library will be used to generate embeddings.

pip install openai

Here’s an example code to get embeddings for the PDF content using OpenAI's API:

import openai
import numpy as np

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def get_embeddings(text):
    """Get embeddings for a text using OpenAI's API."""
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array([embedding['embedding'] for embedding in response['data']])

# Example usage:
text_embeddings = get_embeddings(pdf_text)
print(text_embeddings[:5])  # Print first 5 embeddings for review

This code takes the text of the extracted PDF and converts it into embeddings. For now, we'll keep it simple by converting the entire document into embeddings.
RAG System:

Once you have the embeddings for the content, you’ll need to create a method to retrieve the most relevant passages based on user input and generate a response using ChatGPT. Here’s how you can integrate this:

def retrieve_relevant_passage(user_input, document_text, doc_embeddings):
    """Retrieve the most relevant passage based on user input."""
    input_embedding = get_embeddings(user_input)  # Get embedding for the user's input
    similarities = np.dot(doc_embeddings, input_embedding.T)  # Compute similarity (cosine similarity)
    best_match_idx = np.argmax(similarities)  # Find the index of the best match
    return document_text[best_match_idx]  # Return the relevant passage

def generate_response(user_input, relevant_passage):
    """Generate a response based on the user's input and the relevant passage."""
    prompt = f"Answer the following question based on the provided text:\n\n{relevant_passage}\n\nQuestion: {user_input}\nAnswer:"
    response = openai.Completion.create(model="gpt-4", prompt=prompt, temperature=0.5, max_tokens=200)
    return response.choices[0].text.strip()

# Example of integrating retrieval and generation:
user_input = "What are the key components of the AI Act?"
relevant_passage = retrieve_relevant_passage(user_input, pdf_text, text_embeddings)
response = generate_response(user_input, relevant_passage)
print(response)

This system retrieves the most relevant passage from the document based on the user's query and then uses ChatGPT to generate a response based on that passage.
Step 3: Web Interface with Flask

Now, let's set up a simple Flask web server to interact with the system. The frontend will allow users to submit questions, and the backend will fetch answers based on the content you’ve provided.
Create the Flask App:

from flask import Flask, render_template, request, jsonify
import openai
import numpy as np

app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Assuming you've already extracted text and created embeddings
document_text = pdf_text  # The extracted text from your PDF
document_embeddings = text_embeddings  # The embeddings of the extracted text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.form['question']
    
    # Retrieve relevant passage
    relevant_passage = retrieve_relevant_passage(user_input, document_text, document_embeddings)
    
    # Generate the answer
    answer = generate_response(user_input, relevant_passage)
    
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)

Create the HTML Frontend (index.html):

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System</title>
</head>
<body>
    <h1>AI-powered Regulation Query System</h1>
    <form id="questionForm">
        <input type="text" id="question" placeholder="Ask a question about the AI Act or ISO Standards" required>
        <button type="submit">Ask</button>
    </form>
    <div id="answer"></div>

    <script>
        document.getElementById('questionForm').onsubmit = async function(event) {
            event.preventDefault();
            
            const question = document.getElementById('question').value;
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: 'question=' + encodeURIComponent(question)
            });
            
            const data = await response.json();
            document.getElementById('answer').innerText = data.answer;
        };
    </script>
</body>
</html>

Step 4: Adding Content Dynamically

To allow the ability to add new content dynamically:

    Store PDFs: You could store PDFs (AI Act, ISO standards) in a folder or database.
    Extract Text: Add a route in your Flask app to allow uploading PDFs, extracting text, and updating the system with the new content.

Example code for handling file upload and text extraction:

from flask import request

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_pdf_text(file)
        embeddings = get_embeddings(text)
        
        # Save the text and embeddings to the database or files
        save_content(text, embeddings)
        return jsonify({"message": "Content uploaded and processed successfully!"})
    else:
        return jsonify({"error": "Invalid file format. Please upload a PDF."})

Conclusion

You now have a basic system that can:

    Extract text from PDFs.
    Retrieve relevant information from the extracted content.
    Use ChatGPT to generate answers based on the extracted content.
    Provide a simple Flask web interface where users can ask questions.
    Allow adding new content dynamically via PDF upload.

As you continue to build and refine the system, you can:

    Improve the search and retrieval techniques (e.g., by using embeddings or Elasticsearch).
    Use a more advanced UI framework (e.g., React or Vue.js) for a better user experience.
    Set up a proper deployment using Docker or on a cloud service like AWS or Heroku.
