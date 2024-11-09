import requests
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
app = Flask(__name__)

# Load the dataset with the correct configuration
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train", trust_remote_code=True)

# Load the retriever and model
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="psgs_w100", passages=dataset)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

# FAISS index
faiss_index = None
texts = []


# Chain of Thought Module
class ChainOfThought:
    def __init__(self):
        self.context = []

    def add_to_context(self, message):
        self.context.append(message)
        if len(self.context) > 5:  # Limit context length
            self.context.pop(0)

    def get_context(self):
        return " ".join(self.context)


cot = ChainOfThought()


def fetch_wordpress_posts(api_url):
    response = requests.get(f"{api_url}/wp-json/wp/v2/posts")
    if response.status_code == 200:
        return response.json()
    else:
        return None


def extract_text(post):
    return post['title']['rendered'] + " " + post['content']['rendered']


def generate_embeddings(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use Sentence-BERT for embeddings
    return model.encode(texts)


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index


def search_similar_embeddings(query_embedding, index, k=5):
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    return distances, indices


def generate_response(query):
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def update_embeddings_on_new_post(post):
    text = extract_text(post)
    embeddings = generate_embeddings([text])
    update_vector_database(post['id'], embeddings)


def update_vector_database(post_id, embeddings):
    global faiss_index
    if faiss_index is not None:
        faiss_index.add(np.array(embeddings).astype('float32'))


def process_query_with_chain_of_thought(user_query, previous_context):
    initial_response = generate_response(user_query)
    thought_steps = develop_reasoning_steps(initial_response, previous_context)
    final_response = refine_response_based_on_thought_steps(thought_steps)
    return final_response


def develop_reasoning_steps(initial_response, previous_context):
    return [initial_response]  # Placeholder for demonstration


def refine_response_based_on_thought_steps(thought_steps):
    return " ".join(thought_steps)  # Placeholder for demonstration


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    cot.add_to_context(user_message)

    # Fetch posts and update embeddings
    api_url = "https://grabtheworld3.wordpress.com"  # Replace with your WordPress site
    posts = fetch_wordpress_posts(api_url)
    if posts:
        global texts
        texts = [extract_text(post) for post in posts]
        embeddings = generate_embeddings(texts)
        global faiss_index
        faiss_index = create_faiss_index(embeddings)

        # Update embeddings for each new post
        for post in posts:
            update_embeddings_on_new_post(post)

    # Generate response using Chain of Thought
    context = cot.get_context()
    response = process_query_with_chain_of_thought(user_message, context)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
