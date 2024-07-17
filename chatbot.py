from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
from vertexai.language_models import TextGenerationModel
from google.cloud import storage
from google.cloud import aiplatform
import PyPDF2
import faiss
import numpy as np
import re
import os
import json
import uuid
from vertexai.generative_models import GenerativeModel
import streamlit as st
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import base64
from PyPDF2 import PdfReader
from google.cloud import aiplatform
import fitz  
import random
import string
import psycopg2
from psycopg2.extras import RealDictCursor
###################################################
def create_db_connection():
    return psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="",
        port=5432)

def create_tables():
    conn = create_db_connection()
    with conn.cursor() as cur:
        # Check if tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        existing_tables = [row[0] for row in cur.fetchall()]

        # Create teams table if it doesn't exist
        if 'teams' not in existing_tables:
            cur.execute("""
                CREATE TABLE teams (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    code VARCHAR(6) UNIQUE NOT NULL
                )
            """)

        # Create team_members table if it doesn't exist
        if 'team_members' not in existing_tables:
            cur.execute("""
                CREATE TABLE team_members (
                    id SERIAL PRIMARY KEY,
                    team_id INTEGER REFERENCES teams(id),
                    name VARCHAR(255) NOT NULL
                )
            """)

        # Create chat_history table if it doesn't exist
        if 'chat_history' not in existing_tables:
            cur.execute("""
                CREATE TABLE chat_history (
                    id SERIAL PRIMARY KEY,
                    team_id INTEGER REFERENCES teams(id),
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL
                )
            """)

        # Create pdfs table if it doesn't exist
        if 'pdfs' not in existing_tables:
            cur.execute("""
                CREATE TABLE pdfs (
                    id SERIAL PRIMARY KEY,
                    team_id INTEGER REFERENCES teams(id),
                    file_path VARCHAR(255) NOT NULL
                )
            """)

    conn.commit()
    conn.close()
    
def create_team(team_name):
    team_code = generate_team_code()
    conn = create_db_connection()
    with conn.cursor() as cur:
        cur.execute("INSERT INTO teams (name, code) VALUES (%s, %s) RETURNING id", (team_name, team_code))
        team_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return team_code

def join_team(team_code):
    conn = create_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM teams WHERE code = %s", (team_code,))
        team = cur.fetchone()
    conn.close()
    return team is not None

########################################################################################
def extract_sentences_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    sentences = []
    for page in document:
        text = page.get_text()
        sentences.extend(text.split('.'))  # Simple sentence splitting by period
    document.close()
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def generate_and_save_embeddings(pdf_folder, sentence_file_path, embed_file_path):
    def clean_text(text):
        cleaned_text = re.sub(r'\u2022', '', text)  # Remove bullet points
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra whitespaces and strip
        return cleaned_text

    all_sentences = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            sentences = extract_sentences_from_pdf(pdf_path)
            all_sentences.extend(sentences)

    if all_sentences:
        embeddings = generate_text_embeddings(all_sentences)

        with open(embed_file_path, 'w') as embed_file, open(sentence_file_path, 'w') as sentence_file:
            for sentence, embedding in zip(all_sentences, embeddings):
                cleaned_sentence = clean_text(sentence)
                id = str(uuid.uuid4())

                embed_item = {"id": id, "embedding": embedding}  # Convert numpy array to list
                sentence_item = {"id": id, "sentence": cleaned_sentence}

                json.dump(sentence_item, sentence_file)
                sentence_file.write('\n')
                json.dump(embed_item, embed_file)
                embed_file.write('\n')

def generate_text_embeddings(sentences) -> list:
    aiplatform.init(project=project,location=location)
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings(sentences)
    vectors = [embedding.values for embedding in embeddings]
    print("created vectors")
    return vectors

def create_and_save_faiss_index(embed_file_path, index_file_path):
    embeddings = []
    ids = []
    with open(embed_file_path, 'r') as embed_file:
        for line in embed_file:
            embed_item = json.loads(line)
            embeddings.append(embed_item['embedding'])
            ids.append(embed_item['id'])

    embeddings_array = np.array(embeddings).astype('float32')
    d = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_array)
    faiss.write_index(index, index_file_path)

    with open(index_file_path + '.ids', 'w') as id_file:
        json.dump(ids, id_file)
    print(f"Index saved to {index_file_path}")
    print(f"IDs saved to {index_file_path}.ids")

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to {destination_blob_name}.")

def load_local_faiss_index(index_path, ids_path):
    index = faiss.read_index(index_path)
    with open(ids_path, 'r') as id_file:
        ids = json.load(id_file)
    return index, ids

def generate_embedding(text):
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = embedding_model.get_embeddings([text])
    if embeddings and embeddings[0].values:
        return np.array(embeddings[0].values, dtype=np.float32)
    else:
        raise ValueError("Failed to generate embedding")

def search_faiss_index(index, ids, query_vector, k=5):
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, k)
    results = [
        {"id": ids[idx], "score": float(1 / (1 + dist))}
        for idx, dist in zip(indices[0], distances[0])
    ]
    return results

def fetch_content_by_id(id, content_file_path):
    with open(content_file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            if item['id'] == id:
                return item['sentence']
    return None

def answer_prompt(prompt, index, ids, content_file_path, model):
    query_embedding = generate_embedding(prompt)
    similar_items = search_faiss_index(index, ids, query_embedding)
    context = ""
    for item in similar_items:
        content = fetch_content_by_id(item['id'], content_file_path)
        if content:
            context += content + " "
    full_prompt = f"Based on the following context from a PDF:\n\n{context}\n\nAnswer this question: {prompt}"
    response = model.generate_content(full_prompt)
    return response.text

def generate_workflow(project_name, project_description, model):
    prompt = f"""
    Project Name: {project_name}
    Project Description: {project_description}

    Based on the above information, generate a basic workflow for this project. 
    Just include key words keep it less than 50 words, just key words with arrows point through the flow
    """
    response = model.generate_content(prompt)
    return response.text
###########################################
def generate_team_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def create_team(team_name):
    team_code = generate_team_code()
    # Store team information (you might want to use a database for this)
    if 'teams' not in st.session_state:
        st.session_state.teams = {}
    st.session_state.teams[team_code] = {
        'name': team_name,
        'members': [],
        'chat_history': [],
        'pdfs': []
    }
    return team_code

def join_team(team_code):
    if team_code in st.session_state.teams:
        # Add logic to add the current user to the team
        return True
    return False




############################

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main execution
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "drgenai-7daf412ca440.json"
project="drgenai"
location="us-central1"
pdf_path="pdf_docs"
sentence_file_path = "resume_sentences.json"
index_name="resume_index-final-try"
embed_file_path = 'resume_embeddings.json'
index_file_path = 'faiss_index'
file_name="style.css"
aiplatform.init(project=project,location=location)

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'workflow' not in st.session_state:
        st.session_state.workflow = ""
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = []
    if 'pinned_messages' not in st.session_state:
        st.session_state.pinned_messages = []
    

def main():
    create_tables()
    st.set_page_config("Project Workflow and Chat", layout="wide")
    load_css("style.css")
    init_session_state()

    st.sidebar.title("Team Options")
    team_option = st.sidebar.radio("Choose an option", ["Create Team", "Join Team", "Solo Mode"])

    if team_option == "Create Team":
        team_name = st.sidebar.text_input("Enter team name:")
        if st.sidebar.button("Create Team"):
            team_code = create_team(team_name)
            st.sidebar.success(f"Team created! Your team code is: {team_code}")
            st.session_state.current_team = team_code

    elif team_option == "Join Team":
        team_code = st.sidebar.text_input("Enter team code:")
        if st.sidebar.button("Join Team"):
            if join_team(team_code):
                st.sidebar.success("Joined team successfully!")
                st.session_state.current_team = team_code
            else:
                st.sidebar.error("Invalid team code")

    if 'current_team' in st.session_state:
        st.sidebar.write(f"Current Team: {st.session_state.teams[st.session_state.current_team]['name']}")

    if not st.session_state.show_chat:
        display_workflow_page()
    else:
        display_chat_page()

def display_workflow_page():
    st.header("Generate Project Workflow")

    col1, col2 = st.columns([2, 1])

    with col1:
        project_name = st.text_input("Enter your project name:")
        project_description = st.text_area("Enter a brief description of your project:", height=150)
        
        if st.button("Generate Workflow"):
            if project_name and project_description:
                vertexai.init(project=project, location=location)
                model = GenerativeModel("gemini-pro")
                st.session_state.workflow = generate_workflow(project_name, project_description, model)

    with col2:
        if st.session_state.workflow:
            workflow_edit = st.text_area("Edit the workflow if needed:", value=st.session_state.workflow, height=300)
            
            if st.button("Submit"):
                st.session_state.workflow = workflow_edit
                st.session_state.show_chat = True
                st.experimental_rerun()

def display_chat_page():
    st.title("Chat with PDF")

    with st.sidebar:
        st.header("PDF Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            process_pdfs(pdf_docs)

        if 'current_team' in st.session_state:
            conn = create_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT file_path FROM pdfs WHERE team_id = %s", (st.session_state.current_team,))
                team_pdfs = [row['file_path'] for row in cur.fetchall()]
            conn.close()

            if team_pdfs:
                selected_pdf = st.selectbox("Select a PDF to preview", team_pdfs)
                if selected_pdf:
                    display_pdf(selected_pdf)
        else:
            if st.session_state.uploaded_pdfs:
                selected_pdf = st.selectbox("Select a PDF to preview", st.session_state.uploaded_pdfs)
                if selected_pdf:
                    display_pdf(selected_pdf)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Current Workflow")
        st.text_area("", value=st.session_state.workflow, height=100, disabled=True)

        st.subheader("How can I help you?")
        user_question = st.text_input("Enter your question:")
        if st.button("Submit Question"):
            handle_user_question(user_question)

        if 'current_team' in st.session_state:
            conn = create_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT question, answer FROM chat_history WHERE team_id = %s ORDER BY id", 
                            (st.session_state.current_team,))
                chat_history = cur.fetchall()
            conn.close()

            for i, chat in enumerate(chat_history):
                st.write(f"Q: {chat['question']}")
                st.write(f"A: {chat['answer']}")
                if st.button("Pin", key=f"pin_{i}"):
                    st.session_state.pinned_messages.append((chat['question'], chat['answer']))
                st.write("---")
        else:
            if 'chat_history' in st.session_state:
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    st.write(f"Q: {q}")
                    st.write(f"A: {a}")
                    if st.button("Pin", key=f"pin_{i}"):
                        st.session_state.pinned_messages.append((q, a))
                    st.write("---")

    with col2:
        display_pinned_messages()

def display_pinned_messages():
    st.subheader("Pinned Messages")
    for i, (q, a) in enumerate(st.session_state.pinned_messages):
        if st.checkbox(f"Show pinned message {i+1}", key=f"expand_{i}"):
            st.write(f"Q: {q}")
            st.write(f"A: {a}")

def handle_user_question(user_question):
    if user_question:
        vertexai.init(project=project, location=location)
        model = GenerativeModel("gemini-pro")

        if 'pdfs_processed' in st.session_state and st.session_state.pdfs_processed:
            index_path = "faiss_index"
            ids_path = "faiss_index.ids"
            content_file_path = "resume_sentences.json"
            index, ids = load_local_faiss_index(index_path, ids_path)
            answer = answer_prompt(user_question, index, ids, content_file_path, model)
        else:
            answer = model.generate_content(user_question).text

        if 'current_team' in st.session_state:
            conn = create_db_connection()
            with conn.cursor() as cur:
                cur.execute("INSERT INTO chat_history (team_id, question, answer) VALUES (%s, %s, %s)",
                            (st.session_state.current_team, user_question, answer))
            conn.commit()
            conn.close()
        else:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append((user_question, answer))

        st.write("Reply:", answer)
    else:
        st.warning("Please enter a question.")

def process_pdfs(pdf_docs):
    with st.spinner("Processing..."):
        if not os.path.exists(pdf_path):
            os.makedirs(pdf_path)
        
        if 'current_team' in st.session_state:
            conn = create_db_connection()
            with conn.cursor() as cur:
                for uploaded_file in pdf_docs:
                    file_path = os.path.join(pdf_path, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    cur.execute("INSERT INTO pdfs (team_id, file_path) VALUES (%s, %s)",
                                (st.session_state.current_team, file_path))
            conn.commit()
            conn.close()
        else:
            for uploaded_file in pdf_docs:
                file_path = os.path.join(pdf_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_pdfs.append(file_path)
        
        try:
            generate_and_save_embeddings(pdf_path, sentence_file_path, embed_file_path)
            create_and_save_faiss_index(embed_file_path, index_file_path)
            
            if os.path.exists(index_file_path) and os.path.exists(index_file_path + '.ids'):
                upload_to_gcs('resume-bucket-final', index_file_path, 'faiss_index')
                upload_to_gcs('resume-bucket-final', index_file_path + '.ids', 'faiss_index.ids')
                st.session_state.pdfs_processed = True
                st.sidebar.success("PDFs processed and uploaded successfully!")
            else:
                st.sidebar.error("Index files not found. Processing failed.")
        except Exception as e:
            st.sidebar.error(f"An error occurred during processing: {str(e)}")
            
def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    main()