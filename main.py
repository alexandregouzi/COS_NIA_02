import streamlit as st
import os
import json
import glob
import re

# --- IMPORTATIONS LANGCHAIN ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Mon Assistant de Cours", page_icon="🤖")
st.title("🤖 Spaceflight I(A)nstitute")

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# --- GESTION DES DOSSIERS (NOUVELLE STRUCTURE) ---
base_folder = "./data"
cours_folder = os.path.join(base_folder, "cours")
users_folder = os.path.join(base_folder, "users")

# Création des dossiers s'ils n'existent pas
for folder in [base_folder, cours_folder, users_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_relevant_files(prompt, pdf_folder_path):
    """
    Retourne : (liste_fichiers, est_une_recherche_globale)
    """
    all_pdfs = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    
    # Cas 1 : Pas de question
    if not prompt:
        return all_pdfs, True 

    # Nettoyage
    mots_vides = ["le", "la", "les", "de", "du", "des", "un", "une", "est", "sont", "pour", "comment", "quoi", "quel", "quelle", "sur", "dans"]
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    keywords = [word for word in cleaned_prompt.split() if word not in mots_vides and len(word) > 2]
    
    selected_files = []
    
    for pdf_path in all_pdfs:
        filename = os.path.basename(pdf_path).lower()
        # On cherche si un mot clé est dans le nom
        if any(kw in filename for kw in keywords):
            selected_files.append(pdf_path)
            
    # Cas 2 : Aucun mot clé trouvé dans les titres -> Fallback
    if not selected_files:
        return all_pdfs, True # True signifie "J'ai tout renvoyé par défaut"
    
    # Cas 3 : On a trouvé des fichiers spécifiques
    return list(set(selected_files)), False # False signifie "C'est une sélection précise"

# --- 2. FONCTION RAG MODIFIÉE (Lit JSON dans ./DATA/USERS) ---
def initialize_rag_chain_with_files(selected_files, json_folder_path):
    
    # --- CHARGEMENT JSON (Depuis le dossier users) ---
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))
    
    user_name = "Étudiant"
    ai_tone = "pédagogique"
    preferred_content = "texte"

    # On prend le premier fichier user trouvé (ou on pourrait filtrer par ID plus tard)
    if json_files:
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                user_info = data.get("utilisateur", {})
                user_name = user_info.get("prenom", user_name)
                learning_prefs = user_info.get("preferences_apprentissage", {})
                ai_tone = learning_prefs.get("ton", ai_tone)
                preferred_content = learning_prefs.get("contenu préféré", preferred_content)
        except Exception as e:
            print(f"Erreur lecture JSON: {e}")

    # --- CHARGEMENT DES PDF SÉLECTIONNÉS ---
    if not selected_files:
        return None

    all_pages = []
    for pdf_path in selected_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_pages.extend(loader.load())
        except Exception as e:
            st.error(f"Erreur lecture {os.path.basename(pdf_path)}: {e}")

    if not all_pages:
        return None

    # --- DÉCOUPAGE & VECTORISATION ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_pages)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # --- MODÈLE ---
    llm = Ollama(model="mistral")
    
    system_prompt = (
        f"Tu es un assistant pour {user_name}. Ton ton : {ai_tone}. "
        f"Format préféré : {preferred_content}. "
        "Réponds en utilisant le contexte suivant :"
        "\n\n"
        "{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- INTERFACE (SIDEBAR) ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Ajouter un cours (PDF)", type="pdf")

# Gestion de l'upload : direction le dossier 'cours'
if uploaded_file is not None:
    file_path = os.path.join(cours_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Fichier ajouté dans {cours_folder} !")

# --- HISTORIQUE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ZONE DE CHAT DYNAMIQUE ---
if prompt := st.chat_input("Posez votre question..."):
    
    # 1. Affichage question utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ... DANS LA ZONE DE CHAT ...

    # 2. SELECTION ET REPONSE
    with st.chat_message("assistant"):
        # A. On appelle la nouvelle version de la fonction (récupère 2 variables)
        relevant_files, is_global_search = get_relevant_files(prompt, cours_folder)
        
        # Affichage du feedback utilisateur
        files_names = [os.path.basename(f) for f in relevant_files]
        
        if is_global_search:
            # Cas où aucun mot clé n'a matché
            st.warning("⚠️ Aucun fichier spécifique identifié par le titre. Recherche élargie à tous les cours.")
            with st.expander("Voir les détails (Debug)"):
                st.write(f"Question analysée : {prompt}")
                st.write(f"Fichiers disponibles : {files_names}")
        else:
            # Cas où le filtrage a fonctionné
            st.success(f"🎯 Ciblage réussi : {len(files_names)} document(s) pertinent(s).")
            st.caption(f"Sources : {', '.join(files_names)}")

        # B. On initialise le RAG
        if relevant_files:
             # ... le reste de ton code d'initialisation reste identique ...
            with st.spinner("Analyse des documents..."):
                rag_chain = initialize_rag_chain_with_files(relevant_files, users_folder)
                 
                
                if rag_chain:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Impossible de créer la chaîne d'analyse.")
        else:
            st.warning("Aucun document PDF trouvé dans le dossier 'cours'.")