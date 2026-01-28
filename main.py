import streamlit as st
import os
import json
import glob
import re

# --- IMPORTATIONS LANGCHAIN (Standards) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Spaceflight Institute", page_icon="🚀", layout="wide")
st.title("🤖 Spaceflight I(A)nstitute")

# Configuration Proxy
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# --- GESTION DES DOSSIERS ---
base_folder = "data"
cours_folder = os.path.join(base_folder, "cours")
users_folder = os.path.join(base_folder, "users")

# Création automatique des dossiers
for folder in [base_folder, cours_folder, users_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- 1. FONCTIONS UTILITAIRES UTILISATEURS ---
def get_user_list():
    """Récupère la liste des fichiers JSON dans le dossier users"""
    files = glob.glob(os.path.join(users_folder, "*.json"))
    # On retourne juste le nom du fichier sans l'extension
    return [os.path.splitext(os.path.basename(f))[0] for f in files]

def create_user(username, level, tone):
    """Crée un fichier JSON pour un nouvel utilisateur"""
    filename = f"{username.lower().replace(' ', '_')}.json"
    filepath = os.path.join(users_folder, filename)
    
    data = {
        "utilisateur": {
            "prenom": username,
            "niveau": level, # Ex: Débutant, Expert
            "preferences_apprentissage": {
                "ton": tone,
                "contenu_prefere": "mixte"
            }
        }
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return filename

def load_user_preferences(username_file):
    """Charge les données du fichier JSON sélectionné"""
    filepath = os.path.join(users_folder, f"{username_file}.json")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

# --- 2. SÉLECTION INTELLIGENTE DES FICHIERS ---
def get_relevant_files(prompt, pdf_folder_path):
    all_pdfs = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not prompt or not all_pdfs:
        return all_pdfs 

    # Nettoyage simple du prompt pour extraire des mots-clés
    mots_vides = ["le", "la", "les", "de", "du", "des", "un", "une", "est", "sont", "comment", "quoi"]
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    keywords = [word for word in cleaned_prompt.split() if word not in mots_vides and len(word) > 2]
    
    selected_files = []
    for pdf_path in all_pdfs:
        filename = os.path.basename(pdf_path).lower()
        # On cherche si un mot clé est dans le nom
        if any(kw in filename for kw in keywords):
            selected_files.append(pdf_path)
            
    # Cas 2 : Aucun mot clé trouvé dans les titres -> Fallback
    # Si aucun mot clé ne correspond, on retourne tout (par sécurité)
    if not selected_files:
        return all_pdfs, True # True signifie "J'ai tout renvoyé par défaut"
    
    # Cas 3 : On a trouvé des fichiers spécifiques
    return list(set(selected_files)), False # False signifie "C'est une sélection précise"

# --- 3. INITIALISATION RAG (Pas de cache ici car context dynamique) ---
def initialize_rag_chain_dynamic(selected_files, user_config):
    
    # Extraction des infos utilisateur
    user_info = user_config.get("utilisateur", {})
    user_name = user_info.get("prenom", "Étudiant")
    user_level = user_info.get("niveau", "Intermédiaire")
    ai_tone = user_info.get("preferences_apprentissage", {}).get("ton", "neutre")

    # Chargement PDF
    all_pages = []
    for pdf_path in selected_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_pages.extend(loader.load())
        except Exception as e:
            print(f"Erreur fichier {pdf_path}: {e}")

    if not all_pages:
        return None

    # Vectorisation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_pages)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Modèle
    llm = Ollama(model="deepseek-r1:8b")
    
    # Prompt personnalisé selon le JSON
    system_prompt = (
        f"Tu es un tuteur personnel pour {user_name}. "
        f"Niveau de l'élève : {user_level}. "
        f"Ton style pédagogique doit être : {ai_tone}. "
        "Utilise le contexte fourni pour répondre. Si tu ne sais pas, dis-le."
        "\n\n"
        "{context}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    
    chain = create_stuff_documents_chain(llm, prompt_template)
    rag = create_retrieval_chain(retriever, chain)
    return rag

# --- INTERFACE SIDEBAR (GESTION COMPTES) ---
with st.sidebar:
    st.header("👤 Espace Membre")
    
    # Liste des utilisateurs existants
    existing_users = get_user_list()
    
    mode = st.radio("Option", ["Connexion", "Nouveau Profil"], label_visibility="collapsed")
    
    current_user_data = None
    
    if mode == "Connexion":
        if existing_users:
            selected_user = st.selectbox("Choisir un profil", existing_users)
            current_user_data = load_user_preferences(selected_user)
            if current_user_data:
                u_info = current_user_data["utilisateur"]
                st.info(f"👋 Bonjour **{u_info['prenom']}**\n\nNiveau : {u_info['niveau']}\nStyle : {u_info['preferences_apprentissage']['ton']}")
        else:
            st.warning("Aucun utilisateur. Créez un profil.")
            
    else: # Nouveau Profil
        with st.form("new_user"):
            new_name = st.text_input("Prénom")
            new_level = st.select_slider("Niveau", options=["Débutant", "Intermédiaire", "Expert"])
            new_tone = st.selectbox("Style de l'IA", ["Strict & Concis", "Pédagogique & Illustré", "Socratique (pose des questions)", "Avec un accent africain", "Avec un accent québécois"])
            if st.form_submit_button("Créer"):
                if new_name:
                    create_user(new_name, new_level, new_tone)
                    st.success("Profil créé ! Passez en mode 'Connexion'.")
                    st.rerun()

    st.divider()
    st.header("📚 Bibliothèque")
    uploaded_file = st.file_uploader("Ajouter un cours (PDF)", type="pdf")
    if uploaded_file:
        file_path = os.path.join(cours_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Cours ajouté !")

# --- ZONE DE CHAT ---
if not current_user_data:
    st.info("👈 Veuillez sélectionner ou créer un profil utilisateur dans la barre latérale pour commencer.")
    st.stop()

# Gestion historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Utilisateur
if prompt := st.chat_input("Posez votre question sur les cours..."):
    
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
        st.caption(f"🧠 Analyse basée sur : {', '.join(files_names)}")
        
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
                try :
                    rag_chain = initialize_rag_chain_dynamic(relevant_files, current_user_data)
                
                    if rag_chain:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        # Nettoyage optionnel des balises <think>
                        if "</think>" in answer:
                            answer = answer.split("</think>")[-1].strip()
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Erreur technique : {e}")