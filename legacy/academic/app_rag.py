# Filename: app_rag.py

import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# Load Knowledge Base and Retrieval Model
# ------------------------------
st.title("Natural Language to Python Code with RAG")

st.write("This application uses a Retrieval-Augmented Generation (RAG) approach to generate Python code from natural language descriptions. It first retrieves relevant examples from a large knowledge base, then uses a fine-tuned model (CodeT5) to produce a final code snippet.")

# Load the large knowledge base
@st.cache_data
def load_knowledge_base():
    kb_df = pd.read_csv("600kdataset_table.csv")
    return kb_df

knowledge_base_df = load_knowledge_base()

intents = knowledge_base_df['intent'].tolist()
snippets = knowledge_base_df['snippet'].tolist()

@st.cache_resource
def load_retrieval_model():
    return SentenceTransformer('all-mpnet-base-v2')

retrieval_model = load_retrieval_model()

@st.cache_resource
def compute_kb_embeddings(intents):
    return retrieval_model.encode(intents, convert_to_tensor=True)

kb_embeddings = compute_kb_embeddings(intents)

# ------------------------------
# Retrieval Function
# ------------------------------
def retrieve_knowledge(query, kb_embeddings, intents, snippets, top_k=3):
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, kb_embeddings)
    top_k_indices = torch.topk(similarities, k=top_k).indices[0].tolist()

    retrieved_intents = [intents[i] for i in top_k_indices]
    retrieved_snippets = [snippets[i] for i in top_k_indices]

    retrieved_contexts = [
        f"Problem: {intent} | Code: {snippet}"
        for intent, snippet in zip(retrieved_intents, retrieved_snippets)
    ]
    return retrieved_contexts

def augment_query(query, retrieved_docs, max_len=256):
    context = " ".join(retrieved_docs)
    # Truncation if needed to respect length constraints
    # Just a precaution if needed
    truncated_context = context[:max_len - len(query) - len("Context: Query: ")]
    return f"Context: {context} Query: {query}"

# ------------------------------
# Load the fine-tuned model and tokenizer
# ------------------------------
@st.cache_resource
def load_model_tokenizer():
    model_dir = './finetuned_model'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_tokenizer()

# ------------------------------
# Streamlit UI
# ------------------------------
user_input = st.text_area('Enter a programming task description in English:', height=150)

if st.button('Generate Code'):
    if user_input.strip():
        with st.spinner('Generating code...'):
            # Retrieve top documents
            retrieved_docs = retrieve_knowledge(user_input, kb_embeddings, intents, snippets)

            # Augment the query with retrieved context
            augmented_input = augment_query(user_input, retrieved_docs)
            
            input_ids = tokenizer.encode(
                augmented_input, return_tensors='pt', truncation=True, max_length=256
            ).to(device)

            # Generate code
            outputs = model.generate(
                input_ids=input_ids,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader('Generated Code:')
        st.code(generated_code, language='python')
        
        st.subheader('Retrieved Contexts:')
        for idx, doc in enumerate(retrieved_docs, start=1):
            st.write(f"**Retrieved Doc {idx}:** {doc}")

    else:
        st.warning('Please enter a description.')
