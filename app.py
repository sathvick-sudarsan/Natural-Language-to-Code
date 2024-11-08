# Filename: app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained('./finetuned_model')
tokenizer = AutoTokenizer.from_pretrained('./finetuned_model')
model.eval()

st.title('Natural Language to Python Code Generator')

user_input = st.text_area('Enter a programming task description in English:', height=150)

if st.button('Generate Code with Transformer'):
    if user_input.strip():
        source = "Translate to Python: " + user_input
        input_ids = tokenizer.encode(source, return_tensors='pt', truncation=True, max_length=256)

        outputs = model.generate(
            input_ids=input_ids,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader('Generated Code (Transformer):')
        st.code(generated_code, language='python')
    else:
        st.warning('Please enter a description.')
