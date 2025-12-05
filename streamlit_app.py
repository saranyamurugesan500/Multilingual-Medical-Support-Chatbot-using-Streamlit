import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🩺 Medical Chatbot", layout="wide")
st.title("🩺 Multilingual Medical Support Chatbot")
st.markdown("---")
st.error("⚠️ Not a substitute for professional medical advice")

@st.cache_resource
def load_model():
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("./medical-t5-final")
        tokenizer = AutoTokenizer.from_pretrained("./medical-t5-final")
        return model, tokenizer
    except:
        return pipeline("text2text-generation", model="google/flan-t5-base"), None

def generate_response(model, tokenizer, query):
    if isinstance(model, pipeline):
        return model(f"medical qa: {query}", max_length=100)[0]['generated_text']
    
    inputs = tokenizer(f"medical qa: {query}", return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with st.spinner("Loading models..."):
    model, tokenizer = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Describe symptoms (any language)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                lang = detect(prompt)
                st.info(f"🌐 {lang}")
                response = generate_response(model, tokenizer, prompt)
                st.success(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except: st.error("Try English")

if st.button("🗑️ Clear"): 
    st.session_state.messages = []
    st.rerun()
