import streamlit as st
import requests
from PIL import Image
import time

# Configurações
API_URL = "http://localhost:8000/predict"  # URL da API
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE_MB = 5

# CSS Customizado
st.markdown(
    """
    <style>
    .main {
        max-width: 600px;
        margin: auto;
        padding-top: 2rem;
    }
    .result-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        font-size: 1.2rem;
    }
    .horse-result {
        color: #4a6fa5;
        font-weight: bold;
    }
    .human-result {
        color: #5a8f69;
        font-weight: bold;
    }
    .confidence {
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def validate_image(file):
    """Valida o arquivo de imagem"""
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Arquivo muito grande. Tamanho máximo: {MAX_FILE_SIZE_MB}MB")
        return False
    return True

def display_result(response):
    """Exibe os resultados da classificação"""
    if response.status_code == 200:
        result = response.json()
        confidence = result["confidence"] * 100
        is_horse = result["class"] == "horse"
        
        result_text = "Is a horse" if is_horse else "Is a person"
        result_class = "horse-result" if is_horse else "human-result"
        
        st.markdown(
            f'<div class="result-box">'
            f'<div class="{result_class}">{result_text}</div>'
            f'<div class="confidence">Confidence: {confidence:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.error(f"API Error: {response.text}")

# Interface Principal
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Cabeçalho
    st.title("Horse or Human Classifier")
    st.markdown("Upload an image to classify whether it contains a horse or a person")
    
    # Área de upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=ALLOWED_EXTENSIONS,
        label_visibility="collapsed"
    )

    if uploaded_file and validate_image(uploaded_file):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(uploaded_file, use_container_width=True)

        with col2:
            if st.button("Classify Image", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        start_time = time.time()
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(API_URL, files=files, timeout=30)
                        
                        if response.status_code == 200:
                            st.toast(f"Processed in {time.time() - start_time:.2f}s", icon="⏱️")
                        
                        display_result(response)
                        
                    except requests.exceptions.RequestException:
                        st.error("API Connection Error")
                    except Exception as e:
                        st.error(f"Unexpected Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
