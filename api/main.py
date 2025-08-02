from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from typing import Annotated
import numpy as np
import io
import os
import logging
from pathlib import Path

# Configurações Globais
API_CONFIG = {
    "title": "Horse/Human Classifier API",
    "description": "API para classificação de imagens entre cavalos e humanos",
    "version": "1.0.0",
    "model_path": Path(__file__).parent.parent / "model" / "horse_human_mobilenetv2.h5",
    "allowed_mime_types": ["image/jpeg", "image/png"],
    "input_shape": (224, 224)
}

# Setup Inicial
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"]
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helpers
def load_ml_model():
    """Carrega o modelo ML com tratamento de erros"""
    try:
        model = load_model(API_CONFIG["model_path"])
        logger.info(f"Model loaded from {API_CONFIG['model_path']}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError("Could not load ML model")

def preprocess_image(file_contents: bytes) -> np.ndarray:
    """Pré-processa a imagem para o formato do modelo"""
    img = image.load_img(
        io.BytesIO(file_contents),
        target_size=API_CONFIG["input_shape"]
    )
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Carregamento do Modelo (Durante Startup)
try:
    model = load_ml_model()
except Exception as e:
    logger.critical(f"Failed to initialize model: {str(e)}")
    raise e

# Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Endpoint de health check simplificado"""
    return {"status": "API Operational"}

@app.post(
    "/predict",
    response_model=dict,
    responses={
        200: {"description": "Predição bem-sucedida"},
        400: {"description": "Dados de entrada inválidos"},
        500: {"description": "Erro interno no processamento"}
    }
)
async def predict(
    file: Annotated[UploadFile, File(description="Imagem para classificação (JPEG/PNG)")]
):
    """
    Classifica uma imagem como 'horse' (cavalo) ou 'human' (pessoa)
    
    - **file**: Imagem nos formatos JPEG ou PNG
    """
    try:
        # Validação inicial
        if not file.content_type in API_CONFIG["allowed_mime_types"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de arquivo não suportado. Use: {', '.join(API_CONFIG['allowed_mime_types'])}"
            )

        # Processamento
        contents = await file.read()
        img_array = preprocess_image(contents)
        
        # Predição
        pred = model.predict(img_array)
        class_idx = np.argmax(pred[0])
        class_label = "horse" if class_idx == 0 else "human"
        confidence = float(np.max(pred[0]))
        
        logger.info(f"Prediction successful - Class: {class_label}, Confidence: {confidence:.2f}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder({
                "class": class_label,
                "confidence": confidence,
                "message": f"Classified as {class_label} with {confidence:.2%} confidence"
            })
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocorreu um erro durante o processamento da imagem"
        )

# Documentação Adicional (OpenAPI)
app.openapi_tags = [{
    "name": "classifier",
    "description": "Operações de classificação de imagens"
}]