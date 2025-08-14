# Horse vs Human Classifier - Way2 Test

## Project Overview
This project implements a complete deep learning solution for binary image classification between horses and humans. The solution includes:

1. Training pipeline with MLflow for experiment tracking  
2. MobileNetV2-based model with transfer learning  
3. Inference service via FastAPI  
4. Interactive web interface with Streamlit  
5. Performance evaluation and analysis tools  

## Technical Justification

### Model Architecture
The MobileNetV2 architecture was chosen for its balance between computational efficiency and accuracy, ideal for real-time classification. We used transfer learning with frozen convolutional layers (pre-trained on ImageNet) to extract robust features with limited data, adding Global Average Pooling for dimensionality reduction and a final dense layer with softmax for binary classification.

- **Base Architecture**: MobileNetV2 (ImageNet pre-trained) selected for its accuracy/computational efficiency balance  
- **Transfer Learning**: Frozen base layers to leverage pre-trained features  
- **Additional Layers**:  
  - Global Average Pooling (dimensionality reduction)  
  - Dropout (0.2) for regularization  
  - Softmax-activated Dense Layer for classification  

### Training Configuration
Training used Adam optimizer (learning rate=1e-3) and categorical cross-entropy loss, suitable for binary classification. With limited data, we applied augmentation (rotations, shifts, horizontal flip) to improve model robustness, while ModelCheckpoint and EarlyStopping (patience=3) callbacks ensured best model selection and prevented overfitting.

- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam (learning rate=1e-3)  
- **Monitored Metrics**: Accuracy, Precision, Recall  
- **Data Augmentation**: Rotation (20°), shift (0.2), zoom (0.2), horizontal flip  
- **Callbacks**:  
  - ModelCheckpoint (saves best weights)  
  - EarlyStopping (patience=3)  

### MLflow Monitoring
- Automatic tracking of:  
  - Model parameters  
  - Training/validation metrics  
  - Artifacts (models, graphs)  
- Visual interface via `mlflow ui`  

## Setup Instructions

### Prerequisites
- Python 3.8+  
- TensorFlow 2.12+  
- MLflow 2.3+  
- Full dependencies in `requirements.txt`  

### Installation
```bash
git clone https://github.com/rodrigoguedes09/horse-human-classifier.git
cd horse-human-classifier
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Complete Execution Flow

### 1. Model Training
```bash
python training/train_model.py
```
**Outputs**:  
- Model saved at `model/horse_human_mobilenetv2.h5`  
- Training data logged in MLflow  

### 2. Model Evaluation
```bash
python evaluation/evaluate_model.py
```
**Outputs**:  
- Confusion matrix at `evaluation/output/confusion_matrix.png`  
- Full report at `evaluation/output/metric_report.txt`  
- Additional metrics in MLflow  

### 3. Inference Service (API)
```bash
uvicorn api.main:app --reload
```
**Endpoints**:  
- `POST /predict`: Image classification  
- `GET /docs`: Interactive documentation (Swagger)  

### 4. Web Interface
For easy visualization and testing with different images, we used Streamlit. Despite its simplicity, it provides an aesthetically pleasing application that's easily implemented with Python code.

**NOTE: For proper operation, run the Streamlit app in a separate terminal from the Inference Service (API).**

```bash
# To run correctly, you must be in the correct application directory .../horse-human-classifier
# We'll reuse the same environment from the first terminal (venv), saving reinstallation of requirements.txt

source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

streamlit run streamlit_app/app.py
```
**Features**:  
- Graphical image upload  
- Result visualization  
- Confidence level display  

<img width="939" height="819" alt="image" src="https://github.com/user-attachments/assets/db6d1443-fa5c-4d27-a5e7-4fabc2f265c6" />

## MLflow Monitoring

### Accessing Results
```bash
mlflow ui
```
Access the MLFlow URL for detailed analysis and model monitoring.

**Available features**:  
- Run comparisons  
- Historical metric visualization  
- Model/artifact downloads  

### Automatically Logged Data
- Parameters (learning rate, batch size, etc.)  
- Metrics (loss, accuracy per epoch)  
- Artifacts (models, training graphs)  
- Environment (package versions)  

## Performance Analysis

### Obtained Metrics
| Class   | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| horse    | 1.0000    | 0.9922 | 0.9961   | 128     |
| person   | 0.9922    | 1.0000 | 0.9961   | 128     |

**Global Metrics**  
- **Accuracy**: 0.9961  
- **Macro Avg**: 0.9961 (precision), 0.9961 (recall), 0.9961 (f1-score)  
- **Weighted Avg**: 0.9961 (precision), 0.9961 (recall), 0.9961 (f1-score)  

### Known Limitations
1. Sensitivity to complex backgrounds  
2. Difficulty with low-resolution images  
3. Variability with atypical poses  

### Improvement Roadmap
1. **Dataset**:  
   - Expansion with negative examples  
   - Subclass balancing  

2. **Model**:  
   - Testing with EfficientNetV2  
   - Advanced fine-tuning techniques  

3. **System**:  
   - Docker containerization  
   - CI/CD pipeline  
