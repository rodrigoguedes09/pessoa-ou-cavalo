# Classificador Cavalo vs Humano - Teste Way2

## Visão Geral do Projeto
Este projeto implementa uma solução completa de deep learning para classificação binária de imagens entre cavalos e humanos. A solução inclui:

1. Pipeline de treinamento com MLflow para experiment tracking
2. Modelo baseado em MobileNetV2 com transfer learning
3. Serviço de inferência via FastAPI
4. Interface web interativa com Streamlit
5. Ferramentas de avaliação e análise de performance

## Justificativa Técnica

### Arquitetura do Modelo
A arquitetura MobileNetV2 foi escolhida por seu equilíbrio entre eficiência computacional e precisão, ideal para classificação em tempo real. Foi utilizado transfer learning com as camadas convolucionais congeladas (pré-treinadas no ImageNet) para extrair features robustas com um dataset limitado, adicionando Global Average Pooling para redução dimensional e uma camada densa final com softmax para classificação binária.
- **Base Architecture**: MobileNetV2 (pré-treinada no ImageNet) foi selecionada por seu equilíbrio entre acurácia e eficiência computacional
- **Transfer Learning**: Congelamento das camadas base para aproveitamento de features pré-treinadas
- **Camadas Adicionais**:
  - Global Average Pooling (redução dimensional)
  - Dropout (0.2) para regularização
  - Dense Layer com ativação softmax para classificação

### Configuração de Treinamento
O treinamento foi configurado com otimizador Adam (learning rate=1e-3) e função de perdo categórica cross-entropy, adequada para classificação binária. Como não tinham muitos dados, foi usado data augmentation com rotações, deslocamentos e flip horizontal para aumentar a robustez do modelo, enquanto callbacks de ModelCheckpoint e EarlyStopping (paciência=3) garantiam a seleção do melhor modelo e previniam overfitting.
- **Função de Perda**: Categorical Crossentropy
- **Otimizador**: Adam (learning rate=1e-3)
- **Métricas Monitoradas**: Accuracy, Precision, Recall
- **Data Augmentation**: Rotação (20°), deslocamento (0.2), zoom (0.2), flip horizontal
- **Callbacks**:
  - ModelCheckpoint (salva melhores pesos)
  - EarlyStopping (patience=3)

### Monitoramento com MLflow
- Tracking automático de:
  - Parâmetros do modelo
  - Métricas de treino/validação
  - Artefatos (modelos, gráficos)
- Interface visual disponível via `mlflow ui`

## Instruções de Configuração

### Pré-requisitos
- Python 3.8+
- TensorFlow 2.12+
- MLflow 2.3+
- Dependências completas em `requirements.txt`

### Instalação
```bash
git clone https://github.com/rodrigoguedes09/horse-human-classifier.git
cd horse-human-classifier
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Fluxo de Execução Completo

### 1. Treinamento do Modelo
```bash
python training/train_model.py
```
**Saídas**:
- Modelo salvo em `model/horse_human_mobilenetv2.h5`
- Dados de treino registrados no MLflow

### 2. Avaliação do Modelo
```bash
python evaluation/evaluate_model.py
```
**Saídas**:
- Matriz de confusão em `evaluation/output/matriz_confusao.png`
- Relatório completo em `evaluation/output/relatorio_metricas.txt`
- Métricas adicionais no MLflow

### 3. Serviço de Inferência (API)
```bash
uvicorn api.main:app --reload
```
**Endpoints**:
- `POST /predict`: Classificação de imagens
- `GET /docs`: Documentação interativa (Swagger)

### 4. Interface Web
Para facilitar a visualização e teste com diferentes imagens, o Streamlit foi utilizado. Apesar de simples, essa é uma forma fácil de conseguir uma aplicação estéticamente agradável e que é facilmente implementada com códigos Python.
```bash
streamlit run streamlit_app/app.py
```
**Funcionalidades**:
- Upload de imagens via interface gráfica
- Visualização dos resultados
- Exibição do nível de confiança

<img width="939" height="819" alt="image" src="https://github.com/user-attachments/assets/db6d1443-fa5c-4d27-a5e7-4fabc2f265c6" />



## Monitoramento com MLflow

### Acesso aos Resultados
```bash
mlflow ui
```
Acesse: http://localhost:5000

**Recursos disponíveis**:
- Comparação entre execuções
- Visualização de métricas históricas
- Download de modelos e artefatos

### Dados Registrados Automaticamente
- Parâmetros (learning rate, batch size, etc.)
- Métricas (loss, accuracy por época)
- Artefatos (modelos, gráficos de treinamento)
- Ambiente (versões de pacotes)

## Análise de Performance

### Métricas Obtidas
| Classe   | Precision | Recall | F1-Score | Suporte |
|----------|-----------|--------|----------|---------|
| horse    | 1.0000    | 0.9922 | 0.9961   | 128     |
| person   | 0.9922    | 1.0000 | 0.9961   | 128     |

**Métricas Globais**  
- **Acurácia**: 0.9961  
- **Macro Avg**: 0.9961 (precision), 0.9961 (recall), 0.9961 (f1-score)  
- **Weighted Avg**: 0.9961 (precision), 0.9961 (recall), 0.9961 (f1-score) 

### Limitações Conhecidas
1. Sensibilidade a fundos complexos
2. Dificuldade com imagens de baixa resolução
3. Variabilidade em poses atípicas

### Roadmap de Melhorias
1. **Dataset**:
   - Ampliação com exemplos negativos
   - Balanceamento de subclasses

2. **Modelo**:
   - Teste com EfficientNetV2
   - Técnicas de fine-tuning avançado

3. **Sistema**:
   - Containerização com Docker
   - Pipeline de CI/CD

## Licença
Este projeto é código proprietário desenvolvido para processo de avaliação técnica. Todos os direitos reservados.
