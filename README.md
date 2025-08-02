# Classificador Cavalo vs Humano - Teste Técnico para Engenheiro de IA

## Visão Geral do Projeto
Este projeto implementa uma solução de deep learning para classificação binária de imagens entre cavalos e humanos. A solução inclui um modelo treinado, um serviço de inferência baseado em FastAPI, uma interface web em Streamlit e ferramentas completas de avaliação.

## Justificativa Técnica

### Arquitetura do Modelo
- **Base**: MobileNetV2 (pré-treinada no ImageNet) foi selecionada por seu equilíbrio entre acurácia e eficiência computacional, ideal para cenários que exigem baixa latência.
- **Transfer Learning**: A base do modelo foi congelada para aproveitar os recursos pré-treinados, treinando apenas as camadas de classificação superiores - abordagem eficaz para conjuntos de dados limitados.
- **Camadas Finais**:
  - Global Average Pooling para redução dimensional
  - Dropout (0.2) para regularização
  - Camada Dense com ativação softmax para probabilidades de classe

### Configuração de Treinamento
- **Função de Perda**: Entropia cruzada categórica - apropriada para classificação
- **Otimizador**: Adam com taxa de aprendizado 1e-3
- **Métricas**: Acurácia, Precisão e Recall monitorados
- **Aumento de Dados**: Rotações, deslocamentos, zoom e inversões para melhor generalização
- **Callbacks**:
  - ModelCheckpoint para salvar melhores pesos
  - EarlyStopping (paciência=3) para evitar overfitting

### Validação
- Divisão treino/validação (80/20)
- Normalização e redimensionamento (1/255) consistente
- Amostragem estratificada mantendo equilíbrio de classes

## Instruções de Configuração

### Pré-requisitos
- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Streamlit
- Dependências adicionais em `requirements.txt`

### Instalação
1. Clonar o repositório:
```bash
git clone https://github.com/[seu-repo]/horse-or-human-classifier.git
cd horse-or-human-classifier
```

2. Criar e ativar ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Instalar dependências:
```bash
pip install -r requirements.txt
```

## Instruções de Uso

### Inferência via API
1. Iniciar servidor FastAPI:
```bash
uvicorn api.main:app --reload
```

2. Enviar requisições:
```bash
curl -X POST -F "file=@imagem_teste.jpg" http://localhost:8000/predict
```

**Especificações da API:**
- Endpoint: `POST /predict`
- Formatos aceitos: JPEG, PNG
- Tamanho máximo: 5MB
- Resposta:
```json
{
  "class": "horse|human",
  "confidence": 0.95,
  "message": "Classified as horse with 95.00% confidence"
}
```

### Interface Web
1. Executar aplicação Streamlit:
```bash
streamlit run streamlit_app/app.py
```

2. Utilizar a interface para upload de imagens e visualização de previsões.

### Avaliação do Modelo
Executar script de avaliação:
```bash
python evaluation/evaluate_model.py
```

Saídas geradas:
- Matriz de confusão
- Relatório de classificação (precisão, recall, f1-score)
- Métricas de acurácia e perda

## Análise de Performance

### Efetividade do Modelo
Métricas alcançadas:
- Acurácia na validação: 98.2%
- Precisão (classe humana): 98.5%
- Recall (classe cavalo): 97.8%
- Tempo de inferência: <150ms em CPU

A solução atende plenamente aos requisitos de distinção entre cavalos e humanos em imagens, com desempenho robusto em diversas condições.

### Limitações e Melhorias

**Melhorias Imediatas:**
1. Ampliação do dataset com maior diversidade de exemplos
2. Hard negative mining para casos mal classificados
3. Quantização para otimização

**Melhorias Arquiteturais:**
1. Testar arquiteturas EfficientNet
2. Implementar aumento de dados durante inferência
3. Adicionar estimativa de incerteza

**Melhorias Operacionais:**
1. Containerização dos componentes (Docker)
2. Pipelines CI/CD para retreinamento
3. Monitoramento de desempenho

## Licença
Este projeto é código proprietário desenvolvido para processo de avaliação técnica. Todos os direitos reservados.