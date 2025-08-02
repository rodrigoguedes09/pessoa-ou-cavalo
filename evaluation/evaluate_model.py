# evaluation/evaluate_model.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from pathlib import Path
import mlflow
import sys
sys.path.append(str(Path(__file__).parent.parent))
from mlflow_config import mlflow_cfg

class AvaliadorModelo:
    """Classe para avaliação de modelos de classificação de imagens"""
    
    def __init__(self, caminho_modelo, dir_validacao, dir_saida):
        """
        Inicializa o avaliador de modelo
        
        Args:
            caminho_modelo (str): Caminho para o arquivo .h5 do modelo
            dir_validacao (str): Diretório com dados de validação
            dir_saida (str): Diretório para salvar resultados
        """
        self.caminho_modelo = Path(caminho_modelo)
        self.dir_validacao = Path(dir_validacao)
        self.dir_saida = Path(dir_saida)
        self.modelo = None
        self.gerador_validacao = None
        
        # Garante que o diretório de saída existe
        self.dir_saida.mkdir(parents=True, exist_ok=True)
    
    def carregar_modelo(self):
        """Carrega o modelo Keras a partir do arquivo"""
        try:
            self.modelo = load_model(self.caminho_modelo)
            print(f"Modelo carregado com sucesso: {self.caminho_modelo}")
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def preparar_dados_validacao(self):
        """Prepara o gerador de dados para validação"""
        try:
            datagen = ImageDataGenerator(rescale=1./255)
            self.gerador_validacao = datagen.flow_from_directory(
                self.dir_validacao,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )
            print(f"Dados de validação carregados: {len(self.gerador_validacao)} batches")
            return True
        except Exception as e:
            print(f"Erro ao carregar dados de validação: {e}")
            return False
    
    def avaliar_modelo(self):
        """Executa a avaliação completa do modelo"""
        if not self.carregar_modelo() or not self.preparar_dados_validacao():
            return False
        
        mlflow_cfg.setup()  # Configura o MLflow
        
        with mlflow.start_run(run_name="Model_Evaluation"):
            try:
                # Avaliação básica do modelo
                print("\nAvaliando modelo...")
                resultados = self.modelo.evaluate(self.gerador_validacao)
                
                # Log das métricas básicas no MLflow
                mlflow.log_metrics({
                    "loss": resultados[0],
                    "accuracy": resultados[1]
                })

                # Gera previsões para métricas detalhadas
                print("\nGerando previsões...")
                y_pred = self.modelo.predict(self.gerador_validacao)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true = self.gerador_validacao.classes
                nomes_classes = list(self.gerador_validacao.class_indices.keys())
                
                # Gera e salva matriz de confusão
                self._gerar_matriz_confusao(y_true, y_pred_classes, nomes_classes)
                
                # Gera e salva relatório de métricas
                self._gerar_relatorio_metricas(y_true, y_pred_classes, nomes_classes, 
                                             resultados[0], resultados[1])
                
                # Log de artefatos no MLflow
                mlflow.log_artifact(str(self.dir_saida / 'matriz_confusao.png'))
                mlflow.log_artifact(str(self.dir_saida / 'relatorio_metricas.txt'))
                
                print("\nAvaliação concluída com sucesso!")
                return True
                
            except Exception as e:
                print(f"\nErro durante avaliação: {e}")
                mlflow.log_param("error", str(e))
                return False
    
    def _gerar_matriz_confusao(self, y_true, y_pred, nomes_classes):
        """Gera e salva a matriz de confusão"""
        plt.figure(figsize=(8, 6))
        matriz = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(
            matriz,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=nomes_classes,
            yticklabels=nomes_classes
        )
        
        plt.title('Matriz de Confusão')
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        
        caminho_saida = self.dir_saida / 'matriz_confusao.png'
        plt.savefig(caminho_saida)
        plt.close()
        print(f"Matriz de confusão salva em: {caminho_saida}")
    
    def _gerar_relatorio_metricas(self, y_true, y_pred, nomes_classes, loss, accuracy):
        """Gera e salva o relatório de métricas"""
        relatorio = classification_report(
            y_true, y_pred,
            target_names=nomes_classes,
            digits=4,
            output_dict=True
        )
        
        # Log de métricas detalhadas no MLflow
        mlflow.log_metrics({
            "precision_horse": relatorio["horse"]["precision"],
            "recall_horse": relatorio["horse"]["recall"],
            "f1_horse": relatorio["horse"]["f1-score"],
            "precision_human": relatorio["person"]["precision"],
            "recall_human": relatorio["person"]["recall"],
            "f1_human": relatorio["person"]["f1-score"]
        })

        # Gera relatório formatado
        conteudo = f"""=== RELATÓRIO DE AVALIAÇÃO ===

Métricas básicas:
- Loss: {loss:.4f}
- Acurácia: {accuracy:.4f}

Relatório de classificação:
{classification_report(y_true, y_pred, target_names=nomes_classes, digits=4)}

Total de amostras: {len(y_true)}
"""
        caminho_saida = self.dir_saida / 'relatorio_metricas.txt'
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        
        print(f"Relatório de métricas salvo em: {caminho_saida}")

def main():
    """Função principal"""
    # Configurações - ATUALIZE COM SEUS CAMINHOS
    config = {
        'modelo': Path(__file__).parent.parent / "model" / "horse_human_mobilenetv2.h5",
        'validacao': Path(__file__).parent.parent / "data" / "splits" / "validation",
        'saida': Path(__file__).parent.parent / "evaluation" / "output"
    }
    
    print("Iniciando avaliação do modelo...")
    
    # Cria e executa o avaliador
    avaliador = AvaliadorModelo(
        config['modelo'],
        config['validacao'],
        config['saida']
    )
    
    if not avaliador.avaliar_modelo():
        print("\nA avaliação falhou. Verifique os erros acima.")
        exit(1)

if __name__ == '__main__':
    main()