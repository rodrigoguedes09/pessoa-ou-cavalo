import mlflow
import os
from pathlib import Path

class MLflowConfig:
    def __init__(self):
        # Configuração especial para Windows
        self.tracking_uri = "mlruns"  # Usando caminho relativo
        self.experiment_name = "Horse-Human-Classification"
        
    def setup(self):
        # Cria o diretório se não existir
        os.makedirs("mlruns", exist_ok=True)
        
        # Configuração especial para Windows
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            mlflow.set_experiment(self.experiment_name)
        except:
            pass  # Ignora erros de experimento existente
        
        mlflow.tensorflow.autolog()

mlflow_cfg = MLflowConfig()