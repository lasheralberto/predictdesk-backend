import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader

class FeedbackDataset(Dataset):
    """Clase interna para manejo de datos"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_label': torch.tensor(self.labels['sentiment'][idx]),
            'priority_label': torch.tensor(self.labels['priority'][idx]),
            'category_label': torch.tensor(self.labels['category'][idx]),
            'team_label': torch.tensor(self.labels['team'][idx])
        }

class MultiTaskFeedbackModel:
    """Clase principal que encapsula todo el modelo multitarea"""
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoders = {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        # Mapeo entre tareas internas y nombres reales de columnas
        self.task_to_column = {
            'sentiment': 'sentiment_score',
            'priority': 'priority',
            'category': 'category',
            'team': 'team_assigned'
        }

        # Columna input
        self.text_column = 'feedback'
        #Columnas del fichero
        self.target_columns = list(self.task_to_column.values())
        #Columnas del fichero traducidas para procesar
        self.task_labels =  list(self.task_to_column.keys())

        self.FeedbackDataset = FeedbackDataset


    def load_and_prepare_data(self, filepath):
        """Carga y prepara los datos para entrenamiento"""

        df = pd.read_csv(filepath, on_bad_lines='skip')
        print(f"Datos cargados: {len(df)} registros")

        # Codificar etiquetas
        for col in self.target_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            print(f"{col} classes: {le.classes_}")

        # Preparar estructura de labels
        labels = {
            'sentiment': df['sentiment_score_encoded'].values,
            'priority': df['priority_encoded'].values,
            'category': df['category_encoded'].values,
            'team': df['team_assigned_encoded'].values
        }

        return self.FeedbackDataset(
            df[self.text_column].tolist(),
            labels,
            self.tokenizer
        )


    def initialize_model(self):
        """Inicializa el modelo multitarea"""
        if not self.label_encoders:
            raise ValueError("Debes cargar datos primero para inicializar el modelo")

        self.model = nn.ModuleDict({
            'bert': AutoModel.from_pretrained(self.model_name),
            'dropout': nn.Dropout(0.1),
            'sentiment': nn.Linear(768, len(self.label_encoders['sentiment_score'].classes_)),
            'priority': nn.Linear(768, len(self.label_encoders['priority'].classes_)),
            'category': nn.Linear(768, len(self.label_encoders['category'].classes_)),
            'team': nn.Linear(768, len(self.label_encoders['team_assigned'].classes_))
        }).to(self.device)

    def train(self, dataset, epochs=3, batch_size=8):
        """Entrena el modelo multitarea"""
        if not self.model:
            self.initialize_model()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in dataloader:
                optimizer.zero_grad()

                # Mover datos al dispositivo
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }

                # Forward pass
                outputs = self.model['bert'](**inputs)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                pooled_output = self.model['dropout'](pooled_output)

                # Calcular salidas para cada tarea
                task_outputs = {
                    'sentiment': self.model['sentiment'](pooled_output),
                    'priority': self.model['priority'](pooled_output),
                    'category': self.model['category'](pooled_output),
                    'team': self.model['team'](pooled_output)
                }

                # Calcular pérdida combinada
                loss = sum(
                    loss_fn(task_outputs[task], batch[f'{task}_label'].to(self.device))
                    for task in self.task_labels
                )

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    def predict(self, text):
        """Realiza predicciones para las 4 tareas"""
        if not self.model:
            raise ValueError("El modelo no ha sido entrenado o cargado")

        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model['bert'](**encoding)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            pooled_output = self.model['dropout'](pooled_output)

            predictions = {}
            for task in self.task_labels:
                logits = self.model[task](pooled_output)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs).item()

                # Obtener el nombre real de la columna
                column_name = self.task_to_column[task]

                predictions[column_name] = {
                    'class': self.label_encoders[column_name].inverse_transform([pred_class])[0],
                    'confidence': probs[0][pred_class].item()
                }

        return predictions

    def save(self, path):
        import os
        import torch
        import joblib
        import shutil
        from google.colab import files
        """Guarda el modelo, encoders y permite descargar como zip en Colab"""
        os.makedirs(path, exist_ok=True)

        # Guardar pesos del modelo
        torch.save(self.model.state_dict(), f"{path}/model_weights.pt")

        # Guardar los label encoders
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")

        # Guardar tokenizer
        self.tokenizer.save_pretrained(path)

        # Comprimir la carpeta en un zip
        zip_path = f"{path}.zip"
        shutil.make_archive(path, 'zip', path)

        # Descargar el zip en Colab
        files.download(zip_path)

    def load(self, path):
        """Carga un modelo guardado"""
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.initialize_model()
        self.model.load_state_dict(torch.load(f"{path}/model_weights.pt", map_location=self.device))
        self.model.to(self.device)

    ##Evaluación del modelo
    def evaluate_model(self, model, test_dataset, log_to_mlflow=True):
      """
      Versión simplificada de evaluación del modelo multitarea

      Args:
          model: Modelo entrenado
          test_dataset: Dataset de prueba
          log_to_mlflow: Si registrar métricas en MLflow

      Returns:
          dict: Diccionario con métricas básicas por tarea
      """

      from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
      import numpy as np
      from torch.utils.data import DataLoader
      import torch

      if not model:
          raise ValueError("El modelo no ha sido entrenado o cargado")

      # Crear dataloader para evaluación
      test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

      # Diccionarios para almacenar predicciones y etiquetas reales
      results = {}

      model.eval()
      with torch.no_grad():
          # Inicializar listas para cada tarea
          for task in self.task_labels:
              results[task] = {'predictions': [], 'labels': []}

          # Procesar cada batch
          for batch in test_dataloader:
              # Mover datos al dispositivo
              inputs = {
                  'input_ids': batch['input_ids'].to(self.device),
                  'attention_mask': batch['attention_mask'].to(self.device)
              }

              # Forward pass
              outputs = model['bert'](**inputs)
              pooled_output = outputs.last_hidden_state[:, 0, :]
              pooled_output = model['dropout'](pooled_output)

              # Obtener predicciones para cada tarea
              for task in self.task_labels:
                  logits = model[task](pooled_output)
                  predictions = torch.argmax(logits, dim=1)

                  results[task]['predictions'].extend(predictions.cpu().numpy())
                  results[task]['labels'].extend(batch[f'{task}_label'].numpy())

      # Calcular métricas para cada tarea
      evaluation_results = {}

      for task in self.task_labels:
          y_true = results[task]['labels']
          y_pred = results[task]['predictions']

          # Métricas básicas
          accuracy = accuracy_score(y_true, y_pred)

          # Precision, recall, f1 con manejo de errores
          try:
              precision, recall, f1, _ = precision_recall_fscore_support(
                  y_true, y_pred, average='weighted', zero_division=0
              )
          except:
              precision = recall = f1 = 0.0

          # Matriz de confusión
          try:
              cm = confusion_matrix(y_true, y_pred)
          except:
              cm = np.array([[0]])

          # Obtener nombres de clase de forma segura
          try:
              column_name = self.task_to_column[task]
              print("Column!")
              print(column_name)
              class_names = [str(name) for name in self.label_encoders[column_name].classes_]
          except:
              unique_labels = sorted(list(set(y_true + y_pred)))
              class_names = [f"class_{label}" for label in unique_labels]
              column_name = task

          # Calcular score de relevancia simple
          relevance_score = self._calculate_simple_relevance(accuracy, f1)

          evaluation_results[task] = {
              'column_name': column_name,
              'accuracy': round(accuracy, 4),
              'precision': round(precision, 4),
              'recall': round(recall, 4),
              'f1_score': round(f1, 4),
              'confusion_matrix': cm,
              'class_names': class_names,
              'relevance_score': relevance_score,
              'num_samples': len(y_true)
          }

      # Imprimir resumen simple
      self._print_simple_summary(evaluation_results)

      return evaluation_results


    def _calculate_simple_relevance(self, accuracy, f1):
        """
        Calcula un score de relevancia simple basado en accuracy y f1

        Returns:
            dict: Puntuación de relevancia simplificada
        """
        # Score simple: promedio de accuracy y f1
        score = (accuracy + f1) / 2

        # Determinar nivel de relevancia
        if score >= 0.8:
            level = "MUY ALTA"
        elif score >= 0.65:
            level = "ALTA"
        elif score >= 0.5:
            level = "MEDIA"
        elif score >= 0.35:
            level = "BAJA"
        else:
            level = "MUY BAJA"

        # Interpretación simple
        if accuracy < 0.5:
            interpretation = "⚠️ Precisión baja - Modelo tiene dificultades"
        elif accuracy > 0.8:
            interpretation = "✅ Excelente precisión"
        elif f1 < 0.5:
            interpretation = "⚠️ F1 bajo - Posible desbalance"
        else:
            interpretation = "📊 Rendimiento aceptable"

        return {
            'score': round(score, 4),
            'level': level,
            'interpretation': interpretation
        }


    def _print_simple_summary(self, evaluation_results):
        """
        Imprime un resumen simple de la evaluación
        """
        print("=" * 60)
        print("RESUMEN DE EVALUACIÓN DEL MODELO")
        print("=" * 60)

        for task, metrics in evaluation_results.items():
            print(f"\n📋 TAREA: {task.upper()}")
            print(f"   Campo: {metrics['column_name']}")
            print(f"   Muestras: {metrics['num_samples']}")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   F1-Score: {metrics['f1_score']:.3f}")
            print(f"   Relevancia: {metrics['relevance_score']['level']} ({metrics['relevance_score']['score']:.3f})")
            print(f"   {metrics['relevance_score']['interpretation']}")
            print(f"   Clases: {len(metrics['class_names'])} ({', '.join(metrics['class_names'][:3])}{'...' if len(metrics['class_names']) > 3 else ''})")

        print("\n" + "=" * 60)

        # Resumen general
        avg_accuracy = np.mean([metrics['accuracy'] for metrics in evaluation_results.values()])
        avg_f1 = np.mean([metrics['f1_score'] for metrics in evaluation_results.values()])

        print(f"📊 RENDIMIENTO GENERAL:")
        print(f"   Accuracy promedio: {avg_accuracy:.3f}")
        print(f"   F1-Score promedio: {avg_f1:.3f}")

        # Mejores y peores tareas
        best_task = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['f1_score'])
        worst_task = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['f1_score'])

        print(f"   🥇 Mejor tarea: {best_task} (F1: {evaluation_results[best_task]['f1_score']:.3f})")
        print(f"   🔴 Peor tarea: {worst_task} (F1: {evaluation_results[worst_task]['f1_score']:.3f})")
        print("=" * 60)
