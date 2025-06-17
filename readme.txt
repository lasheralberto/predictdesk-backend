# Container de Docker:
cf push sentimentapp --docker-image lasheralberto/sentiment-analysis-container:tagname

# API de Análisis de Sentimientos

Esta API Flask utiliza la clase `SentimentModelService` para proporcionar análisis de sentimientos a través de endpoints REST.


## 🚀 Instalación y Ejecución

### Dependencias
```bash
pip install flask flask-restful torch transformers datasets evaluate
```

### Estructura de Archivos
```
proyecto/
├── app.py                 # API Flask
├── sentiment_service.py   # Clase SentimentModelService
├── resources/            # Carpeta con modelo y configuraciones
│   ├── model/           # Modelo entrenado
│   ├── config/          # Configuraciones y etiquetas
│   ├── data/            # Datasets
│   └── logs/            # Logs de entrenamiento
└── requirements.txt     # Dependencias
```

### Ejecutar la API
```bash
python app.py
```

La API estará disponible en `http://localhost:5000`

## 📋 Endpoints Disponibles

### 1. Health Check
**GET** `/health`

Verifica el estado de la API y si el modelo está cargado.

```bash
curl -X GET http://localhost:5000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "message": "API funcionando correctamente",
  "model_loaded": true,
  "resources_dir": "resources"
}
```

### 2. Información del Modelo
**GET** `/model/info`

Obtiene información detallada del modelo cargado.

```bash
curl -X GET http://localhost:5000/model/info
```

**Respuesta:**
```json
{
  "model_loaded": true,
  "config": {
    "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "learning_rate": 2e-5,
    "num_train_epochs": 3
  },
  "labels": {
    "0": "Muy enfadado",
    "1": "Enfadado",
    "2": "Neutro",
    "3": "Satisfecho",
    "4": "Muy satisfecho"
  },
  "evaluation_metrics": {
    "eval_accuracy": 0.95
  }
}
```

### 3. Predicción de Sentimientos
**POST** `/predict`

Endpoint principal para análisis de sentimientos. Soporta múltiples formatos de entrada.

#### Formato 1: Texto Individual
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Este producto es increíble, estoy muy satisfecho"
  }'
```

**Respuesta:**
```json
{
  "success": true,
  "predictions": {
    "text": "Este producto es increíble, estoy muy satisfecho",
    "label_id": 4,
    "label_name": "Muy satisfecho",
    "confidence": 0.95,
    "all_probabilities": {
      "Muy enfadado": 0.01,
      "Enfadado": 0.02,
      "Neutro": 0.02,
      "Satisfecho": 0.15,
      "Muy satisfecho": 0.95
    }
  },
  "total_processed": 1
}
```

#### Formato 2: Lista de Textos
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Excelente servicio al cliente",
      "Producto defectuoso, muy molesto",
      "Normal, nada especial"
    ]
  }'
```

**Respuesta:**
```json
{
  "success": true,
  "predictions": [
    {
      "text": "Excelente servicio al cliente",
      "label_id": 4,
      "label_name": "Muy satisfecho",
      "confidence": 0.92
    },
    {
      "text": "Producto defectuoso, muy molesto",
      "label_id": 0,
      "label_name": "Muy enfadado",
      "confidence": 0.88
    },
    {
      "text": "Normal, nada especial",
      "label_id": 2,
      "label_name": "Neutro",
      "confidence": 0.76
    }
  ],
  "total_processed": 3
}
```

#### Formato 3: Datos con ID
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "id": "review_001",
        "text": "Me encanta este producto"
      },
      {
        "id": "review_002", 
        "text": "Terrible experiencia de compra"
      }
    ]
  }'
```

**Respuesta:**
```json
{
  "success": true,
  "predictions": [
    {
      "text": "Me encanta este producto",
      "label_id": 4,
      "label_name": "Muy satisfecho",
      "confidence": 0.89,
      "id": "review_001",
      "all_probabilities": {...}
    },
    {
      "text": "Terrible experiencia de compra",
      "label_id": 0,
      "label_name": "Muy enfadado", 
      "confidence": 0.94,
      "id": "review_002",
      "all_probabilities": {...}
    }
  ],
  "total_processed": 2
}
```

### 4. Entrenar Modelo (Opcional)
**POST** `/train`

Endpoint para entrenar un nuevo modelo (generalmente para administradores).

```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {"text": "Estoy muy feliz", "label": 4},
      {"text": "Producto terrible", "label": 0},
      {"text": "Es normal", "label": 2}
    ],
    "config": {
      "learning_rate": 2e-5,
      "num_train_epochs": 3
    }
  }'
```

## 🐍 Cliente Python

```python
import requests
import json

class SentimentAPIClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def health_check(self):
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_single(self, text):
        data = {"text": text}
        response = requests.post(
            f"{self.base_url}/predict",
            json=data
        )
        return response.json()
    
    def predict_multiple(self, texts):
        data = {"texts": texts}
        response = requests.post(
            f"{self.base_url}/predict",
            json=data  
        )
        return response.json()
    
    def predict_with_ids(self, data_items):
        data = {"data": data_items}
        response = requests.post(
            f"{self.base_url}/predict",
            json=data
        )
        return response.json()

# Ejemplo de uso
client = SentimentAPIClient()

# Verificar estado
print(client.health_check())

# Predicción individual
result = client.predict_single("Este producto es excelente")
print(result)

# Predicciones múltiples
texts = ["Me gusta", "No me gusta", "Está bien"]
results = client.predict_multiple(texts)
print(results)
```

## 🔧 Variables de Entorno

```bash
export FLASK_DEBUG=true          # Modo debug
export FLASK_HOST=0.0.0.0       # Host del servidor
export FLASK_PORT=5000          # Puerto del servidor
export RESOURCES_DIR=resources   # Directorio de recursos
```

## ⚠️ Manejo de Errores

La API retorna códigos de estado HTTP apropiados:

- **200**: Éxito
- **400**: Error de validación en los datos
- **404**: Endpoint no encontrado
- **405**: Método no permitido
- **500**: Error interno del servidor
- **503**: Servicio no disponible (modelo no cargado)

**Ejemplo de error:**
```json
{
  "error": "El campo 'text' debe ser una cadena no vacía"
}
```

## 🚀 Despliegue en Producción

Para producción, considera usar:

- **Gunicorn**: Servidor WSGI más robusto
- **Docker**: Containerización
- **Nginx**: Proxy reverso
- **Variables de entorno**: Para configuración

```bash
# Con Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📝 Notas Importantes

1. **Modelo Requerido**: Asegúrate de entrenar el modelo antes de usar la API
2. **Memoria**: El modelo se carga en memoria al iniciar la aplicación
3. **Concurrencia**: La API es thread-safe para predicciones
4. **Límites**: No hay límites de rate limiting implementados por defecto
5. **Logs**: Los logs se escriben en stdout por defecto