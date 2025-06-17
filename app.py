from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import threading
import pandas as pd
import os
import json
import logging
from datetime import datetime
import traceback

# Importar la clase del modelo (asumiendo que está en un archivo llamado multitask_model.py)
from multitask_model import MultiTaskFeedbackModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)

# Configuración
MODEL_DIR = "resources/model"
TEMP_DATA_DIR = "resources/temp"

# Asegurar que los directorios existen
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Variable global para mantener el modelo
model_instance = None

# Variable global para mantener el estado del entrenamiento
training_status = {
    'is_training': False,
    'status': 'idle',
    'progress': 0,
    'error': None,
    'start_time': None,
    'end_time': None
}

def async_train(data, temp_file):

    """Función que ejecuta el entrenamiento en background"""
    global model_instance, training_status
    try:
        training_status['is_training'] = True
        training_status['status'] = 'training'
        training_status['start_time'] = datetime.now().isoformat()
        
        # Inicializar y entrenar modelo
        model_instance = MultiTaskFeedbackModel()
        
        # Cargar y preparar datos
        dataset = model_instance.load_and_prepare_data(temp_file)
        
        # Obtener parámetros de entrenamiento
        epochs = data.get('epochs', 3)
        batch_size = data.get('batch_size', 8)
        
        # Entrenar modelo
        model_instance.train(dataset, epochs=epochs, batch_size=batch_size)
        
        # Guardar modelo entrenado
        model_instance.save(MODEL_DIR)
        
        # Limpiar archivo temporal
        os.remove(temp_file)
        
        training_status['status'] = 'completed'
        training_status['progress'] = 100
        logger.info("Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        logger.error(traceback.format_exc())
        training_status['status'] = 'error'
        training_status['error'] = str(e)
    finally:
        training_status['is_training'] = False
        training_status['end_time'] = datetime.now().isoformat()

class ModelTrainingResource(Resource):
    """Endpoint para entrenar el modelo con datos JSON de forma asíncrona"""
    
    def post(self):
        try:
            global training_status
            
            # Verificar si ya hay un entrenamiento en curso
            if training_status['is_training']:
                return {
                    'error': 'Ya existe un entrenamiento en curso',
                    'status': 'error',
                    'current_status': training_status
                }, 409
            
            # Validar que se recibió JSON
            if not request.is_json:
                return {
                    'error': 'Content-Type debe ser application/json',
                    'status': 'error'
                }, 400
            
            data = request.get_json()
            
            # Validar estructura básica
            if 'data' not in data:
                return {
                    'error': 'JSON debe contener una clave "data" con los registros',
                    'status': 'error'
                }, 400
            
            training_data = data['data']
            
            # Validar que data es una lista
            if not isinstance(training_data, list):
                return {
                    'error': 'El campo "data" debe ser una lista de registros',
                    'status': 'error'
                }, 400
            
            # Validar que hay datos
            if len(training_data) == 0:
                return {
                    'error': 'No se proporcionaron datos para entrenar',
                    'status': 'error'
                }, 400
            
            # Validar estructura de los registros
            required_columns = ['feedback', 'sentiment_score', 'priority', 'category', 'team_assigned']
            first_record = training_data[0]
            missing_columns = [col for col in required_columns if col not in first_record]
            
            if missing_columns:
                return {
                    'error': f'Columnas faltantes en los datos: {missing_columns}',
                    'required_columns': required_columns,
                    'status': 'error'
                }, 400
            
            logger.info(f"Iniciando entrenamiento con {len(training_data)} registros")
            
            # Crear DataFrame temporal
            df = pd.DataFrame(training_data)
            
            # Guardar temporalmente como CSV para usar con el modelo existente
            temp_file = os.path.join(TEMP_DATA_DIR, f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(temp_file, index=False)
            
            # Iniciar entrenamiento en thread separado
            training_thread = threading.Thread(
                target=async_train,
                args=(data, temp_file)
            )
            training_thread.start()
            
            return {
                'message': 'Entrenamiento iniciado en background',
                'status': 'accepted',
                'training_status': training_status
            }, 202
            
        except Exception as e:
            logger.error(f"Error al iniciar entrenamiento: {str(e)}")
            return {
                'error': f'Error al iniciar entrenamiento: {str(e)}',
                'status': 'error'
            }, 500

class ModelPredictionResource(Resource):
    """Endpoint para realizar predicciones"""
    
    def post(self):
        try:
            global model_instance
            
            # Validar que se recibió JSON
            if not request.is_json:
                return {
                    'error': 'Content-Type debe ser application/json',
                    'status': 'error'
                }, 400
            
            data = request.get_json()
            
            # Validar que se proporcionó texto
            if 'text' not in data:
                return {
                    'error': 'Debe proporcionar un campo "text" con el feedback a analizar',
                    'status': 'error'
                }, 400
            
            text = data['text']
            
            if not text or not text.strip():
                return {
                    'error': 'El texto no puede estar vacío',
                    'status': 'error'
                }, 400
            
            # Cargar modelo si no está en memoria
            if model_instance is None:
                if not os.path.exists(os.path.join(MODEL_DIR, "model_weights.pt")):
                    return {
                        'error': 'No se encontró un modelo entrenado. Entrene el modelo primero.',
                        'status': 'error'
                    }, 404
                
                model_instance = MultiTaskFeedbackModel()
                model_instance.load(MODEL_DIR)
                logger.info("Modelo cargado desde disco")
            
            # Realizar predicción
            predictions = model_instance.predict(text)
            
            logger.info(f"Predicción realizada para texto: {text[:50]}...")
            
            return {
                'status': 'success',
                'input_text': text,
                'predictions': predictions,
                'timestamp': datetime.now().isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Error durante la predicción: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': f'Error durante la predicción: {str(e)}',
                'status': 'error'
            }, 500


class ModelEvaluationResource(Resource):
    """Endpoint para evaluación del modelo"""
    
    def post(self):
        try:
            global model_instance
            
            # Validar que se recibió JSON
            if not request.is_json:
                return {
                    'error': 'Content-Type debe ser application/json',
                    'status': 'error'
                }, 400
            
            data = request.get_json()
            
            # Validar estructura básica para datos de test
            if 'test_data' not in data:
                return {
                    'error': 'JSON debe contener una clave "test_data" con los datos de evaluación',
                    'status': 'error'
                }, 400
            
            test_data = data['test_data']
            
            if not isinstance(test_data, list) or len(test_data) == 0:
                return {
                    'error': 'test_data debe ser una lista no vacía de registros',
                    'status': 'error'
                }, 400
            
            # Cargar modelo si no está en memoria
            if model_instance is None:
                if not os.path.exists(os.path.join(MODEL_DIR, "model_weights.pt")):
                    return {
                        'error': 'No se encontró un modelo entrenado. Entrene el modelo primero.',
                        'status': 'error'
                    }, 404
                
                model_instance = MultiTaskFeedbackModel()
                model_instance.load(MODEL_DIR)
                logger.info("Modelo cargado desde disco para evaluación")
            
            # Crear DataFrame temporal para datos de test
            df_test = pd.DataFrame(test_data)
            
            # Validar columnas necesarias
            required_columns = ['feedback', 'sentiment_score', 'priority', 'category', 'team_assigned']
            missing_columns = [col for col in required_columns if col not in df_test.columns]
            
            if missing_columns:
                return {
                    'error': f'Columnas faltantes en test_data: {missing_columns}',
                    'required_columns': required_columns,
                    'status': 'error'
                }, 400
            
            # Preparar datos de test usando los encoders existentes
            temp_test_file = os.path.join(TEMP_DATA_DIR, f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df_test.to_csv(temp_test_file, index=False)
            
            # Crear dataset de test
            test_dataset = model_instance.load_and_prepare_data(temp_test_file)
            
            # Evaluar modelo
            evaluation_results = model_instance.evaluate_model(model_instance.model, test_dataset)
            
            # Generar resumen de evaluación
            summary = model_instance.model_evaluation_summary(evaluation_results)
            
            # Limpiar archivo temporal
            os.remove(temp_test_file)
            
            logger.info("Evaluación del modelo completada")
            
            # Convertir matrices de confusión a listas para JSON
            serializable_results = {}
            for task, metrics in evaluation_results.items():
                serializable_results[task] = {
                    'column_name': metrics['column_name'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'confusion_matrix': metrics['confusion_matrix'].tolist(),
                    'class_names': metrics['class_names'],
                    'relevance_score': metrics['relevance_score'],
                    'num_samples': metrics['num_samples']
                }
            
            return {
                'status': 'success',
                'evaluation_results': serializable_results,
                'summary': summary,
                'test_samples': len(test_data),
                'timestamp': datetime.now().isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Error durante la evaluación: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': f'Error durante la evaluación: {str(e)}',
                'status': 'error'
            }, 500


class ModelStatusResource(Resource):
    """Endpoint para verificar el estado del modelo"""
    
    def get(self):
        try:
            model_exists = os.path.exists(os.path.join(MODEL_DIR, "model_weights.pt"))
            encoders_exist = os.path.exists(os.path.join(MODEL_DIR, "label_encoders.pkl"))
            
            global model_instance
            model_loaded = model_instance is not None
            
            return {
                'status': 'success',
                'model_status': {
                    'model_file_exists': model_exists,
                    'encoders_file_exists': encoders_exist,
                    'model_loaded_in_memory': model_loaded,
                    'model_ready': model_exists and encoders_exist,
                    'model_directory': MODEL_DIR
                },
                'timestamp': datetime.now().isoformat()
            }, 200
            
        except Exception as e:
            logger.error(f"Error verificando estado del modelo: {str(e)}")
            return {
                'error': f'Error verificando estado: {str(e)}',
                'status': 'error'
            }, 500


class ModelTrainingStatusResource(Resource):
    """Endpoint para consultar el estado del entrenamiento"""
    
    
    def get(self):
        return {
            'status': 'success',
            'training_status': training_status
        }, 200


# Registrar endpoints
api.add_resource(ModelTrainingResource, '/api/model/train')
api.add_resource(ModelPredictionResource, '/api/model/predict')
api.add_resource(ModelEvaluationResource, '/api/model/evaluate')
api.add_resource(ModelStatusResource, '/api/model/status')
api.add_resource(ModelTrainingStatusResource, '/api/model/train/status')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint básico de salud"""
    return jsonify({
        'status': 'healthy',
        'service': 'MultiTask Feedback Model API',
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint no encontrado',
        'status': 'error',
        'available_endpoints': [
            'POST /api/model/train',
            'POST /api/model/predict', 
            'POST /api/model/evaluate',
            'GET /api/model/status',
            'GET /api/health'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Error interno del servidor',
        'status': 'error'
    }), 500


if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5000)