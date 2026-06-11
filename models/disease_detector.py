# models/disease_detector.py

import numpy as np
from PIL import Image
import io
from typing import Dict
import tensorflow as tf

class DiseaseDetector:
    def __init__(self):
        # Pour l'instant, modèle simulé
        # TODO: Charger un vrai modèle entraîné
        self.model = None
        self.classes = [
            'mildew',      # Mildiou
            'rust',        # Rouille
            'leafSpot',    # Taches foliaires
            'yellowing',   # Jaunissement
            'blight',      # Brûlure
            'healthy'      # Sain
        ]
        
    def predict(self, image_bytes: bytes) -> Dict:
        """Prédiction CNN sur image"""
        
        # Charger l'image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        
        # Simulation de prédiction
        # TODO: Remplacer par vrai modèle CNN
        probabilities = np.random.dirichlet(np.ones(6))
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Recommandations
        recommendations = self._get_recommendations(predicted_class)
        
        # Probabilités par maladie
        all_probabilities = {
            cls: float(prob) 
            for cls, prob in zip(self.classes, probabilities)
        }
        
        return {
            'disease': predicted_class,
            'confidence': confidence,
            'imagePath': 'uploaded_image',
            'detectedAt': '2025-06-12T10:30:00',
            'recommendations': recommendations,
            'allProbabilities': all_probabilities
        }
    
    def _get_recommendations(self, disease: str):
        recommendations_db = {
            'mildew': [
                {
                    'title': 'Traitement Fongicide Urgent',
                    'description': 'Le mildiou se propage rapidement.',
                    'treatment': 'Appliquer bouillie bordelaise.',
                    'isUrgent': True
                }
            ],
            'rust': [
                {
                    'title': 'Traitement Fongicide Systémique',
                    'description': 'La rouille affecte la photosynthèse.',
                    'treatment': 'Utiliser Tébuconazole.',
                    'isUrgent': True
                }
            ],
            'healthy': [
                {
                    'title': 'Maintien de la Santé',
                    'description': 'Plante en bonne santé.',
                    'treatment': 'Continuer surveillance.',
                    'isUrgent': False
                }
            ]
        }
        
        return recommendations_db.get(disease, [])