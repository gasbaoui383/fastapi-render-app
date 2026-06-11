# models/fuzzy_controller.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict

class FuzzyController:
    def __init__(self):
        # Variables d'entrée
        self.soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
        self.temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
        self.wind_speed = ctrl.Antecedent(np.arange(0, 51, 1), 'wind_speed')
        
        # Variable de sortie
        self.water_amount = ctrl.Consequent(np.arange(0, 11, 0.1), 'water_amount')
        
        # Fonctions d'appartenance pour humidité
        self.soil_moisture['dry'] = fuzz.trapmf(self.soil_moisture.universe, [0, 0, 20, 40])
        self.soil_moisture['medium'] = fuzz.trimf(self.soil_moisture.universe, [30, 50, 70])
        self.soil_moisture['wet'] = fuzz.trapmf(self.soil_moisture.universe, [60, 80, 100, 100])
        
        # Fonctions d'appartenance pour température
        self.temperature['cold'] = fuzz.trapmf(self.temperature.universe, [0, 0, 15, 25])
        self.temperature['moderate'] = fuzz.trimf(self.temperature.universe, [20, 30, 40])
        self.temperature['hot'] = fuzz.trapmf(self.temperature.universe, [35, 45, 50, 50])
        
        # Fonctions d'appartenance pour vent
        self.wind_speed['low'] = fuzz.trapmf(self.wind_speed.universe, [0, 0, 10, 20])
        self.wind_speed['medium'] = fuzz.trimf(self.wind_speed.universe, [15, 25, 35])
        self.wind_speed['high'] = fuzz.trapmf(self.wind_speed.universe, [30, 40, 50, 50])
        
        # Fonctions d'appartenance pour sortie
        self.water_amount['none'] = fuzz.trapmf(self.water_amount.universe, [0, 0, 0.5, 1])
        self.water_amount['low'] = fuzz.trimf(self.water_amount.universe, [0.5, 2, 4])
        self.water_amount['medium'] = fuzz.trimf(self.water_amount.universe, [3, 5, 7])
        self.water_amount['high'] = fuzz.trimf(self.water_amount.universe, [6, 8, 10])
        self.water_amount['maximum'] = fuzz.trapmf(self.water_amount.universe, [9, 10, 10, 10])
        
        # Règles floues
        self.create_rules()
        
    def create_rules(self):
        """Créer les règles d'inférence"""
        
        rule1 = ctrl.Rule(
            self.soil_moisture['dry'] & self.temperature['hot'],
            self.water_amount['maximum']
        )
        
        rule2 = ctrl.Rule(
            self.soil_moisture['dry'] & self.temperature['moderate'],
            self.water_amount['high']
        )
        
        rule3 = ctrl.Rule(
            self.soil_moisture['dry'] & self.temperature['cold'],
            self.water_amount['medium']
        )
        
        rule4 = ctrl.Rule(
            self.soil_moisture['medium'] & self.temperature['hot'],
            self.water_amount['medium']
        )
        
        rule5 = ctrl.Rule(
            self.soil_moisture['medium'] & self.temperature['moderate'],
            self.water_amount['low']
        )
        
        rule6 = ctrl.Rule(
            self.soil_moisture['medium'] & self.temperature['cold'],
            self.water_amount['low']
        )
        
        rule7 = ctrl.Rule(
            self.soil_moisture['wet'] & self.temperature['hot'],
            self.water_amount['low']
        )
        
        rule8 = ctrl.Rule(
            self.soil_moisture['wet'] & self.temperature['moderate'],
            self.water_amount['none']
        )
        
        rule9 = ctrl.Rule(
            self.soil_moisture['wet'] & self.temperature['cold'],
            self.water_amount['none']
        )
        
        rule10 = ctrl.Rule(
            self.wind_speed['high'],
            self.water_amount['high']
        )
        
        # Système de contrôle
        self.irrigation_ctrl = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5,
            rule6, rule7, rule8, rule9, rule10
        ])
        
        self.irrigation = ctrl.ControlSystemSimulation(self.irrigation_ctrl)
        
    def calculate_irrigation(self, soil_moisture: float, temperature: float,
                             wind_speed: float, plant_health: float) -> Dict:
        """Calcul de l'irrigation via logique floue"""
        
        # Définir les entrées
        self.irrigation.input['soil_moisture'] = soil_moisture
        self.irrigation.input['temperature'] = temperature
        self.irrigation.input['wind_speed'] = wind_speed
        
        # Calculer
        self.irrigation.compute()
        
        # Résultat
        water_output = float(self.irrigation.output['water_amount'])
        
        # Ajustement selon santé plante
        if plant_health < 50:
            water_output *= 0.8
        
        # Générer raisonnement
        reasoning = self._generate_reasoning(
            soil_moisture, temperature, wind_speed, plant_health, water_output
        )
        
        # Règles activées (simulé)
        rules_activated = {
            'R1': float(np.random.uniform(0, 1)),
            'R2': float(np.random.uniform(0, 1)),
            'R4': float(np.random.uniform(0, 1)),
        }
        
        return {
            'waterAmount': water_output,
            'reasoning': reasoning,
            'ruleActivations': rules_activated,
            'calculatedAt': '2025-06-12T10:30:00'
        }
    
    def _generate_reasoning(self, moisture, temp, wind, health, water):
        reasons = []
        
        if moisture < 40:
            reasons.append(f'Sol détecté comme SEC ({moisture:.0f}%)')
        elif moisture > 70:
            reasons.append(f'Sol détecté comme HUMIDE ({moisture:.0f}%)')
        else:
            reasons.append(f'Sol détecté comme MOYEN ({moisture:.0f}%)')
        
        if temp > 35:
            reasons.append('Température ÉLEVÉE → évaporation importante')
        elif temp < 20:
            reasons.append('Température BASSE → évaporation réduite')
        
        if wind > 30:
            reasons.append(f'Vent FORT détecté → compensation évaporation')
        
        if health < 50:
            reasons.append('Plante en mauvaise santé → irrigation réduite')
        
        reasons.append(f'{len(self.irrigation_ctrl.rules)} règles floues évaluées')
        
        return ' • '.join(reasons)