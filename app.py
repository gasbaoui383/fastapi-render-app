from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from models.pso_optimizer import PSOOptimizer
from models.ga_optimizer import GAOptimizer
from models.aco_optimizer import ACOOptimizer
from models.nsga2_optimizer import NSGA2Optimizer
from models.fuzzy_controller import FuzzyController
from models.disease_detector import DiseaseDetector

app = FastAPI(
    title="GAI-MADT API",
    description="Backend API pour Irrigation Intelligente",
    version="1.0.0"
)

# CORS pour Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════
# MODELS PYDANTIC
# ═══════════════════════════════════════════════════════════════════

class OptimizationParams(BaseModel):
    population_size: int = 30
    max_iterations: int = 50
    mutation_rate: float = 0.02
    crossover_rate: float = 0.8

class FuzzyInput(BaseModel):
    soil_moisture: float
    temperature: float
    wind_speed: float
    plant_health: float

# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
def read_root():
    return {
        "message": "GAI-MADT API",
        "version": "1.0.0",
        "status": "operational"
    }

# ─── PSO ──────────────────────────────────────────────────────────

@app.post("/optimize/pso")
def optimize_pso(params: OptimizationParams):
    optimizer = PSOOptimizer(
        population_size=params.population_size,
        max_iterations=params.max_iterations
    )
    result = optimizer.optimize()
    return result

# ─── GA ───────────────────────────────────────────────────────────

@app.post("/optimize/ga")
def optimize_ga(params: OptimizationParams):
    optimizer = GAOptimizer(
        population_size=params.population_size,
        max_iterations=params.max_iterations,
        mutation_rate=params.mutation_rate,
        crossover_rate=params.crossover_rate
    )
    result = optimizer.optimize()
    return result

# ─── ACO ──────────────────────────────────────────────────────────

@app.post("/optimize/aco")
def optimize_aco(params: OptimizationParams):
    optimizer = ACOOptimizer(
        population_size=params.population_size,
        max_iterations=params.max_iterations
    )
    result = optimizer.optimize()
    return result

# ─── NSGA-II ──────────────────────────────────────────────────────

@app.post("/optimize/nsga2")
def optimize_nsga2(params: OptimizationParams):
    optimizer = NSGA2Optimizer(
        population_size=params.population_size,
        max_iterations=params.max_iterations
    )
    result = optimizer.optimize()
    return result

# ─── FUZZY LOGIC ──────────────────────────────────────────────────

@app.post("/fuzzy/calculate")
def fuzzy_calculate(input_data: FuzzyInput):
    controller = FuzzyController()
    result = controller.calculate_irrigation(
        soil_moisture=input_data.soil_moisture,
        temperature=input_data.temperature,
        wind_speed=input_data.wind_speed,
        plant_health=input_data.plant_health
    )
    return result

# ─── CNN DISEASE DETECTION ────────────────────────────────────────

@app.post("/cnn/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    detector = DiseaseDetector()
    
    # Lire l'image
    contents = await file.read()
    
    # Détecter
    result = detector.predict(contents)
    
    return result

# ─── HEALTH CHECK ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "services": {
            "pso": "operational",
            "ga": "operational",
            "aco": "operational",
            "nsga2": "operational",
            "fuzzy": "operational",
            "cnn": "operational"
        }
    }

# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )