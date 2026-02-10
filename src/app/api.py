from fastapi import FastAPI, HTTPException
from src.app.core.schemas import APIResponse, Meta, Result, GenerateRequest
from src.app.core.dataset import generate_dataset
from src.app.routers import clean, eda, mv, ml, ml2

app = FastAPI(
    title="FastAPI Data Science Project",
    description="5-Phase Data Science API (Clean, EDA, Multivariate, ML Baseline, ML Advanced)",
    version="1.0.0"
)

# Include Routers
app.include_router(clean.router)
app.include_router(eda.router)
app.include_router(mv.router)
app.include_router(ml.router)
app.include_router(ml2.router)

@app.post("/dataset/generate", response_model=APIResponse, tags=["dataset"])
def generate_data(request: GenerateRequest):
    try:
        dataset_id, df = generate_dataset(request.phase, request.seed, request.n)
        return APIResponse(
            meta=Meta(dataset_id=dataset_id, params={"phase": request.phase, "seed": request.seed, "n": request.n}),
            result=Result(
                data=df.head(20).to_dict(orient="records"),
                metrics={"columns": list(df.columns), "shape": df.shape}
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI Data Science Project"}
