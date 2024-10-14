# src/api/main.py
from fastapi import FastAPI
from src.api.routes import queries

app = FastAPI(title="Query API")

# Include the queries route
app.include_router(queries.router, prefix="/api/v1")
