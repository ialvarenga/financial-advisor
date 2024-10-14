# src/api/routes/queries.py
from fastapi import APIRouter, HTTPException
from src.api.schemas.query_request import QueryRequest
from src.core.intent_classifier import classify_intent
from src.core.llama_index_handler import process_financial_query

router = APIRouter()

@router.post("/query", summary="Handle a natural language query")
async def handle_query(request: QueryRequest):
    # Classify the intent of the query
    intent = classify_intent(request.query)
    
    if intent == "financial":
        # Process financial queries using Llama Index
        result = process_financial_query(request.query)
        return {"intent": intent, "result": result}
    else:
        raise HTTPException(status_code=400, detail="Unsupported query type")
