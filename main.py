# ================================
# Trip and Travel Agent - FastAPI
# ================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import asyncio

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    class ChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            pass
        def invoke(self, prompt):
            class R: pass
            r = R()
            r.content = "[stubbed response] This is a mock response because the langchain_google_genai package is not installed."
            return r
        async def ainvoke(self, prompt):
            class R: pass
            r = R()
            r.content = "[stubbed async response] This is a mock response."
            return r

# ================================
# Load Environment Variables
# ================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# ================================
# Initialize FastAPI App
# ================================

app = FastAPI(title="Trip and Travel Agent")

# ================================
# Pydantic Models
# ================================

class TravelRequest(BaseModel):
    destination: str
    theme: str
    days: int


class ModeRequest(BaseModel):
    factual_prompt: str | None = None
    creative_prompt: str | None = None


# ================================
# LLM Configurations
# ================================

# Factual / Precise Model (Low temperature)
factual_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    max_output_tokens=512
)

# Creative Model (High temperature)
creative_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.9,
    max_output_tokens=1024
)

# ================================
# TASK 1: Baseline Itinerary Generator (Synchronous)
# ================================

@app.post("/task1")
def generate_itinerary(request: TravelRequest):
    try:
        prompt = f"""
        Create a {request.days}-day travel itinerary for {request.destination}.
        Theme: {request.theme}.
        Provide a day-wise plan with places to visit.
        """

        response = factual_llm.invoke(prompt)

        return {
            "destination": request.destination,
            "theme": request.theme,
            "days": request.days,
            "itinerary": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# TASK 2: Factual & Creative Modes
# ================================

@app.post("/task2")
def factual_and_creative_modes(request: ModeRequest):
    try:
        result = {}

        if request.factual_prompt:
            factual_response = factual_llm.invoke(
                f"Answer factually and concisely: {request.factual_prompt}"
            )
            result["factual_response"] = factual_response.content

        if request.creative_prompt:
            creative_response = creative_llm.invoke(
                f"Answer creatively with imagination: {request.creative_prompt}"
            )
            result["creative_response"] = creative_response.content

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# TASK 3: Asynchronous Suggestion API
# ================================

async def async_llm_call(prompt: str):
    async_llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.5
    )
    return await async_llm.ainvoke(prompt)


@app.get("/task3")
async def async_suggestion(question: str):
    try:
        response = await async_llm_call(
            f"Answer this travel-related question: {question}"
        )

        return {
            "question": question,
            "response": response.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# Local Execution
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)