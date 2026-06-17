from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(title="AstroSentinel Local API")

# Allow Chrome Extension to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import json

BOT_SET = set()
HANDLE_MAP = {}

@app.on_event("startup")
def load_data():
    global BOT_SET, HANDLE_MAP
    print("Loading bot database...")
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "labeled_data.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            bots = df[df["is_bot"] == 1]["author_id"].dropna().astype(str).tolist()
            BOT_SET = set(bots)
            print(f"✅ Loaded {len(BOT_SET)} known bot accounts.")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            
    map_path = os.path.join(os.path.dirname(__file__), "handle_map.json")
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            HANDLE_MAP = json.load(f)
        print(f"✅ Loaded {len(HANDLE_MAP)} handle mappings.")

@app.get("/check_bot/{author_id}")
def check_bot(author_id: str):
    """
    Check if a specific YouTube author_id is in our known bots database.
    """
    # If the extension sends a handle (e.g. @Username), resolve it to UC...
    if author_id.startswith('@'):
        resolved_id = HANDLE_MAP.get(author_id, author_id)
    else:
        resolved_id = author_id
        
    is_bot = resolved_id in BOT_SET
    return {"author_id": resolved_id, "is_bot": is_bot}

if __name__ == "__main__":
    import uvicorn
    # Runs the local API server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
