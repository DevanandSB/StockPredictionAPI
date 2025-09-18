from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

# --- App Setup ---
# Create the main FastAPI application instance for the intro page
app = FastAPI(
    title="StockSense AI Intro",
    description="Serves the galaxy-themed introduction page for StockSense AI.",
    version="1.0.0"
)

# --- Template Configuration ---
# Get the absolute path to the current file's directory
# This makes sure the app can find the 'intro.html' file
# no matter how it's run.
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=current_dir)


# --- API Route (Endpoint) ---

@app.get("/", response_class=HTMLResponse, tags=["Introduction"])
async def read_intro(request: Request):
    """
    This is the main endpoint that serves the galaxy intro page.
    When App Platform routes traffic from 'intro.stocksense.tech'
    to this component, this function will handle the root '/' request.
    """
    # The template response will look for 'intro.html' in the same
    # directory as this main.py file.
    return templates.TemplateResponse("intro.html", {"request": request})

# Note: When deploying to App Platform, your run command for this
# component should be something like:
# uvicorn main:app --host 0.0.0.0 --port $PORT