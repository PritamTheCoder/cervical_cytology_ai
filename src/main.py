import shutil
import uuid
import logging
from contextlib import asynccontextmanager
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Import logic
from pipeline import InferencePipeline 
from config import segmentation_config, report_config

# --- Lifespan Management ---
ai_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_pipeline
    logging.info(" Initializing AI Models (Cellpose & MobileViT)...")
    ai_pipeline = InferencePipeline()
    logging.info("[OK] Pipeline Ready.")
    yield
    logging.info("[X] Shutting down AI System.")

app = FastAPI(
    title="Cervical Cytology AI System", 
    version="2.0",
    lifespan=lifespan
)

# --- Middleware & Static Serving ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves files from the 'outputs' directory
# A file at outputs/reports/abc.pdf becomes http://localhost:8000/download/reports/abc.pdf
app.mount("/download", StaticFiles(directory="data"), name="data")

# --- Health Check & Root ---
@app.get("/")
async def root():
    return {"status": "online", "system": "Cytology-AI", "version": "2.0"}

# --- The Main Analysis Endpoint ---
@app.post("/analyze-slide/")
async def analyze_slide(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    return_pdf_directly: bool = False # Query param to toggle response type
):
    slide_id = str(uuid.uuid4())[:8]
    save_dir = segmentation_config.INPUT_DIR 
    saved_paths = []

    try:
        # Save files
        for file in files:
            if not file.filename or not file.filename.lower().endswith(('.bmp', '.jpg', '.png')):
                continue
            
            safe_name = f"{slide_id}_{file.filename}"
            file_path = save_dir / safe_name
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid images.")

        # result should contain: {"pdf_path": "...", "risk_flag": "...", etc.}
        if ai_pipeline is None:
            raise HTTPException(status_code=500, detail="AI Pipeline not initialized.")
        result = ai_pipeline.run_slide_analysis(slide_id, saved_paths)
        
        # --- Cleanup Task ---
        # We delete the RAW input images (BMPs, jpgs, pngs) immediately after the pipeline finishes
        # to prevent disk bloat. We do NOT delete the PDF yet.
        background_tasks.add_task(cleanup_inputs, saved_paths)

        # --- Delivery Logic ---
        if return_pdf_directly:
            # Option A: Send the actual PDF file bytes
            return FileResponse(
                path=result["pdf_local_path"], 
                media_type='application/pdf',
                filename=f"Report_{slide_id}.pdf"
            )
        
        # Option B: Return JSON with the link (Best for Dashboards)
        result["pdf_url"] = f"http://localhost:8000/download/reports/{slide_id}_report.pdf"
        return result

    except Exception as e:
        # Immediate cleanup on error
        cleanup_inputs(saved_paths)
        logging.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_inputs(paths: List[Path]):
    """Removes the source BMP images to save space."""
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            logging.error(f"Cleanup failed for {p}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)