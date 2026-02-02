from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid
import asyncio
from app.services.transcription import TranscriptionService

app = FastAPI()

# Setup Upload Directory
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Services
# We will use lazy loading or startup event to load models
transcription_service = None

# Simple in-memory job store
# { job_id: { status: 'pending'|'processing'|'completed'|'failed', result: ..., error: ... } }
jobs = {}

@app.on_event("startup")
async def startup_event():
    global transcription_service
    # Point implementation to /app/models inside container
    models_path = os.getenv("MODELS_PATH", "models")
    transcription_service = TranscriptionService(models_dir=models_path)

def process_transcription(job_id: str, file_path: str):
    try:
        jobs[job_id]['status'] = 'processing'
        # Run synchronous transcription
        result = transcription_service.transcribe(file_path)
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = result
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/transcribe")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a WAV, MP3, or M4A file.")

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        jobs[job_id] = {'status': 'pending'}
        background_tasks.add_task(process_transcription, job_id, file_path)
        
        return JSONResponse(content={"status": "queued", "job_id": job_id})

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(content=jobs[job_id])

@app.get("/")
async def root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="he" dir="rtl">
    <head>
        <title>תמלול אודיו - WhisperX</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { text-align: center; }
            form { display: flex; flex-direction: column; gap: 10px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
            button { padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            pre { background: #f4f4f4; padding: 10px; overflow-x: auto; white-space: pre-wrap; }
            .spinner {
                display: none;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 10px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <h1>תמלול אודיו בעברית</h1>
        <form id="uploadForm">
            <label for="file">בחר קובץ אודיו (WAV, MP3, M4A):</label>
            <input type="file" id="file" name="file" accept=".wav,.mp3,.m4a" required>
            <button type="submit">התחל תמלול</button>
        </form>
        <div id="spinner" class="spinner"></div>
        <div id="status" style="text-align:center; display:none;">מעבד...</div>
        <div id="result"></div>
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const resultDiv = document.getElementById('result');
                const spinner = document.getElementById('spinner');
                const statusDiv = document.getElementById('status');
                
                resultDiv.innerHTML = '';
                statusDiv.style.display = 'block';
                statusDiv.textContent = 'מעלה קובץ...';
                spinner.style.display = 'block';
                
                try {
                    // Start job
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.status === 'queued') {
                        const jobId = data.job_id;
                        statusDiv.textContent = 'הקובץ התקבל. מתחיל עיבוד...';
                        pollJob(jobId);
                    } else {
                        spinner.style.display = 'none';
                        statusDiv.style.display = 'none';
                        resultDiv.innerHTML = `<p style="color:red">שגיאה: ${data.detail}</p>`;
                    }
                } catch (err) {
                    spinner.style.display = 'none';
                    statusDiv.style.display = 'none';
                    resultDiv.innerHTML = `<p style="color:red">שגיאה בתקשורת: ${err.message}</p>`;
                }
            };

            async function pollJob(jobId) {
                const resultDiv = document.getElementById('result');
                const spinner = document.getElementById('spinner');
                const statusDiv = document.getElementById('status');
                
                const interval = setInterval(async () => {
                    try {
                        const res = await fetch(`/job/${jobId}`);
                        const job = await res.json();
                        
                        if (job.status === 'completed') {
                            clearInterval(interval);
                            spinner.style.display = 'none';
                            statusDiv.style.display = 'none';
                            
                            let html = '<h2>תוצאה:</h2>';
                            if (job.result.segments) {
                                job.result.segments.forEach(seg => {
                                    html += `<p><strong>${seg.speaker || 'Unknown'} [${seg.start.toFixed(1)}-${seg.end.toFixed(1)}]:</strong> ${seg.text}</p>`;
                                });
                            } else {
                                html += `<pre>${JSON.stringify(job.result, null, 2)}</pre>`;
                            }
                            resultDiv.innerHTML = html;
                        } else if (job.status === 'failed') {
                            clearInterval(interval);
                            spinner.style.display = 'none';
                            statusDiv.style.display = 'none';
                            resultDiv.innerHTML = `<p style="color:red">נכשל: ${job.error}</p>`;
                        } else {
                            statusDiv.textContent = 'מעבד... אנא המתן (' + job.status + ')';
                        }
                    } catch (e) {
                        // ignore poll errors
                        console.error(e);
                    }
                }, 2000);
            }
        </script>
    </body>
    </html>
    """)
