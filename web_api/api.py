from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn
import logging
from contextlib import asynccontextmanager
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from inference import EventTimeInference

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = '/home/ubuntu/graph-transformer-exp/checkpoints/best_model.pt'
NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1.c6fskces27nt.us-east-1.neptune.amazonaws.com:8182"
HOST = "0.0.0.0"
PORT = 8000
MAX_WORKERS = 10
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
inference_engine: Optional[EventTimeInference] = None
executor: Optional[ThreadPoolExecutor] = None


# ============================================================================
# HTML Template for JSON Display (with escaped braces)
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header .meta {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge.success {{
            background: #10b981;
            color: white;
        }}
        
        .badge.error {{
            background: #ef4444;
            color: white;
        }}
        
        .badge.info {{
            background: #3b82f6;
            color: white;
        }}
        
        .badge.warning {{
            background: #f59e0b;
            color: white;
        }}
        
        .json-container {{
            background: #1e1e1e;
            padding: 30px;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            overflow-x: auto;
        }}
        
        .json-content {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 14px;
            line-height: 1.6;
            color: #d4d4d4;
        }}
        
        .json-key {{
            color: #9cdcfe;
            font-weight: bold;
        }}
        
        .json-string {{
            color: #ce9178;
        }}
        
        .json-number {{
            color: #b5cea8;
        }}
        
        .json-boolean {{
            color: #569cd6;
        }}
        
        .json-null {{
            color: #569cd6;
        }}
        
        .json-bracket {{
            color: #ffd700;
            font-weight: bold;
        }}
        
        .copy-button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 20px;
            transition: background 0.3s;
        }}
        
        .copy-button:hover {{
            background: #5568d3;
        }}
        
        .copy-button:active {{
            background: #4451b8;
        }}
        
        .copy-feedback {{
            display: inline-block;
            margin-left: 10px;
            color: #10b981;
            font-weight: bold;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .copy-feedback.show {{
            opacity: 1;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .stat-box {{
            background: #f3f4f6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1f2937;
            margin-top: 5px;
        }}
        
        .actions {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s;
        }}
        
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #5568d3;
        }}
        
        .btn-secondary {{
            background: #6b7280;
            color: white;
        }}
        
        .btn-secondary:hover {{
            background: #4b5563;
        }}
        
        pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 {title}</h1>
            <div class="meta">
                {badges}
            </div>
            {stats}
            <div class="actions">
                <button class="btn btn-primary copy-button" onclick="copyToClipboard()">
                    📋 Copy JSON
                </button>
                <span class="copy-feedback" id="copyFeedback">✓ Copied!</span>
                <a href="/docs" class="btn btn-secondary">📚 API Docs</a>
                <a href="/" class="btn btn-secondary">🏠 Home</a>
            </div>
        </div>
        
        <div class="json-container">
            <pre class="json-content" id="jsonContent">{json_content}</pre>
        </div>
    </div>
    
    <script>
        const rawJson = {raw_json};
        
        function copyToClipboard() {{
            const text = JSON.stringify(rawJson, null, 2);
            navigator.clipboard.writeText(text).then(() => {{
                const feedback = document.getElementById('copyFeedback');
                feedback.classList.add('show');
                setTimeout(() => {{
                    feedback.classList.remove('show');
                }}, 2000);
            }});
        }}
        
        // Syntax highlighting
        function syntaxHighlight(json) {{
            if (typeof json !== 'string') {{
                json = JSON.stringify(json, null, 2);
            }}
            json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return json.replace(/("(\\u[a-zA-Z0-9]{{4}}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {{
                let cls = 'json-number';
                if (/^"/.test(match)) {{
                    if (/:$/.test(match)) {{
                        cls = 'json-key';
                    }} else {{
                        cls = 'json-string';
                    }}
                }} else if (/true|false/.test(match)) {{
                    cls = 'json-boolean';
                }} else if (/null/.test(match)) {{
                    cls = 'json-null';
                }}
                return '<span class="' + cls + '">' + match + '</span>';
            }});
        }}
        
        // Apply syntax highlighting on load
        document.addEventListener('DOMContentLoaded', function() {{
            const content = document.getElementById('jsonContent');
            content.innerHTML = syntaxHighlight(rawJson);
        }});
    </script>
</body>
</html>
"""


# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    neptune_connected: bool
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============================================================================
# Helper Functions
# ============================================================================

def get_current_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_error_response(error_message: str, detail: Optional[str] = None) -> Dict:
    return {
        "error": error_message,
        "detail": detail,
        "timestamp": get_current_timestamp()
    }


def format_json_as_html(data: dict, title: str = "API Response", status: str = "success") -> str:
    """Format JSON data as beautiful HTML"""
    
    # Create badges
    badges = []
    if status == "success":
        badges.append('<span class="badge success">✓ Success</span>')
    elif status == "error":
        badges.append('<span class="badge error">✗ Error</span>')
    else:
        badges.append('<span class="badge info">ℹ Info</span>')
    
    # Add timestamp badge
    if 'timestamp' in data:
        badges.append(f'<span class="badge info">🕒 {data["timestamp"]}</span>')
    
    badges_html = '\n'.join(badges)
    
    # Create stats if available
    stats_html = ""
    if 'total' in data or 'successful' in data or 'failed' in data:
        stats = []
        if 'total' in data:
            stats.append(f'''
                <div class="stat-box">
                    <div class="stat-label">Total</div>
                    <div class="stat-value">{data['total']}</div>
                </div>
            ''')
        if 'successful' in data:
            stats.append(f'''
                <div class="stat-box" style="border-left-color: #10b981;">
                    <div class="stat-label">Successful</div>
                    <div class="stat-value" style="color: #10b981;">{data['successful']}</div>
                </div>
            ''')
        if 'failed' in data:
            stats.append(f'''
                <div class="stat-box" style="border-left-color: #ef4444;">
                    <div class="stat-label">Failed</div>
                    <div class="stat-value" style="color: #ef4444;">{data['failed']}</div>
                </div>
            ''')
        stats_html = f'<div class="stats">{"".join(stats)}</div>'
    
    # Format JSON content
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Create HTML
    html = HTML_TEMPLATE.format(
        title=title,
        badges=badges_html,
        stats=stats_html,
        json_content=json_str,
        raw_json=json.dumps(data)
    )
    
    return html


def should_return_html(request: Request, format_param: Optional[str] = None) -> bool:
    """Determine if HTML response should be returned"""
    # Check format parameter first
    if format_param:
        return format_param.lower() == 'html'
    
    # Check Accept header
    accept = request.headers.get('accept', '')
    if 'text/html' in accept and 'application/json' not in accept:
        return True
    
    # Check User-Agent (browser detection)
    user_agent = request.headers.get('user-agent', '').lower()
    is_browser = any(browser in user_agent for browser in ['mozilla', 'chrome', 'safari', 'edge', 'opera'])
    
    return is_browser


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, executor
    
    logger.info("Starting up Event Time Prediction API...")
    
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    logger.info(f"✓ Thread pool executor created with {MAX_WORKERS} workers")
    
    try:
        inference_engine = EventTimeInference(
            checkpoint_path=CHECKPOINT_PATH,
            neptune_endpoint=NEPTUNE_ENDPOINT
        )
        logger.info("✓ Inference engine initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize inference engine: {e}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    logger.info("Shutting down Event Time Prediction API...")
    
    if inference_engine:
        inference_engine.close()
        logger.info("✓ Inference engine closed")
    
    if executor:
        executor.shutdown(wait=True)
        logger.info("✓ Thread pool executor shut down")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Event Time Prediction API",
    description="API for predicting package event times using Graph Neural Networks",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root(request: Request, format: Optional[str] = Query(None, description="Response format: json or html")):
    """Root endpoint with API information"""
    data = {
        "name": "Event Time Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": get_current_timestamp(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict_single": "/predict/{package_id}",
            "predict_batch": "/predict/batch?package_ids=id1,id2,id3"
        }
    }
    
    if should_return_html(request, format):
        html = format_json_as_html(data, "Event Time Prediction API", "info")
        return HTMLResponse(content=html)
    
    return JSONResponse(content=data)


@app.get("/health", tags=["General"])
async def health_check(request: Request, format: Optional[str] = Query(None)):
    """Health check endpoint"""
    model_loaded = inference_engine is not None and inference_engine.model is not None
    neptune_connected = inference_engine is not None and inference_engine.neptune_extractor is not None
    
    data = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "neptune_connected": neptune_connected,
        "timestamp": get_current_timestamp()
    }
    
    if should_return_html(request, format):
        status = "success" if model_loaded else "error"
        html = format_json_as_html(data, "Health Check", status)
        return HTMLResponse(content=html)
    
    return JSONResponse(content=data)


@app.get("/predict/{package_id}", tags=["Prediction"])
async def predict_single_package(
    package_id: str,
    request: Request,
    format: Optional[str] = Query(None, description="Response format: json or html")
):
    """Predict event times for a single package"""
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail=create_error_response(
                "Inference engine not initialized",
                "Service is starting up or failed to initialize"
            )
        )
    
    try:
        logger.info(f"Predicting for package: {package_id}")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            inference_engine.predict_from_package_id,
            package_id
        )
        
        if 'error' in result:
            logger.warning(f"Prediction failed for {package_id}: {result['error']}")
            
            if should_return_html(request, format):
                html = format_json_as_html(result, f"Prediction Failed: {package_id}", "error")
                return HTMLResponse(content=html, status_code=404 if "not found" in result['error'].lower() else 400)
            
            raise HTTPException(
                status_code=404 if "not found" in result['error'].lower() else 400,
                detail=create_error_response(result['error'])
            )
        
        logger.info(f"✓ Successfully predicted for {package_id}")
        
        if should_return_html(request, format):
            html = format_json_as_html(result, f"Prediction: {package_id}", "success")
            return HTMLResponse(content=html)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting for {package_id}: {e}")
        logger.error(traceback.format_exc())
        
        error_data = create_error_response("Internal server error during prediction", str(e))
        
        if should_return_html(request, format):
            html = format_json_as_html(error_data, f"Error: {package_id}", "error")
            return HTMLResponse(content=html, status_code=500)
        
        raise HTTPException(status_code=500, detail=error_data)


@app.get("/predict/batch", tags=["Prediction"])
async def predict_batch_packages(
    request: Request,
    package_ids: List[str] = Query(..., min_length=1, max_length=100),
    format: Optional[str] = Query(None, description="Response format: json or html")
):
    """Predict event times for multiple packages"""
    if not inference_engine:
        raise HTTPException(
            status_code=503,
            detail=create_error_response(
                "Inference engine not initialized",
                "Service is starting up or failed to initialize"
            )
        )
    
    # Handle comma-separated values
    expanded_ids = []
    for pid in package_ids:
        if ',' in pid:
            expanded_ids.extend([p.strip() for p in pid.split(',') if p.strip()])
        else:
            expanded_ids.append(pid.strip())
    
    # Remove duplicates
    unique_ids = []
    seen = set()
    for pid in expanded_ids:
        if pid not in seen:
            unique_ids.append(pid)
            seen.add(pid)
    
    if not unique_ids:
        raise HTTPException(
            status_code=400,
            detail=create_error_response("No valid package IDs provided")
        )
    
    if len(unique_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                f"Too many package IDs. Maximum 100 allowed, got {len(unique_ids)}"
            )
        )
    
    try:
        logger.info(f"Batch predicting for {len(unique_ids)} packages")
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            inference_engine.batch_predict_from_package_ids,
            unique_ids
        )
        
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        logger.info(f"✓ Batch prediction complete: {successful} successful, {failed} failed")
        
        response_data = {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
            "timestamp": get_current_timestamp()
        }
        
        if should_return_html(request, format):
            html = format_json_as_html(response_data, f"Batch Prediction ({len(unique_ids)} packages)", "success")
            return HTMLResponse(content=html)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        logger.error(traceback.format_exc())
        
        error_data = create_error_response("Internal server error during batch prediction", str(e))
        
        if should_return_html(request, format):
            html = format_json_as_html(error_data, "Batch Prediction Error", "error")
            return HTMLResponse(content=html, status_code=500)
        
        raise HTTPException(status_code=500, detail=error_data)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {
            "error": str(exc.detail),
            "timestamp": get_current_timestamp()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=create_error_response("Internal server error", str(exc))
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("EVENT TIME PREDICTION API")
    logger.info("="*80)
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"Neptune: {NEPTUNE_ENDPOINT}")
    logger.info(f"Thread Pool Workers: {MAX_WORKERS}")
    logger.info("="*80)
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()