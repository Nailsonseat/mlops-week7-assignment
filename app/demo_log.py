import os
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
import logging
import time
import json
import sys

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace.sampling import AlwaysOnSampler


# --- Observability Setup ---

# ✅ UPDATED: Use an AlwaysOnSampler to ensure every trace is captured for debugging
sampler = AlwaysOnSampler()
trace.set_tracer_provider(TracerProvider(sampler=sampler))
tracer = trace.get_tracer(_name_)

# 1. Setup Google Cloud Trace Exporter (for production)
cloud_trace_exporter = CloudTraceSpanExporter()
span_processor_gcp = BatchSpanProcessor(cloud_trace_exporter)
trace.get_tracer_provider().add_span_processor(span_processor_gcp)

# 2. ✅ ADDED: Setup Console Exporter (for debugging)
# This will print traces directly to your container logs.
console_exporter = ConsoleSpanExporter()
span_processor_console = SimpleSpanProcessor(console_exporter)
trace.get_tracer_provider().add_span_processor(span_processor_console)


# 3. Setup structured JSON logging for Google Cloud Logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, self.datefmt),
        }
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        return json.dumps(log_record)

logger = logging.getLogger("iris-classifier")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)


# --- FastAPI Application ---

app = FastAPI(title="Iris Classifier API")

# Instrument the FastAPI app to automatically create traces
FastAPIInstrumentor.instrument_app(app)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

model_path = "model.joblib"
model = None
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    """Load the model and set app state at startup."""
    global model
    logger.info({"event": "startup_begin", "message": "Starting model loading process..."})
    time.sleep(1)
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            app_state["is_ready"] = True
            logger.info({"event": "startup_success", "message": "Model loaded successfully."})
        except Exception as e:
            app_state["is_ready"] = False
            logger.error({"event": "model_load_error", "message": f"Failed to load model: {str(e)}"})
    else:
        app_state["is_ready"] = False
        logger.error({"event": "startup_failure", "message": f"Model file not found at {model_path}"})
        
    logger.info({"event": "startup_complete", "message": f"Startup completed. Ready state: {app_state['is_ready']}"})

# --- Health Probes ---

@app.get("/live_check", tags=["Probes"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probes"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# --- API Endpoints ---

@app.post("/predict/", tags=["Prediction"])
def predict_species(data: IrisInput):
    with tracer.start_as_current_span("model_inference") as span:
        if not app_state["is_ready"] or model is None:
            raise HTTPException(status_code=503, detail="Model is not ready.")
            
        try:
            input_df = pd.DataFrame([data.dict()])
            prediction = model.predict(input_df)[0]
            
            logger.info({
                "event": "prediction_success",
                "input": data.dict(),
                "result": prediction
            })
            return {"predicted_class": prediction}
        except Exception as e:
            logger.error({
                "event": "prediction_error",
                "error": str(e)
            })
            raise HTTPException(status_code=500, detail="Prediction failed.")
