import asyncio
import concurrent.futures
import logging

import service
import uvicorn
from config import APP_NAME, ERROR_RESPONSE, STREAM_DEFAULT, init_logger
from fastapi import FastAPI, status
from fastapi_utils.tasks import repeat_every

init_logger()
logger = logging.getLogger(APP_NAME)
executor = concurrent.futures.ThreadPoolExecutor()

# FastAPI
app = FastAPI()


@app.on_event("startup")
@repeat_every(seconds=3600, logger=logger)
async def startup_event():
    # Start the background task to check for new model
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, service.check_for_new_model)


@app.get(
    "/ready",
    description="Readiness probe - to check if API can serve requests",
    status_code=status.HTTP_200_OK,
    tags=["Maintenance"],
)
async def ready():
    return {"status": "ready"}


@app.get(
    "/alive",
    description="Liveness probe for - to check if the API is alive",
    status_code=status.HTTP_200_OK,
    tags=["Maintenance"],
)
async def alive():
    return {"status": "alive"}


@app.post("/v1/completions")
async def completions(content: dict):
    try:
        response = service.completions_response(content)
        return response
    except Exception as ex:
        logger.exception(f"An error has occurred: {ex}")
        return service.get_generic_message(ERROR_RESPONSE, stream=content.get("stream", STREAM_DEFAULT))


@app.post("/v1/chat/completions")
async def chat_completions(content: dict):
    try:
        response = service.chat_completions_response(content)
        return response
    except Exception as ex:
        logger.exception(f"An error has occurred: {ex}")
        return service.get_generic_message(ERROR_RESPONSE, stream=content.get("stream", STREAM_DEFAULT))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
