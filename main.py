from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from detector import detectar


app = FastAPI(
    title="MICROS API"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def inicio():
    return {
        "status": "MICROS API online"
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    resultado = await detectar(file)

    return resultado