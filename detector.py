import os
import shutil

from fastapi import UploadFile
from ultralytics import YOLO


UPLOAD_FOLDER = "uploads"


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


model = YOLO("best.pt")



async def detectar(file: UploadFile):


    caminho = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )


    with open(caminho,"wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    resultados = model(
        caminho,
        conf=0.70
    )


    detections = []


    for resultado in resultados:

        for caixa in resultado.boxes:


            confianca = float(
                caixa.conf[0]
            )


            if confianca < 0.70:
                continue


            classe_id = int(
                caixa.cls[0]
            )


            nome = model.names[classe_id]


            x1,y1,x2,y2 = (
                caixa.xyxy[0].tolist()
            )


            detections.append({

                "classe": nome,

                "confianca": confianca,

                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)

            })


    if len(detections)==0:

        return {

            "status":"sucesso",

            "mensagem":
            "Nenhuma uva encontrada",

            "detections":[]

        }



    return {

        "status":"sucesso",

        "mensagem":
        "Uva encontrada",

        "detections":detections

    }