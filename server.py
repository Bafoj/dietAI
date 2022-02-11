from random import sample
from typing import Optional
from fastapi import FastAPI, Query, HTTPException, UploadFile
from prisma import Client
from prisma.models import Recetas, Usuario
from prisma.types import RecetasWhereInput
from surprise import SVD
from surprise.dump import load
from itertools import islice
import io
import os
from google.oauth2 import service_account
from google.cloud import vision
import numpy as np
from numpy.linalg import norm

app = FastAPI(title="API NutridAIet")
client = Client(auto_register=True)


cred = os.path.abspath("igneous-bond-254520-d57c8039f85d.json")
credentials = service_account.Credentials.from_service_account_file(cred)
google = vision.ImageAnnotatorClient(credentials=credentials)


recommender: Optional[SVD] = load("recomida.model")[1]


@app.on_event("startup")
async def startup() -> None:
    await client.connect()
    # global recommender
    # _, recommender = load("recomida.model")


@app.on_event("shutdown")
async def shutdown() -> None:
    if client.is_connected():
        await client.disconnect()


@app.get("/recetas")
async def getdbase(cantidad: int = 10) -> list[Recetas]:

    # recetas = await client.recetas.find_many(where={"id": {"in":}})
    return await client.query_raw(
        """
        WITH users_receta as (SELECT "recetaId" as id, COUNT(*) as total
                      from "Interaccion" 
                      GROUP BY "recetaId" 
                      ORDER BY total desc
                      LIMIT $1
                     )
        select r.*
        from "Recetas" r join users_receta i Using(id)
        WHERE r.description IS NOT NULL;
        """,
        cantidad,
        model=Recetas,
    )


@app.post("/new_interaction")
async def new_interaction(*, rate: int = 0, recipeId: int, username: str):
    user = await client.usuario.find_first(where={"nombre": username})
    recipe = await client.recetas.find_first(where={"id": recipeId})
    if not user:
        await client.usuario.create(
            {
                "nombre": username,
            }
        )

    if not recipe:
        return HTTPException(404, "La receta no existe")

    return await client.interaccion.create(
        {
            "puntuacion": rate,
            "receta": {
                "connect": {"id": recipeId},
            },
            "usuario": {
                "connect": {"nombre": username},
            },
        },
        include={"usuario": True, "receta": True},
    )


@app.get("/recomendations")
async def get_recomendation(
    *,
    limitCaloriesMax: int = Query(10_000, gt=0),
    limitCaloriesMin: int = Query(0, gt=0),
    username: str,
    recetasVistas: bool = False,
    solo_despensa: bool = False,
):
    user = await client.usuario.find_first(
        where={"nombre": username}, include={"inter": True}
    )

    if not user:
        return HTTPException(404, f"Usuario con nombre {username} no encontrado")

    recommendation_user_id = user.id

    if not user.trained:
        aux_user = await find_similar_user(user, client)
        if aux_user == None:
            return HTTPException(400, "No se han podido generar recomendaciones")
        recommendation_user_id = aux_user.id

    params: RecetasWhereInput = {
        "calorias": {"gte": limitCaloriesMin, "lt": limitCaloriesMax},
    }
    if not recetasVistas:
        params["inter"] = {"none": {"usuarioId": user.id}}

    if solo_despensa:
        params["ingredientes"] = {
            "some": {"IngredientesDespensa": {"some": {"usuarioId": user.id}}}
        }

    recetas = await client.recetas.find_many(
        where=params, include={"ingredientes": True}, take=5_000
    )
    recetas = sample(recetas, min(500, len(recetas)))

    dict_recetas = {r.id: r for r in recetas}

    if recommender:
        info = [(recommendation_user_id, r.id, -1) for r in recetas]
        predictions = recommender.test(info)
        sorted_preds = islice(sorted(predictions, key=lambda x: -x.est), 10)
        # print("\n".join([str((p.iid, p.est)) for p in predictions]))
        return [dict_recetas[p.iid] for p in sorted_preds]


async def find_similar_user(user: Usuario, client: Client):
    if user == None or user.inter == None:
        return None

    user_vals = {r.recetaId: r.puntuacion for r in user.inter}

    val_ids = list(user_vals.keys())
    candidates = await client.usuario.find_many(
        where={"inter": {"some": {"recetaId": {"in": val_ids}}}, "trained": True},
        include={"inter": {"where": {"recetaId": {"in": val_ids}}}},
    )

    if len(candidates) == 0:
        return None

    similarities = []
    user_vec = np.array([user_vals[i] for i in val_ids]) 

    for candidate in candidates:
        if candidate.inter != None:
            candidate_vals = {r.recetaId: r.puntuacion for r in candidate.inter}
            candidate_vec = np.array([candidate_vals.get(i, 0) for i in val_ids])
            # Cosine similarity
            sim = (candidate_vec @ user_vec.T) / (norm(candidate_vec) * norm(user_vec))
            similarities.append(sim)

    return candidates[np.argmax(similarities)]


@app.get("/pantry")
async def get_pantry(username: str):
    user = await client.usuario.find_first(
        where={"nombre": username},
        include={"IngredientesDespensa": {"include": {"Ingrediente": True}}},
    )
    if not user:
        return HTTPException(404, f"Usuario con nombre {username} no encontrado")
    return user.IngredientesDespensa


@app.post("/pantry")
async def post_pantry(username: str, ingredientId: int, cantidad: int):
    user = await client.usuario.find_first(where={"nombre": username})
    if not user:
        return HTTPException(404, f"Usuario con nombre {username} no encontrado")

    if cantidad <= 0:
        return await client.ingredientesdespensa.delete(
            where={
                "usuarioId_ingredientesId": {
                    "usuarioId": user.id,
                    "ingredientesId": ingredientId,
                }
            }
        )

    return await client.ingredientesdespensa.upsert(
        where={
            "usuarioId_ingredientesId": {
                "usuarioId": user.id,
                "ingredientesId": ingredientId,
            }
        },
        data={
            "create": {
                "cantidad": cantidad,
                "usuario": {"connect": {"id": user.id}},
                "Ingrediente": {"connect": {"id": ingredientId}},
            },
            "update": {"cantidad": cantidad},
        },
    )


@app.post("/pantry/ticket")
async def post_ticket(username: str, file: UploadFile):
    content = await file.read()
    image = vision.Image(content=content)
    response = google.text_detection(image=image) #type: ignore
    texts = response.text_annotations
    return texts[0].description
