import operator
import os
import re
import string
from itertools import islice
from random import sample
from typing import Optional

import gensim
import numpy as np
import spacy
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import LsiModel, TfidfModel
from google.cloud import vision
from google.oauth2 import service_account
from numpy.linalg import norm
from prisma import Client
from prisma.models import Recetas, Usuario
from prisma.types import RecetasWhereInput
from surprise import SVD
from surprise.dump import load

app = FastAPI(title="API NutridAIet")
client = Client(auto_register=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "*"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cred = os.path.abspath("cred.json")
credentials = service_account.Credentials.from_service_account_file(cred)
google = vision.ImageAnnotatorClient(credentials=credentials)
spacy_nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS #type: ignore


recommender: Optional[SVD] = load("recomida.model")[1]

dictionary = gensim.utils.SaveLoad.load('NLP/dictionary')
food_lsi_model = LsiModel.load('NLP/lsi')
##food_lsi_projection = gensim.utils.SaveLoad.load('lsi.projection')
food_tfidf_model = TfidfModel.load('NLP/tfidf')
food_index = gensim.utils.SaveLoad.load('NLP/food_index')


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
async def getdbase(cantidad: int = Query(10,ge=1, le=100))->list:

    # recetas = await client.recetas.find_many(where={"id": {"in":}})
    recetas = await client.query_raw(
        """
        WITH users_receta as (SELECT "recetaId" as id, COUNT(*) as total
                      from "Interaccion" 
                      GROUP BY "recetaId" 
                      ORDER BY total desc
                      LIMIT 500
                     )
        select r.*
        from "Recetas" r join users_receta i Using(id)
        WHERE r.description IS NOT NULL;
        """,
        model=Recetas,
    )

    return sample(recetas, min(cantidad, len(recetas)))


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
async def post_pantry(username: int, ingredientId: int, cantidad: int):
    user = await client.usuario.find_first(where={"id": username})
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
async def post_ticket(user: int, file: UploadFile):
    
    content = await file.read()
    image = vision.Image(content=content)
    response = google.text_detection(image=image) #type: ignore
    texts = response.text_annotations
    words = spacy_tokenizer(texts[0].description) 
    food_list = []
    for word in words:
        similar = search_similar_food(word)
        if len(similar) > 0:
            await client.ingredientesdespensa.upsert(
                where={
                    "usuarioId_ingredientesId": {
                        "usuarioId": user,
                        "ingredientesId": similar[0],
                    }
                },
                data={
                    "create": {
                        "cantidad": 1,
                        "usuario": {"connect": {"id": user}},
                        "Ingrediente": {"connect": {"id": similar[0]}},
                    },
                    "update": {"cantidad": 1},
                },
            )
    usuario = await client.usuario.find_first(where={"id": user}, include={"IngredientesDespensa": {"include": {"Ingrediente": True}}})

    if usuario == None:
        return HTTPException(404, "Usuario no encontrado")

    return usuario.IngredientesDespensa

def spacy_tokenizer(sentence):
    sentence = re.sub('\'','',sentence)
    sentence = re.sub('\w*\d\w*','',sentence)
    sentence = re.sub(' +',' ',sentence)
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    sentence = re.sub(r'\n',' ',sentence)
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    tokens = spacy_nlp(sentence)
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    return tokens

def search_similar_food(search_term):

    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = food_tfidf_model[query_bow]
    query_lsi = food_lsi_model[query_tfidf]

    food_index.num_best = 1

    food_list = food_index[query_lsi]

    food_list.sort(key=operator.itemgetter(1), reverse=True)
    food_names = []

    for j, food in enumerate(food_list):

        food_names.append(int(food[0]))

        if j == (food_index.num_best-1):
            break
    return food_names
    #return pd.DataFrame(food_names, columns=['Relevance','Food ingredient','Food Plot'])
