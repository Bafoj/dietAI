import asyncio
from cmath import nan
from prisma import Client
import pandas as pd
import prisma
from ast import literal_eval
from tqdm import tqdm


async def main() -> None:
    client = Client()
    await client.connect()

    print("Cargando ingredientes:")
    try:
        ingr = pd.read_pickle("datasets/ingr_map.pkl")
        ingr["replaced"] = ingr[
            "replaced"
        ]  # .apply(lambda x: x.replace("'","").replace("\"",""))
        ingr = ingr[["id", "replaced"]].drop_duplicates()
        await client.ingredientes.create_many(
            [{"name": o["replaced"]} for _, o in ingr.iterrows()]
        )
    except prisma.errors.UniqueViolationError as e:
        print(e)

    print("Cargando usuarios:")

    try:
        users = pd.read_csv("datasets/PP_users.csv")
        await client.usuario.create_many(
            [
                {
                    "id":int(o["u"]),
                    "limiteCalorias": -1,
                    "nombre": str(o["u"]),
                    "trained": True,
                }
                for _, o in users.iterrows()
            ]
        )
    except prisma.errors.UniqueViolationError as e:
        print(e)


    print("Cargando recetas:")

    try:
        recipes = pd.read_csv("datasets/FormatedRecipes.csv")
        # recipes['calories'] = recipes['calories'].apply(lambda x: x[1:-1].split(','))
        recipes["ingredients"] = recipes["ingredients"].apply(literal_eval)
        recipes["steps"] = recipes["steps"].apply(literal_eval)
        for _, o in tqdm(recipes.iterrows(),total=recipes.shape[0]):  # type: ignore
            #
            # lista = [{"name":r.replace("'","").replace("\"","")} for r in o["ingredients"]]
            # print(o['ingredients'])
            await client.recetas.create(
                {
                    "id": int(o["i"]),
                    "nombre": str(o["name"]),
                    "calorias": float(o["calories"]),  # type: ignore
                    "ingredientes": {
                        "connect": [{"name": str(n)} for n in o["ingredients"]]
                    },
                    "description": o["description"] if str(o["description"]) != 'nan' else None,
                    "minutes": int(o["minutes"]),
                    "steps": o["steps"]
                }
            )
    except prisma.errors.UniqueViolationError as e:
        print(e)

    print("Cargando interacciones:")
    try:
        interactions = pd.read_csv("datasets/interactions.csv")
        for _, o in tqdm(interactions.iterrows(),total= interactions.shape[0]):  # type: ignore
            await client.interaccion.create(
                {
                    "puntuacion": o["rating"],
                    "receta": {"connect": {"id": o["i"]}},
                    "usuario": {"connect": {"id": o["u"]}},
                }
            )
    except prisma.errors.UniqueViolationError as e:
        print(e)


    await client.disconnect()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
