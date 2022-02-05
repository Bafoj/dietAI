import asyncio
from prisma import Client
import pandas as pd
import prisma
from ast import literal_eval

async def main() -> None:
    client = Client()
    await client.connect()

    try:
        ingr = pd.read_pickle('datasets/ingr_map.pkl')
        ingr["replaced"] = ingr["replaced"]#.apply(lambda x: x.replace("'","").replace("\"",""))
        ingr = ingr[["id","replaced"]].drop_duplicates()
        await client.ingredientes.create_many([{'name':o["replaced"]} for _,o in ingr.iterrows()])
    except prisma.errors.UniqueViolationError as e:
        print(e)

    # try:
    #     users = pd.read_csv('datasets/PP_users.csv')
    #     await client.usuario.create_many([{"limiteCalorias":-1,"nombre":str(o["u"])} for _,o in users.iterrows()])
    # except prisma.errors.UniqueViolationError as e:
    #     print(e)

    # try:
    #     recipes = pd.read_csv("datasets/FormatedRecipes.csv")
    #     # recipes['calories'] = recipes['calories'].apply(lambda x: x[1:-1].split(','))
    #     recipes['ingredients'] = recipes['ingredients'].apply(literal_eval)
    #     for _, o in recipes.iterrows(): # type: ignore
    #     #    
    #         # lista = [{"name":r.replace("'","").replace("\"","")} for r in o["ingredients"]]
    #         # print(o['ingredients'])
    #         await client.recetas.create(
    #             {
    #                 "nombre": str(o["name"]),
    #                 "calorias": o["calories"], # type: ignore
    #                 "ingredientes": {
    #                     "connect":[{"name":n} for n in o['ingredients']]
    #                 },
    #             }
    #         )
    # except prisma.errors.UniqueViolationError as e:
    #     print(e)

    # await client._ingredientesToRecetas.create({})
    # await client.interaccion.create({'puntuacion':4,'usuario':{"connect":{"id":"swad"}}})

    await client.disconnect()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
