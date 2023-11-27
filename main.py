import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI,HTTPException
import logging
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importamos los datos que se encuentran en formato parquet para dataframes
df_PlayTimeGenre = pd.read_parquet("Datasets/df_PlayTimeGenre_hour_final1.parquet")
df_UserForGenre = pd.read_parquet("Datasets/df_UsersForGenre2_final.parquet")
df_UsersRecommend = pd.read_parquet("Datasets/df_UsersRecommend2_final.parquet")
df_UsersWorstDeveloper = pd.read_parquet("Datasets/df_UserWorstDeveloper_final1.parquet")
df_Sentiment_Analysis = pd.read_parquet("Datasets/df_Sentiment_analysis_final.parquet")


# Primera funcion: PlaytimeGenre

@app.get("/PlayTimeGenre")
def PlayTimeGenre(genero:str):
    """
    La funcion devuelve el año con mas horas jugadas para dicho género.
    """
    generos = df_PlayTimeGenre[df_PlayTimeGenre["main_genre"]== genero] #Filtramos en el dataframe el genero que fue solicitado
    if generos.empty:  #Con esta linea nos aseguramos que si para ese genero no hay resultado se notifique
        return f"No se encontraron datos para el género {genero}"
    año_max = generos.loc[generos["playtime_hour"].idxmax()] #Primero identificamos la fila (indice) que tiene la máxima cantidad de horas jugadas para el género dado y posteriormente se selecciona esa fila a partir del indice
    result = {
        'Genero': genero,
        'Año con Más Horas Jugadas': int(año_max["release_year"]),
        'Total de Horas Jugadas': año_max["playtime_hour"]
    }

    return result

# Segunda funcion: UserForGenre

@app.get("/UserForGenre")
def UserForGenre(genero: str):
    """
    La funcion devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
    """
    generos2 = df_UserForGenre[df_UserForGenre["main_genre"]== genero]
    user_max = generos2.loc[generos2["playtime_hour"].idxmax()]["user_id"]
    horas_x_año = generos2.groupby(["release_year"])["playtime_hour"].sum().reset_index()
    horas_lista = horas_x_año.rename(columns={"release_year": "Año", "playtime_hour": "Horas"}).to_dict(orient="records")    
    result2 = {
        "Genero": genero,
        "Usuario con Más Horas Jugadas": user_max,
        "Total de Horas Jugadas Por Año": horas_lista
    }
    return result2

# Tercera funcion: UsersRecommend

@app.get("/UsersRecommend")
def UsersRecommend(anio:int):
    """
    Funcion que devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
    """
    df_año= df_UsersRecommend[df_UsersRecommend["year_posted"]== anio]
    if type(anio) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if anio < df_UsersRecommend["year_posted"].min() or anio > df_UsersRecommend["year_posted"].max():
        return {"Año no encontrado"}
    df_ordenado_recomendacion = df_año.sort_values(by="recommendation_count", ascending=False)
    top_3_juegos = df_ordenado_recomendacion.head(3)[["app_name","recommendation_count"]]
    result3 ={
        "Año": anio,
        "Top 3 Juegos Más Recomendados": top_3_juegos.to_dict(orient="records")
    }
    return result3

# Cuarta funcion: UsersWorstDeveloper

@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper(anio:int):
    """
    Funcion que devuelve el top 3 de desarrolladoras con juegos MENOS 
    recomendados por usuarios para el año dado.
    """
    df_año2 = df_UsersWorstDeveloper[df_UsersWorstDeveloper["year_posted"]== anio]
    if type(anio) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if anio < df_UsersRecommend["year_posted"].min() or anio > df_UsersRecommend["year_posted"].max():
        return {"Año no encontrado "}
    df_ordenado_recomendacion2 = df_año2.sort_values(by="recommendation_count", ascending=False)
    top_3_developers = df_ordenado_recomendacion2.head(3)[["developer","recommendation_count"]]
    result4 = {
        'Año': anio,
        'Top 3 Desarrolladoras Menos Recomendadas': top_3_developers.rename(columns={"developer": "Desarrolladora", "recommendation_count": "Conteo Recomendacion"}).to_dict(orient="records")
    }
    return result4

# Quinta funcion : sentiment_analysis

@app.get("/SentimentAnalysis")
def sentiment_analysis( desarrolladora : str ):
    """
    Funcion que devuelve un diccionario con el nombre de la desarrolladora como llave y una lista 
    con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con 
    un análisis de sentimiento como valor.
    """
    if type(desarrolladora) != str:
        return "Debes colocar un developer de tipo str, EJ:'07th Expansion'"
    if len(desarrolladora) == 0:
        return "Debes colocar un developer en tipo String"
    df_developer = df_Sentiment_Analysis[df_Sentiment_Analysis["developer"]== desarrolladora]
    sentiment_counts = df_developer.groupby("sentiment_analysis")["sentiment_analysis_count"].sum().to_dict()
    sentiment_dicc = {0: "Negativo", 1: "Neutral", 2: "Positivo"}
    sentiment_counts = {sentiment_dicc[key]: value for key, value in sentiment_counts.items()}
    result50 = {desarrolladora: sentiment_counts}
    return result50

# Sexta funcion: Sistema de recomendacion de juegos

item_similarity_df = pd.read_csv("Datasets/item_similarity_df_final01.csv")
unique_games = pd.read_csv("Datasets/unique_games_final01.csv")

@app.get("/sistema_recomendacion_juego")
def recomendacion_juego(id_juego: int,item_similarity_df,unique_games):
    # Funcion que devuelve una lista con 5 juegos recomendados similares al ingresado.
    if id_juego not in item_similarity_df.index:
        raise ValueError(f"Juego con ID {id_juego} no encontrado en la matriz de similitud.")
    similitudes = item_similarity_df.loc[id_juego].sort_values(ascending=False)[1:6]
    juegos_similares = unique_games.loc[unique_games["item_id"].isin(similitudes.index), "item_name"].tolist()
    resultado = [f"Juegos similares al juego con ID {id_juego} incluyen:"]
    for i, juego in enumerate(juegos_similares, start=1):
        resultado.append(f"{i}. {juego}")
    return resultado
