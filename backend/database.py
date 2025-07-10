import psycopg2
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv

# Carga las variables de entorno del archivo .env al entorno del sistema.
load_dotenv()

def get_db_connection():
    """
    Establece y retorna una conexion a la base de datos PostgreSQL.
    Las credenciales se obtienen de las variables de entorno.
    """
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    return conn

def create_table():
    """
    Crea la tabla 'books' en la base de datos si no existe.
    Tambien habilita la extension 'vector' de pgvector.
    La tabla esta disenada para almacenar informacion de libros, incluyendo un embedding vectorial.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Habilita la extension pgvector para el manejo de datos vectoriales.
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Define la estructura de la tabla 'books'.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id SERIAL PRIMARY KEY,
            title TEXT,
            author TEXT,
            description TEXT,
            embedding VECTOR(384) -- Almacena el embedding del libro (dimension 384).
        );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Si el script se ejecuta directamente, se llama a la funcion para crear la tabla.
    # Esto es util para la configuracion inicial de la base de datos.
    create_table()
