import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from database import get_db_connection, create_table
from pgvector.psycopg2 import register_vector
import json
import os

# --- Inicializacion de la App ---
# Crea una instancia de la aplicacion FastAPI.
app = FastAPI(
    title="Book Search API",
    description="Una API para buscar libros usando embeddings y pgvector.",
    version="1.0.0"
)

# --- Middleware de CORS ---
# Permite que el frontend (que se ejecuta en un origen diferente) se comunique con este backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En produccion, restringir a dominios especificos.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Carga del Modelo ---
# Carga el modelo de Sentence Transformer al iniciar la aplicacion.
# Esto evita tener que cargarlo en cada solicitud, mejorando el rendimiento.
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Modelos Pydantic (Estructuras de Datos) ---
class SearchQuery(BaseModel):
    """Define la estructura para una consulta de busqueda."""
    query: str
    threshold: float = 0.5  # Umbral de similitud minimo, con un valor por defecto.

class BookSummary(BaseModel):
    """Define una estructura resumida de un libro."""
    title: str
    author: str

class Book(BaseModel):
    """Define la estructura de un libro para las respuestas de la API."""
    title: str
    author: str
    description: str
    embedding: list[float] | None = None

class SearchResult(BaseModel):
    """Define la estructura para un resultado de busqueda, incluyendo el libro y la puntuacion de similitud."""
    book: Book
    score: float

class SearchResponse(BaseModel):
    """Define la estructura de la respuesta de busqueda, incluyendo el embedding 2D de la consulta."""
    query_embedding_2d: list[float]
    results: list[SearchResult]

# --- Endpoints de la API ---
@app.get("/", summary="Mensaje de bienvenida")
def read_root():
    """
    Endpoint raiz que muestra un mensaje de bienvenida y enlaces utiles.
    """
    return {
        "message": "Bienvenido a la API de Busqueda de Libros",
        "docs_url": "http://localhost:8000/docs",
        "front_url": "http://localhost:8080",
        "pgadmin_url": "http://localhost:5050"
    }

@app.get("/api/books", response_model=list[BookSummary], summary="Obtener todos los libros")
def get_all_books():
    """
    Retorna una lista completa de todos los libros almacenados en la base de datos.
    No incluye los embeddings para mantener la respuesta ligera.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT title, author FROM books;")
    books = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [{"title": row[0], "author": row[1]} for row in books]

async def book_processing_stream(books_to_upload: list[Book]):
    """
    Un generador asincrono que procesa la carga de libros y emite logs en tiempo real.
    """
    yield "Iniciando proceso de carga...\n"
    await asyncio.sleep(0.1)

    try:
        # 1. Asegura que la tabla exista
        create_table()
        yield "Tabla 'books' asegurada.\n"
        await asyncio.sleep(0.1)

        if not books_to_upload:
            yield "Error: La lista de libros no puede estar vacía.\n"
            return

        conn = get_db_connection()
        register_vector(conn)
        cursor = conn.cursor()

        # 2. Verifica los libros existentes
        yield "Verificando libros existentes en la base de datos...\n"
        await asyncio.sleep(0.1)
        cursor.execute("SELECT title FROM books")
        existing_titles = {row[0] for row in cursor.fetchall()}
        yield f"Se encontraron {len(existing_titles)} libros existentes.\n"
        await asyncio.sleep(0.1)

        new_books_data = []
        processed_count = 0
        
        for book in books_to_upload:
            processed_count += 1
            yield f"\n({processed_count}/{len(books_to_upload)}) Procesando libro: '{book.title}'...\n"
            await asyncio.sleep(0.1)
            if book.title in existing_titles:
                yield f"Resultado: El libro ya existe. Omitiendo.\n"
                await asyncio.sleep(0.1)
            else:
                new_books_data.append(book)
                yield f"Resultado: Libro nuevo. Se añadirá a la base de datos.\n"
                await asyncio.sleep(0.1)

        if not new_books_data:
            yield "\nNo hay libros nuevos para añadir.\n"
            cursor.close()
            conn.close()
            yield "Proceso finalizado.\n"
            return

        # 3. Genera embeddings para los libros nuevos
        yield f"\nGenerando embeddings para {len(new_books_data)} libros nuevos...\n"
        await asyncio.sleep(0.1)
        descriptions = [book.description for book in new_books_data]
        
        # model.encode no es async, así que lo ejecutamos en el pool de hilos por defecto
        embeddings = await asyncio.to_thread(model.encode, descriptions, show_progress_bar=False)
        yield "Embeddings generados exitosamente.\n"
        await asyncio.sleep(0.1)

        # 4. Inserta los libros nuevos en la base de datos
        yield "Insertando libros nuevos en la base de datos...\n"
        await asyncio.sleep(0.1)
        for book, embedding in zip(new_books_data, embeddings):
            embedding_preview = f"[{', '.join(f'{x:.4f}' for x in embedding[:4])}, ...]"
            yield f"   - Embedding para '{book.title}': {embedding_preview}\n"
            await asyncio.sleep(0.05)
            
            cursor.execute(
                "INSERT INTO books (title, author, description, embedding) VALUES (%s, %s, %s, %s)",
                (book.title, book.author, book.description, embedding)
            )
            yield f"   - '{book.title}' insertado en la base de datos.\n"
            await asyncio.sleep(0.05)
        
        conn.commit()
        count = len(new_books_data)
        cursor.close()
        conn.close()

        yield f"\n¡Éxito! Se han añadido {count} libros nuevos a la base de datos.\n"
        yield "Proceso finalizado.\n"

    except Exception as e:
        yield f"\nHa ocurrido un error inesperado: {str(e)}\n"
        yield "Proceso interrumpido.\n"


@app.post("/api/upload_books", summary="Cargar una lista de libros y procesarlos con logs en tiempo real")
def upload_books_stream(books_to_upload: list[Book]):
    """
    Carga una lista de libros y devuelve un stream de logs del proceso.
    """
    return StreamingResponse(book_processing_stream(books_to_upload), media_type="text/plain")

@app.post("/api/search", response_model=SearchResponse, summary="Buscar libros por similitud semantica")
def search_books(search_query: SearchQuery):
    """
    Realiza una busqueda semantica basada en la consulta del usuario.
    Utiliza pgvector para encontrar los libros mas similares en la base de datos.
    Retorna los resultados y el embedding 2D de la consulta.
    """
    if not search_query.query:
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacia.")

    # 1. Genera el embedding para la consulta de busqueda.
    query_embedding = model.encode([search_query.query])[0]

    # 2. Realiza la busqueda por similitud en la base de datos.
    conn = get_db_connection()
    register_vector(conn)
    cursor = conn.cursor()
    
    distance_threshold = float(np.sqrt(2 * (1 - search_query.threshold)))

    cursor.execute(
        """
        SELECT title, author, description, embedding, embedding <-> %s AS distance 
        FROM books 
        WHERE (embedding <-> %s) < %s 
        ORDER BY distance 
        LIMIT 5
        """,
        (query_embedding, query_embedding, distance_threshold)
    )
    
    search_results_db = cursor.fetchall()
    cursor.close()
    conn.close()

    # 3. Formatea los resultados, aplicando PCA al conjunto.
    if not search_results_db:
        return SearchResponse(query_embedding_2d=[], results=[])

    # Combina el embedding de la consulta con los de los resultados para un PCA coherente.
    original_embeddings = np.array([row[3] for row in search_results_db])
    all_embeddings = np.vstack([query_embedding, original_embeddings])

    # Inicializa PCA y reduce la dimensionalidad a 2.
    pca = PCA(n_components=2)
    reduced_embeddings_all = pca.fit_transform(all_embeddings)

    # El primer vector es el de la consulta; el resto son los resultados.
    query_embedding_2d = reduced_embeddings_all[0].tolist()
    reduced_embeddings_results = reduced_embeddings_all[1:]

    results = []
    for i, row in enumerate(search_results_db):
        similarity_score = 1 - (row[4] ** 2) / 2
        embedding_list = reduced_embeddings_results[i].tolist()

        results.append(
            SearchResult(
                book=Book(
                    title=row[0],
                    author=row[1],
                    description=row[2],
                    embedding=embedding_list
                ),
                score=similarity_score
            )
        )
        
    return SearchResponse(query_embedding_2d=query_embedding_2d, results=results)

# --- Ejecutor de Uvicorn ---
if __name__ == "__main__":
    # Inicia el servidor de desarrollo Uvicorn si el script se ejecuta directamente.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
