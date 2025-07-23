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
from datetime import datetime
from typing import List, Dict, Any, TypedDict

# Importaciones para RAG con LangGraph y LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# --- Inicializacion de la App ---
app = FastAPI(
    title="Book Search & RAG API",
    description="Una API para buscar libros usando embeddings, pgvector y un chatbot RAG.",
    version="2.0.0"
)

# --- Middleware de CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En produccion, restringir a dominios especificos.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuración de modelos ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuración de Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    llm = None
else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )


class RAGState(TypedDict):
    """
    Estado centralizado para el sistema RAG con LangGraph.
    """
    # Entrada del usuario
    question: str
    
    # Parámetros de recuperación
    k: int
    similarity_threshold: float
    
    # Datos de documentos recuperados
    retrieved_docs: List[Document]
    context_text: str
    
    # Análisis de la consulta
    search_intent: str  # "exact_match", "recommendation", "general_query"
    found_exact_match: bool
    
    # Respuesta generada
    answer: str
    sources: List[str]
    recommended_books: List[Dict[str, Any]]
    
    # Metadatos adicionales
    processing_metadata: Dict[str, Any]

# --- Prompt Template para RAG ---
book_recommendation_prompt = ChatPromptTemplate.from_messages([
    ("system", """Eres un bibliotecario experto y amigable especializado en recomendaciones de libros. 
    Tu trabajo es ayudar a los usuarios a encontrar libros perfectos basándote en la información de nuestra biblioteca.

    CONTEXTO DE LA BIBLIOTECA:
    {context}

    INSTRUCCIONES:
    1. Si el usuario pregunta por un libro específico que existe en nuestra biblioteca, proporciona información detallada sobre ese libro.
    
    2. Si el libro no existe exactamente pero hay libros similares, recomienda los más relevantes explicando por qué son buenas alternativas.
    
    3. Para consultas generales (géneros, temas, autores), recomienda los mejores libros de nuestra colección que coincidan.
    
    4. Siempre incluye:
       - Título y autor de cada libro recomendado
       - Breve descripción de por qué es relevante
       - El género o tema principal
    
    5. Mantén un tono conversacional, entusiasta y personalizado.
    
    6. Si no hay libros relevantes en nuestra biblioteca, sugiere ampliar la búsqueda o buscar en otras fuentes.

    FORMATO DE RESPUESTA:
    - Comienza con una respuesta directa a la pregunta
    - Lista los libros recomendados de forma clara
    - Termina con una invitación a seguir preguntando sobre otros libros.
    """),

    ("human", "Consulta del usuario: {question}")
])

# --- Modelos Pydantic existentes ---
class SearchQuery(BaseModel):
    """Define la estructura para una consulta de busqueda."""
    query: str
    threshold: float = 0.5

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



# --- Nuevos modelos para RAG ---
class ChatQuery(BaseModel):
    """Modelo para consultas del chatbot RAG."""
    question: str
    k: int = 5  # Número de libros a recuperar
    similarity_threshold: float = 0.3  # Umbral de similitud

class ChatResponse(BaseModel):
    """Respuesta del chatbot RAG."""
    answer: str
    sources: List[str]
    recommended_books: List[Dict[str, Any]]
    search_intent: str
    found_exact_match: bool
    processing_metadata: Dict[str, Any]

# --- Funciones auxiliares para RAG ---
def get_books_from_db(query_embedding: np.ndarray, k: int = 5, similarity_threshold: float = 0.3) -> List[Document]:
    """
    Recupera documentos de libros similares desde la base de datos.
    """
    conn = get_db_connection()
    register_vector(conn)
    cursor = conn.cursor()
    
    # Convertir umbral de similitud a distancia
    distance_threshold = float(np.sqrt(2 * (1 - similarity_threshold)))
    
    cursor.execute(
        """
        SELECT title, author, description, embedding, embedding <-> %s AS distance 
        FROM books 
        WHERE (embedding <-> %s) < %s 
        ORDER BY distance 
        LIMIT %s
        """,
        (query_embedding, query_embedding, distance_threshold, k)
    )
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    documents = []
    for row in results:
        similarity_score = 1 - (row[4] ** 2) / 2
        
        # Crear documento LangChain
        doc = Document(
            page_content=f"Título: {row[0]}\nAutor: {row[1]}\nDescripción: {row[2]}",
            metadata={
                "title": row[0],
                "author": row[1], 
                "similarity_score": similarity_score,
                "source_type": "book_database"
            }
        )
        documents.append(doc)
    
    return documents

def analyze_search_intent(question: str, retrieved_docs: List[Document]) -> tuple[str, bool]:
    """
    Analiza la intención de búsqueda del usuario.
    """
    question_lower = question.lower()
    
    # Buscar coincidencia exacta por título
    exact_match = False
    for doc in retrieved_docs:
        title_lower = doc.metadata["title"].lower()
        if title_lower in question_lower or any(word in title_lower for word in question_lower.split() if len(word) > 3):
            if doc.metadata["similarity_score"] > 0.8:
                exact_match = True
                break
    
    # Determinar intención
    if exact_match:
        intent = "exact_match"
    elif any(word in question_lower for word in ["recomienda", "similar", "parecido", "como", "tipo"]):
        intent = "recommendation"
    else:
        intent = "general_query"
    
    return intent, exact_match

# --- Nodos del grafo RAG ---
def retrieve_documents(state: RAGState) -> RAGState:
    """
    Nodo de recuperación de documentos.
    """
    question = state["question"]
    k = state.get("k", 5)
    similarity_threshold = state.get("similarity_threshold", 0.3)
    
    # Generar embedding para la consulta
    query_embedding = model.encode([question])[0]
    
    # Recuperar documentos relevantes
    retrieved_docs = get_books_from_db(query_embedding, k, similarity_threshold)
    
    # Registrar metadatos del proceso
    processing_metadata = {
        "documents_retrieved": len(retrieved_docs),
        "retrieval_timestamp": datetime.now().isoformat(),
        "query_embedding_generated": True
    }
    
    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "processing_metadata": processing_metadata
    }

def analyze_intent(state: RAGState) -> RAGState:
    """
    Nodo de análisis de intención de búsqueda.
    """
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]
    
    # Analizar intención de búsqueda
    search_intent, found_exact_match = analyze_search_intent(question, retrieved_docs)
    
    # Actualizar metadatos
    processing_metadata = state.get("processing_metadata", {})
    processing_metadata.update({
        "intent_analysis_timestamp": datetime.now().isoformat(),
        "search_intent_detected": search_intent
    })
    
    return {
        **state,
        "search_intent": search_intent,
        "found_exact_match": found_exact_match,
        "processing_metadata": processing_metadata
    }

def prepare_context(state: RAGState) -> RAGState:
    """
    Nodo de preparación de contexto.
    """
    retrieved_docs = state["retrieved_docs"]
    
    # Preparar el contexto concatenando contenido
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Extraer fuentes y libros recomendados
    sources = []
    recommended_books = []
    
    for doc in retrieved_docs:
        source_info = f"{doc.metadata['title']} por {doc.metadata['author']}"
        sources.append(source_info)
        
        recommended_books.append({
            "title": doc.metadata["title"],
            "author": doc.metadata["author"],
            "similarity_score": doc.metadata["similarity_score"]
        })
    
    # Actualizar metadatos
    processing_metadata = state.get("processing_metadata", {})
    processing_metadata.update({
        "context_length": len(context_text),
        "sources_count": len(set(sources)),
        "context_preparation_timestamp": datetime.now().isoformat()
    })
    
    return {
        **state,
        "context_text": context_text,
        "sources": sources,
        "recommended_books": recommended_books,
        "processing_metadata": processing_metadata
    }

def generate_response(state: RAGState) -> RAGState:
    """
    Nodo de generación de respuesta.
    """
    
    if not llm:
        return {
            **state,
            "answer": "Lo siento, el servicio de chatbot no está disponible. GOOGLE_API_KEY no configurada.",
            "processing_metadata": {**state.get("processing_metadata", {}), "error": "GOOGLE_API_KEY not configured"}
        }
    
    question = state["question"]
    context_text = state["context_text"]
    
    # Construir mensaje usando el prompt
    messages = book_recommendation_prompt.invoke({
        "question": question,
        "context": context_text
    })
    
    # Generar respuesta con el LLM
    response = llm.invoke(messages)
    
    # Actualizar metadatos finales
    processing_metadata = state.get("processing_metadata", {})
    processing_metadata.update({
        "response_length": len(response.content),
        "generation_timestamp": datetime.now().isoformat(),
        "model_used": "gemini-2.0-flash-exp"
    })
    
    return {
        **state,
        "answer": response.content,
        "processing_metadata": processing_metadata
    }

def create_rag_graph():
    """
    Crea el grafo RAG usando LangGraph StateGraph.
    """
    workflow = StateGraph(RAGState)
    
    # Agregar nodos de procesamiento
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("prepare_context", prepare_context) 
    workflow.add_node("generate", generate_response)
    
    # Definir el flujo
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "analyze_intent")
    workflow.add_edge("analyze_intent", "prepare_context")
    workflow.add_edge("prepare_context", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# Inicializar el grafo RAG
rag_graph = create_rag_graph()
@app.get("/", summary="Mensaje de bienvenida")
def read_root():
    """
    Endpoint raiz que muestra un mensaje de bienvenida y enlaces utiles.
    """
    return {
        "message": "Bienvenido a la API de Busqueda de Libros con RAG",
        "docs_url": "http://localhost:8000/docs",
        "front_url": "http://localhost:8080",
        "pgadmin_url": "http://localhost:5050",
        "chatbot_url": "/api/chat",
        "chatbot_stream_url": "/api/chat/stream"
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



# --- Nuevos endpoints para RAG ---
@app.post("/api/chat", response_model=ChatResponse, summary="Chatbot RAG para recomendaciones de libros")
def chat_with_rag(chat_query: ChatQuery):
    """
    Endpoint principal del chatbot RAG.
    Procesa consultas sobre libros y genera recomendaciones inteligentes.
    """
    if not chat_query.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    if not llm:
        raise HTTPException(
            status_code=503, 
            detail="Servicio RAG no disponible. GOOGLE_API_KEY no configurada."
        )
    
    try:
        # Estado inicial para el grafo RAG
        initial_state: RAGState = {
            "question": chat_query.question,
            "k": chat_query.k,
            "similarity_threshold": chat_query.similarity_threshold,
            "retrieved_docs": [],
            "context_text": "",
            "search_intent": "",
            "found_exact_match": False,
            "answer": "",
            "sources": [],
            "recommended_books": [],
            "processing_metadata": {}
        }
        
        # Ejecutar el grafo RAG
        final_state = rag_graph.invoke(initial_state)
        
        return ChatResponse(
            answer=final_state["answer"],
            sources=final_state["sources"],
            recommended_books=final_state["recommended_books"],
            search_intent=final_state["search_intent"],
            found_exact_match=final_state["found_exact_match"],
            processing_metadata=final_state["processing_metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento RAG: {str(e)}")

async def chat_stream_generator(chat_query: ChatQuery):
    """
    Generador para streaming de respuestas del chatbot.
    """
    yield "🤖 Iniciando consulta al chatbot RAG...\n\n"
    await asyncio.sleep(0.1)
    
    try:
        if not llm:
            yield "❌ Error: Servicio RAG no disponible. GOOGLE_API_KEY no configurada.\n"
            return
            
        yield "🔍 Buscando libros relevantes en la biblioteca...\n"
        await asyncio.sleep(0.1)
        
        # Ejecutar RAG paso a paso con logging
        initial_state: RAGState = {
            "question": chat_query.question,
            "k": chat_query.k,
            "similarity_threshold": chat_query.similarity_threshold,
            "retrieved_docs": [],
            "context_text": "",
            "search_intent": "",
            "found_exact_match": False,
            "answer": "",
            "sources": [],
            "recommended_books": [],
            "processing_metadata": {}
        }
        
        yield f"📚 Encontrados libros relevantes, analizando intención...\n"
        await asyncio.sleep(0.1)
        
        # Ejecutar el grafo
        final_state = await asyncio.to_thread(rag_graph.invoke, initial_state)
        
        yield f"✨ Generando respuesta personalizada...\n\n"
        await asyncio.sleep(0.1)
        
        yield "📖 **RESPUESTA:**\n"
        yield f"{final_state['answer']}\n\n"
        
        if final_state["recommended_books"]:
            yield "📋 **LIBROS RECOMENDADOS:**\n"
            for book in final_state["recommended_books"]:
                yield f"• {book['title']} por {book['author']} (Similitud: {book['similarity_score']:.1%})\n"
        
        yield f"\n🎯 **Tipo de búsqueda:** {final_state['search_intent']}\n"
        yield f"✅ **Coincidencia exacta:** {'Sí' if final_state['found_exact_match'] else 'No'}\n"
        
    except Exception as e:
        yield f"\n❌ Error: {str(e)}\n"

@app.post("/api/chat/stream", summary="Chatbot RAG con respuesta en streaming")
def chat_stream(chat_query: ChatQuery):
    """
    Versión streaming del chatbot RAG que muestra el progreso en tiempo real.
    """
    if not chat_query.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    return StreamingResponse(
        chat_stream_generator(chat_query), 
        media_type="text/plain"
    )

@app.get("/api/chat/health", summary="Verificar estado del sistema RAG")
def rag_health_check():
    """
    Endpoint para verificar que todos los componentes RAG estén funcionando.
    """
    health_status = {
        "rag_system": "operational",
        "gemini_api": "configured" if llm else "not_configured",
        "sentence_transformer": "loaded",
        "database": "connected",
        "timestamp": datetime.now().isoformat()
    }
    
    # Verificar conexión a base de datos
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM books;")
        book_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        health_status["book_count"] = book_count
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
    
# --- Ejecutor de Uvicorn ---
if __name__ == "__main__":
    # Inicia el servidor de desarrollo Uvicorn si el script se ejecuta directamente.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)