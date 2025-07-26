# Seneca Book Finder

Este proyecto es una aplicación web que permite realizar búsquedas semánticas de libros. Utiliza embeddings de texto para encontrar libros basados en el significado de la consulta del usuario, en lugar de solo coincidencias de palabras clave.

La aplicación está completamente empaquetada en contenedores docker para un despliegue simple.

## Componentes del Proyecto

El proyecto está dividido en los siguientes servicios, orquestados con Docker Compose:

1.  **Frontend (`./frontend`)**
    *   **Tecnología:** HTML, CSS, JavaScript y D3.js.
    *   **Descripción:** Una interfaz de usuario que permite a los usuarios introducir consultas de búsqueda, ajustar un umbral de similitud y ver los resultados. Incluye una visualización de embeddings 2D interactiva para explorar la relación semántica entre la consulta y los resultados.
    *   **Servidor:** Un contenedor `nginx` ligero.
    *   **Puerto expuesto:** `8080`

2.  **Backend (`./backend`)**
    *   **Tecnología:** Python + FastAPI, SentenceTransformers, pgvector y Psycopg2.
    *   **Descripción:** API REST que recibe las consultas de búsqueda, genera un embedding para la consulta, y utiliza la extensión `pgvector` en PostgreSQL para encontrar los libros más similares en la base de datos. También realiza una reducción de dimensionalidad (PCA) para la visualización en el frontend.
    *   **Puerto expuesto:** `8000`

3.  **Base de Datos (`db`)**
    *   **Tecnología:** PostgreSQL con la extensión `pgvector`.
    *   **Descripción:** Almacena la información de los libros (título, autor, descripción) junto con sus embeddings vectoriales precalculados. `pgvector` permite realizar búsquedas de similitud de vectores de manera eficiente.
    *   **Puerto expuesto:** El definido por `DB_PORT` en el archivo `.env` (por defecto `5432`).

4.  **pgAdmin (`pgadmin`)**
    *   **Tecnología:** dpage/pgadmin4.
    *   **Descripción:** Un administrador web para la base de datos PostgreSQL, útil para explorar los datos y la estructura de las tablas. Viene preconfigurada para conectarse a la base de datos del proyecto (password: password).
    *   **Puerto expuesto:** `5050`

## Requisitos Previos

*   Docker
*   Docker Compose

## Cómo Ejecutar el Proyecto

1.  **Clonar el repositorio:**

2.  **Configurar el archivo de entorno:**
    Ajuste el archivo `.env` en la raíz del proyecto con las credenciales de la base de datos. El archivo `.env.example` presente en la raíz del proyecto ejemplifica el contenido que debe contener el archivo y puede ser usado como el archivo solicitado eliminando el fragmento `.example`.

3.  **Levantar los servicios con Docker Compose:**
    Desde la raíz del proyecto, ejecuta el siguiente comando. Este comando construirá las imágenes de los contenedores y los iniciará.

    ```bash
    docker compose up --build
    ```
4.   Si al ejecutar `docker compose up` en Windows ve un error como `no such file or directory`, probablemente se debe a que Windows cambió las terminaciones de línea de los scripts del proyecto (de `LF` a `CRLF`).  
Para solucionarlo fácilmente, abra el archivo afectado (en este caso `entrypoint.sh`) en Visual Studio Code, haga clic en `CRLF` (abajo a la derecha), cambie a `LF` y guarde el archivo.

    La primera vez que se ejecute, el servicio del backend preprocesará los datos de los libros, generará los embeddings y los insertará en la base de datos. Este proceso puede tardar unos minutos. En ejecuciones posteriores, el script verificará que no haya libros duplicados. Recuerde cargar el archivo `books.json` usando la opción **Cargar Libros** presente en la página principal del frontend antes de administrar la base de datos. 

## Acceso a los Servicios

Una vez que todos los contenedores estén en funcionamiento, podrás acceder a los diferentes componentes de la aplicación:

*   **Frontend:** [http://localhost:8080](http://localhost:8080)
*   **Documentación de la API:** [http://localhost:8000/docs](http://localhost:8000/docs)
*   **pgAdmin:** [http://localhost:5050](http://localhost:5050)
    *   **Usuario:** `admin@example.com`
    *   **Contraseña:** `admin`
