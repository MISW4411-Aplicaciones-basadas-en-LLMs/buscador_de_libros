# Usa una imagen base de Python delgada.
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia el archivo de requerimientos primero para aprovechar el cache de Docker.
COPY requirements.txt .

# Instala las dependencias de Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del codigo de la aplicacion al directorio de trabajo.
COPY . .

# Expone el puerto en el que se ejecuta la aplicacion.
EXPOSE 8000

# Copia el script de entrypoint y lo hace el punto de entrada.
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
