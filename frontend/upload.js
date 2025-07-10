document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const fileNameSpan = document.getElementById('file-name');
    const logContainer = document.getElementById('log-container');
    const logOutput = document.getElementById('log-output');

    let selectedFile = null;

    fileInput.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            fileNameSpan.textContent = selectedFile.name;
            logContainer.style.display = 'none';
            logOutput.textContent = '';
        } else {
            fileNameSpan.textContent = 'Seleccionar archivo...';
        }
    });

    uploadBtn.addEventListener('click', () => {
        if (!selectedFile) {
            alert('Por favor, selecciona un archivo JSON primero.');
            return;
        }
        if (selectedFile.type !== 'application/json') {
            alert('El archivo debe ser de tipo JSON.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (event) => {
            try {
                const books = JSON.parse(event.target.result);
                if (!Array.isArray(books)) {
                    throw new Error('El contenido del JSON debe ser un array de libros.');
                }
                streamUpload(books);
            } catch (error) {
                logContainer.style.display = 'block';
                logOutput.textContent = `Error al leer el archivo JSON: ${error.message}`;
            }
        };
        reader.onerror = () => {
            logContainer.style.display = 'block';
            logOutput.textContent = 'No se pudo leer el archivo.';
        };
        reader.readAsText(selectedFile);
    });

    async function streamUpload(books) {
        logContainer.style.display = 'block';
        logOutput.textContent = '';
        uploadBtn.disabled = true;
        fileInput.disabled = true;

        try {
            const response = await fetch('http://localhost:8000/api/upload_books', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(books),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Error del servidor: ${response.status} ${errorText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    break;
                }
                const chunk = decoder.decode(value, { stream: true });
                logOutput.textContent += chunk;
                logContainer.scrollTop = logContainer.scrollHeight; // Auto-scroll
            }

        } catch (error) {
            logOutput.textContent += `\n--- ERROR ---\n${error.message}`;
        } finally {
            uploadBtn.disabled = false;
            fileInput.disabled = false;
        }
    }
});
