document.addEventListener('DOMContentLoaded', () => {
    // Elementos del DOM
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const chatLoader = document.getElementById('chat-loader');
    
    // Configuraciones
    const chatKInput = document.getElementById('chat-k');
    const chatKValue = document.getElementById('chat-k-value');
    const chatThresholdInput = document.getElementById('chat-threshold');
    const chatThresholdValue = document.getElementById('chat-threshold-value');
    const useStreamingCheckbox = false; // No se usa en este momento, pero se puede activar si se desea

    // Estado del chat
    let isProcessing = false;
    let currentStreamMessage = null;

    // Actualizar valores de configuración
    chatKInput.addEventListener('input', () => {
        chatKValue.textContent = chatKInput.value;
    });

    chatThresholdInput.addEventListener('input', () => {
        const percentage = Math.round(chatThresholdInput.value * 100);
        chatThresholdValue.textContent = `${percentage}%`;
    });

    // Manejar envío del formulario
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!isProcessing && chatInput.value.trim()) {
            sendMessage();
        }
    });

    // Función principal para enviar mensaje
    async function sendMessage() {
        const question = chatInput.value.trim();
        if (!question) return;

        // Limpiar respuesta anterior
        clearPreviousResponse();
        
        // Mostrar mensaje del usuario
        addUserMessage(question);
        
        // Limpiar input y deshabilitar
        chatInput.value = '';
        setProcessingState(true);
        
        // Configurar parámetros
        const params = {
            question: question,
            k: parseInt(chatKInput.value),
            similarity_threshold: parseFloat(chatThresholdInput.value)
        };

        try {
            if (useStreamingCheckbox.checked) {
                await sendStreamingMessage(params);
            } else {
                await sendRegularMessage(params);
            }
        } catch (error) {
            console.error('Error al enviar mensaje:', error);
            addErrorMessage('Lo siento, ha ocurrido un error. Por favor, intenta de nuevo.');
        } finally {
            setProcessingState(false);
        }
    }

    // Enviar mensaje con streaming
    async function sendStreamingMessage(params) {
        const response = await fetch('http://localhost:8000/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            throw new Error(`Error del servidor: ${response.status}`);
        }

        // Crear mensaje del bot para streaming
        const messageElement = addBotMessage('', true);
        currentStreamMessage = messageElement;
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            accumulatedText += chunk;
            
            // Actualizar el contenido del mensaje
            updateStreamMessage(messageElement, accumulatedText);
        }

        // Finalizar streaming
        finalizeStreamMessage(messageElement);
    }

    // Enviar mensaje regular
    async function sendRegularMessage(params) {
        const response = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            throw new Error(`Error del servidor: ${response.status}`);
        }

        const result = await response.json();
        
        // Mostrar respuesta del bot
        addBotMessage(result.answer);
        
        // Mostrar libros recomendados si existen
        if (result.recommended_books && result.recommended_books.length > 0) {
            addRecommendedBooks(result.recommended_books);
        }
    }

    // Agregar mensaje del usuario
    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message user-message';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-bubble">
                    ${escapeHtml(text)}
                </div>
                <div class="message-time">${getCurrentTime()}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Agregar mensaje del bot
    function addBotMessage(text, isStreaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message bot-message ${isStreaming ? 'streaming' : ''}`;
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-bubble">
                    <div class="message-text">${isStreaming ? '' : formatBotMessage(text)}</div>
                    ${isStreaming ? '<div class="typing-indicator">...</div>' : ''}
                </div>
                <div class="message-time">${getCurrentTime()}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv;
    }

    // Actualizar mensaje durante streaming
    function updateStreamMessage(messageElement, text) {
        const messageText = messageElement.querySelector('.message-text');
        
        // Formatear el texto y mantener el indicador de escritura
        messageText.innerHTML = formatBotMessage(text);
        
        scrollToBottom();
    }

    // Finalizar mensaje de streaming
    function finalizeStreamMessage(messageElement) {
        const typingIndicator = messageElement.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        messageElement.classList.remove('streaming');
        currentStreamMessage = null;
    }

    // Agregar libros recomendados
    function addRecommendedBooks(books) {
        const booksDiv = document.createElement('div');
        booksDiv.className = 'chat-message bot-message books-recommendation';
        
        let booksHtml = books.map(book => `
            <div class="recommended-book">
                <div class="book-info">
                    <h4>${escapeHtml(book.title)}</h4>
                    <p><strong>Autor:</strong> ${escapeHtml(book.author)}</p>
                    <p class="similarity">Similitud: ${Math.round(book.similarity_score * 100)}%</p>
                </div>
            </div>
        `).join('');
        
        booksDiv.innerHTML = `
            <div class="message-content">
                <div class="message-bubble books-bubble">
                    <h3>Libros recomendados:</h3>
                    <div class="books-list">
                        ${booksHtml}
                    </div>
                </div>
                <div class="message-time">${getCurrentTime()}</div>
            </div>
        `;
        
        chatMessages.appendChild(booksDiv);
        scrollToBottom();
    }

    // Agregar mensaje de error
    function addErrorMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message error-message';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-bubble error-bubble">
                    ${escapeHtml(text)}
                </div>
                <div class="message-time">${getCurrentTime()}</div>
            </div>
        `;
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Limpiar respuesta anterior
    function clearPreviousResponse() {
        // Remover todos los mensajes del bot
        const botMessages = chatMessages.querySelectorAll('.chat-message.bot-message, .chat-message.books-recommendation, .chat-message.error-message');
        botMessages.forEach(message => message.remove());
    }

    // Establecer estado de procesamiento
    function setProcessingState(processing) {
        isProcessing = processing;
        chatInput.disabled = processing;
        chatSendBtn.disabled = processing;
        
        if (processing) {
            chatLoader.style.display = 'flex';
            chatSendBtn.textContent = 'Procesando...';
        } else {
            chatLoader.style.display = 'none';
            chatSendBtn.textContent = 'Preguntar';
        }
    }

    // Formatear mensaje del bot
    function formatBotMessage(text) {
        // Reemplazar saltos de línea con <br> y escapar HTML
        let formatted = escapeHtml(text).replace(/\n/g, '<br>');
        
        // Hacer texto en negrita más visible
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        return formatted;
    }

    // Utilidades
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function getCurrentTime() {
        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    // Inicialización
    chatInput.focus();
});