document.addEventListener('DOMContentLoaded', () => {
    const chatbox = document.getElementById('chatbox');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const sessionIdDisplay = document.getElementById('session-id');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');

    // --- Configuration ---
    const API_BASE_URL = 'http://localhost:8000'; // CHANGE IF YOUR API IS ELSEWHERE
    // ---------------------

    let currentSessionId = '';

    // --- Initialize ---
    function initializeChat() {
        // Generate a simple UUID for the session
        currentSessionId = generateUUID();
        sessionIdDisplay.textContent = currentSessionId;
        console.log("Chat initialized with Session ID:", currentSessionId);
        // You could add logic here to load history from localStorage if needed
    }

    // --- Helper Functions ---
    function generateUUID() { // Public Domain/MIT
        var d = new Date().getTime();//Timestamp
        var d2 = ((typeof performance !== 'undefined') && performance.now && (performance.now()*1000)) || 0;//Time in microseconds since page-load or 0 if unsupported
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16;//random number between 0 and 16
            if(d > 0){//Use timestamp until depleted
                r = (d + r)%16 | 0;
                d = Math.floor(d/16);
            } else {//Use microseconds since page-load if supported
                r = (d2 + r)%16 | 0;
                d2 = Math.floor(d2/16);
            }
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
    }

    function displayMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);

        if (sender === 'bot') {
            // Use Marked to convert Markdown to HTML
            // Use DOMPurify to sanitize the HTML before inserting
            const rawHtml = marked.parse(message);
            messageElement.innerHTML = DOMPurify.sanitize(rawHtml);
        } else {
            // For user messages, just set text content to prevent XSS
            messageElement.textContent = message;
        }

        chatbox.appendChild(messageElement);
        // Scroll to the bottom
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function setLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
        sendBtn.disabled = isLoading;
        messageInput.disabled = isLoading;
    }

    function displayError(message) {
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block';
        // Optionally hide after a few seconds
        setTimeout(() => {
             errorMessageDiv.style.display = 'none';
             errorMessageDiv.textContent = '';
        }, 5000);
    }

    // --- Event Handlers ---
    async function handleSendMessage() {
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        displayMessage(userMessage, 'user');
        messageInput.value = '';
        setLoading(true);
        errorMessageDiv.style.display = 'none'; // Hide previous errors

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json' // Good practice to accept JSON
                },
                body: JSON.stringify({
                    query: userMessage,
                    session_id: currentSessionId
                })
            });

            setLoading(false);

            if (!response.ok) {
                // Try to get error detail from response body
                let errorDetail = `HTTP error! Status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || JSON.stringify(errorData);
                } catch (e) {
                    // If response is not JSON or empty
                     errorDetail += ` - ${response.statusText}`;
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            displayMessage(data.response, 'bot');
            // Verify session ID hasn't changed unexpectedly (optional)
            if(data.session_id !== currentSessionId) {
                console.warn("Session ID mismatch from server response!");
                // Decide if you want to update it:
                // currentSessionId = data.session_id;
                // sessionIdDisplay.textContent = currentSessionId;
            }

        } catch (error) {
            console.error('Error sending message:', error);
            setLoading(false);
            displayError(`Failed to get response: ${error.message}`);
            // Optionally display a generic error message to the user in the chat
            // displayMessage(`Error: ${error.message}`, 'bot'); // Or a more user-friendly message
        }
    }

    async function handleClearHistory() {
        if (!currentSessionId) {
            displayError("No session ID found.");
            return;
        }
        if (!confirm("Are you sure you want to clear the chat history for this session?")) {
             return;
        }

        console.log(`Attempting to clear history for session: ${currentSessionId}`);
        setLoading(true); // Indicate activity
        errorMessageDiv.style.display = 'none';

        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/${currentSessionId}/history`, {
                method: 'DELETE',
                headers: {
                    'Accept': 'application/json' // Although 204 has no body, useful for error handling
                }
            });

            setLoading(false);

            // Status 204 means success (No Content)
            if (response.status === 204) {
                chatbox.innerHTML = ''; // Clear the visual chatbox
                displayMessage("Chat history cleared.", "bot"); // Confirmation message
                console.log(`History cleared for session: ${currentSessionId}`);
            } else {
                 // Try to get error detail
                let errorDetail = `Failed to clear history. Status: ${response.status}`;
                 try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || JSON.stringify(errorData);
                 } catch (e) {
                     errorDetail += ` - ${response.statusText}`;
                 }
                throw new Error(errorDetail);
            }

        } catch (error) {
            console.error('Error clearing history:', error);
            setLoading(false);
            displayError(`Failed to clear history: ${error.message}`);
        }
    }

    // --- Attach Event Listeners ---
    sendBtn.addEventListener('click', handleSendMessage);
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handleSendMessage();
        }
    });
    clearHistoryBtn.addEventListener('click', handleClearHistory);

    // --- Start the application ---
    initializeChat();
});