// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Function to add a message to the chat box
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender); // sender should be 'user', 'assistant', or 'error'

        if (sender === 'assistant') {
            // Parse Markdown, sanitize, and set as innerHTML for assistant messages
            const rawHtml = marked.parse(text);
            messageDiv.innerHTML = DOMPurify.sanitize(rawHtml);
        } else {
            // For user messages and errors, just set textContent
            messageDiv.textContent = text;
        }

        chatBox.appendChild(messageDiv);
        // Scroll to the bottom
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to handle sending a message
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') {
            return; // Don't send empty messages
        }

        // Display user message
        addMessage(messageText, 'user');
        userInput.value = ''; // Clear input field
        sendButton.disabled = true; // Disable button while waiting for response

        try {
            // Send message to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            if (!response.ok) {
                // Try to get error message from backend, otherwise show generic error
                let errorMsg = `Error: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMsg = `Error: ${errorData.error || errorMsg}`;
                } catch (e) {
                    // Ignore if response is not JSON
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();

            // Display assistant response
            if (data.answer) {
                addMessage(data.answer, 'assistant');
            } else if (data.error) {
                // Handle errors specifically returned in JSON
                 addMessage(`Backend Error: ${data.error}`, 'error');
            } else {
                 addMessage("Received an empty response from the server.", 'error');
            }

        } catch (error) {
            console.error('Fetch error:', error);
            addMessage(`Failed to connect or get response: ${error.message}`, 'error');
        } finally {
            sendButton.disabled = false; // Re-enable button
            userInput.focus(); // Put focus back to input field
        }
    }

    // Event listener for the send button
    sendButton.addEventListener('click', sendMessage);

    // Event listener for pressing Enter in the input field
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent default form submission (if any)
            sendMessage();
        }
    });

     // Initial focus on the input field
     userInput.focus();
});