async function sendMessage() {
    const userInput = document.getElementById('user-input').value.trim();
    if (userInput === "") return;
    
    // Clear the input field
    document.getElementById('user-input').value = "";
    
    // Append user message
    appendMessage('user', userInput);
    
    // Show loading indicator
    const loadingElement = document.createElement('div');
    loadingElement.className = 'chat-message bot-message loading';
    loadingElement.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    document.getElementById('chat-messages').appendChild(loadingElement);
    
    // Scroll to the bottom of the chat
    scrollChatToBottom();
    
    // Make API call to process the message
    fetch('/api/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading indicator
        const loadingIndicator = document.querySelector('.loading');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
        
        // Handle the response
        if (data.success) {
            // Append the bot's message
            appendMessage('bot', data.response);
            
            // Check if recommendations are available
            if (data.has_recommendations || data.recommendations) {
                showRecommendationsButton();
                
                if (data.has_recommendations) {
                    appendMessage('bot', 'I've found some recommendations for you! Click the \'View Recommendations\' button to see them.');
                }
            }
        } else {
            // Display error message
            appendMessage('bot', 'Sorry, there was an error processing your message.');
        }
        
        // Scroll to the bottom of the chat
        scrollChatToBottom();
    })
    .catch(error => {
        console.error('Error:', error);
        // Remove loading indicator
        const loadingIndicator = document.querySelector('.loading');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
        // Display error message
        appendMessage('bot', 'Sorry, there was an error processing your message.');
        // Scroll to the bottom of the chat
        scrollChatToBottom();
    });
}

// Function to append a message to the chat
function appendMessage(sender, message) {
    // Get the chat messages container
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error('Chat messages container not found');
        return;
    }
    
    // Create a new message element
    const messageElement = document.createElement('div');
    messageElement.className = `chat-message ${sender}-message`;
    
    // Format the message with links
    let formattedMessage = message;
    
    // Convert URLs to clickable links
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    formattedMessage = formattedMessage.replace(urlRegex, function(url) {
        return `<a href="${url}" target="_blank">${url}</a>`;
    });
    
    messageElement.innerHTML = formattedMessage;
    
    // Append the message to the chat
    chatMessages.appendChild(messageElement);
    
    // Scroll to the bottom of the chat
    scrollChatToBottom();
}

function scrollChatToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function showRecommendationsButton() {
    // Show the recommendations button if it exists
    const recommendationsBtn = document.getElementById('view-recommendations-btn');
    if (recommendationsBtn) {
        recommendationsBtn.classList.remove('d-none');
        recommendationsBtn.classList.add('d-flex');
    }
}

function generateRecommendations() {
    // Show a loading message
    appendMessage('bot', 'Generating recommendations...');
    
    // Make API call to generate recommendations
    fetch('/api/recommendations', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        // Check if recommendations were generated successfully
        if (data && (Array.isArray(data) || data.companies || data.success)) {
            appendMessage('bot', 'Recommendations are ready! Click the \'View Recommendations\' button to see them.');
            showRecommendationsButton();
            
            // Optionally redirect to recommendations page
            // window.location.href = "/recommendations";
        } else if (data.error || data.message) {
            // Display error message
            appendMessage('bot', `Sorry, I couldn't generate recommendations: ${data.error || data.message}`);
        } else {
            // Generic error
            appendMessage('bot', 'Sorry, I couldn\'t generate recommendations at this time.');
        }
        
        // Scroll to the bottom of the chat
        scrollChatToBottom();
    })
    .catch(error => {
        console.error('Error generating recommendations:', error);
        appendMessage('bot', 'Sorry, there was an error generating recommendations.');
        scrollChatToBottom();
    });
}

// Listen for Enter key in the input field
document.addEventListener('DOMContentLoaded', function() {
    const inputField = document.getElementById('user-input');
    if (inputField) {
        inputField.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                sendMessage();
            }
        });
    }
    
    // Set up event listeners for recommendations button
    const recommendationsBtn = document.getElementById('view-recommendations-btn');
    if (recommendationsBtn) {
        recommendationsBtn.addEventListener('click', function() {
            window.location.href = '/recommendations';
        });
    }
    
    // Set up event listeners for generate recommendations button
    const generateRecommendationsBtn = document.getElementById('generate-recommendations-btn');
    if (generateRecommendationsBtn) {
        generateRecommendationsBtn.addEventListener('click', generateRecommendations);
    }
});

// Helper function to check if the microphone is available and has permission
async function checkMicrophonePermission() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        // Stop all tracks to release the microphone
        stream.getTracks().forEach(track => track.stop());
        return true;
    } catch (error) {
        console.error('Microphone permission error:', error);
        return false;
    }
}

// Initialize microphone permission check when page loads
document.addEventListener('DOMContentLoaded', async function() {
    const voiceButton = document.getElementById('voice-button');
    if (voiceButton) {
        // Check if microphone is supported
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('getUserMedia is not supported in this browser');
            voiceButton.disabled = true;
            voiceButton.title = 'Voice input not supported in your browser';
            return;
        }
        
        // Start with button enabled, permissions will be requested on click
        voiceButton.disabled = false;
    }
});

// Handle voice recognition
function setupVoiceRecognition() {
    const voiceButton = document.getElementById('voice-button');
    const voiceFeedback = document.getElementById('voice-feedback');
    const progressBar = document.getElementById('voice-progress');
    const statusText = document.getElementById('voice-status');
    
    if (!voiceButton) return;
    
    let recognition;
    try {
        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            let finalTranscript = '';
            let isListening = false;
            
            recognition.onstart = function() {
                isListening = true;
                voiceFeedback.classList.remove('d-none');
                statusText.textContent = 'Listening...';
                progressBar.style.width = '0%';
                voiceButton.classList.add('btn-danger');
                voiceButton.classList.remove('btn-primary');
                finalTranscript = '';
                
                // Animate progress bar
                let progress = 0;
                const interval = setInterval(() => {
                    if (!isListening) {
                        clearInterval(interval);
                        return;
                    }
                    progress += 1;
                    progressBar.style.width = `${Math.min(progress, 100)}%`;
                    if (progress >= 100) {
                        if (isListening) {
                            recognition.stop();
                        }
                        clearInterval(interval);
                    }
                }, 50);
            };
            
            recognition.onresult = function(event) {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                
                statusText.textContent = 'Listening: ' + (finalTranscript || interimTranscript);
            };
            
            recognition.onerror = function(event) {
                isListening = false;
                voiceFeedback.classList.add('d-none');
                voiceButton.classList.remove('btn-danger');
                voiceButton.classList.add('btn-primary');
                console.error('Speech recognition error', event.error);
                
                if (event.error === 'not-allowed') {
                    // Show microphone permission dialog
                    showMicrophonePermissionDialog();
                }
            };
            
            recognition.onend = function() {
                isListening = false;
                voiceFeedback.classList.add('d-none');
                voiceButton.classList.remove('btn-danger');
                voiceButton.classList.add('btn-primary');
                statusText.textContent = 'Speech recognition ended';
                
                if (finalTranscript) {
                    // Display the transcript in the message input
                    const messageInput = document.getElementById('message');
                    if (messageInput) {
                        messageInput.value = finalTranscript;
                    }
                    
                    // Get the current step from a hidden field or data attribute
                    const currentStep = document.getElementById('current-step')?.value || 'product';
                    
                    // Send the voice interaction to the backend
                    handleVoiceInteraction(finalTranscript, currentStep);
                }
            };
            
            voiceButton.addEventListener('click', function() {
                if (isListening) {
                    recognition.stop();
                } else {
                    try {
                        recognition.start();
                    } catch (e) {
                        console.error('Error starting recognition:', e);
                        showMicrophonePermissionDialog();
                    }
                }
            });
        } else {
            voiceButton.style.display = 'none';
            console.log('Speech recognition not supported');
        }
    } catch (e) {
        console.error('Error setting up voice recognition:', e);
        if (voiceButton) voiceButton.style.display = 'none';
    }
}

// Handle voice interaction with backend
async function handleVoiceInteraction(text, step) {
    try {
        // Show loading indicator
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            appendMessage('user', text);
            appendMessage('assistant', '<em>Processing...</em>');
        }
        
        // Call the API
        const response = await fetch('/api/voice_interaction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                step: step
            })
        });
        
        console.log('Voice interaction response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }
        
        const responseData = await response.json();
        
        // Update the chat with the assistant's response
        if (chatMessages) {
            // Remove the "Processing..." message
            chatMessages.removeChild(chatMessages.lastChild);
            
            // Add the actual response
            if (responseData.success) {
                appendMessage('assistant', responseData.text);
                
                // Update the current step
                if (responseData.next_step) {
                    const currentStepInput = document.getElementById('current-step');
                    if (currentStepInput) {
                        currentStepInput.value = responseData.next_step;
                    }
                }
                
                // Play audio if available
                if (responseData.audio) {
                    playAudio(responseData.audio);
                }
                
                // Show recommendations tab if available
                if (responseData.show_recommendations_tab) {
                    const recsButton = document.getElementById('view-recommendations-btn');
                    if (recsButton) {
                        recsButton.classList.remove('d-none');
                        recsButton.classList.add('d-block');
                    }
                }
            } else {
                appendMessage('assistant', responseData.text || 'Sorry, I encountered an error. Please try again.');
            }
        }
    } catch (error) {
        console.error('Error in voice interaction:', error);
        
        // Remove the "Processing..." message if it exists
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages && chatMessages.lastChild) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // Show error message
        appendMessage('assistant', 'Sorry, I encountered an error processing your voice input. Please try text input instead.');
    }
}

// Function to play audio from base64 string
function playAudio(base64Audio) {
    if (!base64Audio) return;
    
    try {
        const audio = new Audio();
        audio.src = 'data:audio/mp3;base64,' + base64Audio;
        audio.play().catch(e => console.error('Error playing audio:', e));
    } catch (e) {
        console.error('Error setting up audio playback:', e);
    }
}

// Function to show microphone permission dialog
function showMicrophonePermissionDialog() {
    const dialog = document.createElement('div');
    dialog.className = 'mic-permission-dialog';
    dialog.innerHTML = `
        <div class="mic-permission-content">
            <h4>Microphone Access Required</h4>
            <p>This app needs access to your microphone for voice input.</p>
            <p>Please click "Allow" when your browser asks for microphone permission.</p>
            <p>If you've already denied permission, you'll need to reset it in your browser settings.</p>
            <button id="mic-permission-close" class="btn btn-primary">Got it</button>
        </div>
    `;
    
    document.body.appendChild(dialog);
    
    document.getElementById('mic-permission-close').addEventListener('click', function() {
        dialog.remove();
    });
}

// Initialize voice recognition when the page loads
document.addEventListener('DOMContentLoaded', function() {
    setupVoiceRecognition();
    
    // Enable enter key to send message
    const messageInput = document.getElementById('message');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                document.getElementById('send-button').click();
            }
        });
    }
    
    // Request microphone permission on page load
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                // Permission granted, stop the stream
                stream.getTracks().forEach(track => track.stop());
                console.log('Microphone permission granted');
            })
            .catch(function(err) {
                console.error('Microphone permission denied:', err);
                showMicrophonePermissionDialog();
            });
    }
}); 