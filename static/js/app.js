// Global variables
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isListening = false;
let selectedSector = '';
let selectedProduct = '';
let isProcessingInterim = false;
let interimTranscript = '';
let finalTranscript = '';
let interimProcessTimeout;
let lastInterimProcessed = '';
let speechRecognition = null;
let chatVoiceBox = null;
let currentQuestion = '';
let questionQueue = [];
let isAnswering = false;

// Audio recording configuration
const audioConfig = {
    type: 'audio/webm;codecs=opus',
    sampleRate: 16000,
    channelCount: 1,
    bitsPerSecond: 16000
};

// DOM Elements
const recordButton = document.getElementById('recordButton');
const status = document.getElementById('status');
const errorMessage = document.getElementById('errorMessage');
const transcriptContainer = document.getElementById('transcriptContainer');
const assistantMessage = document.getElementById('assistantMessage');
const insightsContainer = document.getElementById('insightsContainer');
const thinkingProcess = document.getElementById('thinkingProcess');
const thinkingContainer = document.getElementById('thinkingContainer');
const chatContainer = document.getElementById('chatContainer');
const keywordsContainer = document.getElementById('keywordsContainer');

// Event Listeners
recordButton?.addEventListener('click', toggleRecording);

// Initialize speech recognition
function initSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        speechRecognition = new SpeechRecognition();
        speechRecognition.continuous = false;
        speechRecognition.interimResults = true;
        speechRecognition.lang = 'en-US';
        
        speechRecognition.onstart = function() {
            isListening = true;
            updateUI('listening');
            
            // Reset transcripts
            interimTranscript = '';
            finalTranscript = '';
            
            // Show voice indicator if it exists
            if (document.getElementById('voice-indicator')) {
                document.getElementById('voice-indicator').style.display = 'block';
            }
        };
        
        speechRecognition.onresult = function(event) {
            // Get interim results
            interimTranscript = '';
            finalTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            
            // Update transcript display if it exists
            if (transcriptContainer) {
                transcriptContainer.style.display = 'block';
                transcriptContainer.textContent = `You said: ${finalTranscript || interimTranscript}`;
            }
            
            // Process interim results for real-time interaction
            if (interimTranscript && interimTranscript.length > 10 && interimTranscript !== lastInterimProcessed) {
                processInterimTranscript(interimTranscript);
            }
        };
        
        speechRecognition.onend = function() {
            isListening = false;
            updateUI('ready');
            
            // Hide voice indicator if it exists
            if (document.getElementById('voice-indicator')) {
                document.getElementById('voice-indicator').style.display = 'none';
            }
            
            // Process final transcript if available
            if (finalTranscript) {
                processVoiceInteraction(finalTranscript);
            }
        };
        
        speechRecognition.onerror = function(event) {
            console.error('Speech recognition error', event.error);
            isListening = false;
            updateUI('ready');
            
            // Show error message
            if (errorMessage) {
                errorMessage.textContent = event.error === 'not-allowed' 
                    ? 'Microphone access denied. Please check your browser permissions.'
                    : 'Voice recognition error. Please try again.';
                errorMessage.style.display = 'block';
            }
        };
        
        console.log('Speech recognition initialized successfully');
        return speechRecognition;
    } else {
        console.warn('Speech recognition not supported in this browser');
        return null;
    }
}

// Toggle speech recognition
function toggleSpeechRecognition() {
    if (!speechRecognition) {
        speechRecognition = initSpeechRecognition();
        if (!speechRecognition) {
            if (errorMessage) {
                errorMessage.textContent = 'Speech recognition is not supported in your browser.';
                errorMessage.style.display = 'block';
            }
            return;
        }
    }
    
    if (isListening) {
        speechRecognition.stop();
    } else {
        speechRecognition.start();
    }
}

// Toggle recording
function toggleRecording() {
    // If speech recognition is available, use it
    if (speechRecognition !== null) {
        toggleSpeechRecognition();
        return;
    }
    
    // Otherwise, fall back to MediaRecorder
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Start recording audio
function startRecording() {
    if (isRecording) return;
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                processAudio();
            };
            
            audioChunks = [];
            mediaRecorder.start();
            isRecording = true;
            updateUI('recording');
            
            // Show voice indicator if it exists
            if (document.getElementById('voice-indicator')) {
                document.getElementById('voice-indicator').style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error accessing microphone:', error);
            handleError(new Error('Microphone access denied. Please check your browser permissions.'));
        });
}

// Stop recording audio
function stopRecording() {
    if (!isRecording || !mediaRecorder) return;
    
    mediaRecorder.stop();
    isRecording = false;
    updateUI('processing');
    
    // Hide voice indicator if it exists
    if (document.getElementById('voice-indicator')) {
        document.getElementById('voice-indicator').style.display = 'none';
    }
}

// Function to process interim transcript for real-time interaction
function processInterimTranscript(transcript) {
    // Avoid processing the same transcript multiple times or processing too frequently
    if (isProcessingInterim || transcript === lastInterimProcessed) {
        return;
    }
    
    // Clear any existing timeout
    if (interimProcessTimeout) {
        clearTimeout(interimProcessTimeout);
    }
    
    // Set a timeout to avoid too many API calls
    interimProcessTimeout = setTimeout(() => {
        // Only process if we have enough content and not already processing
        if (transcript.length > 10 && !isProcessingInterim) {
            isProcessingInterim = true;
            lastInterimProcessed = transcript;
            
            const company = document.getElementById('companyInput')?.value;
            if (!company) {
                isProcessingInterim = false;
                return;
            }
            
            // Call the voice interaction API
            fetch('/api/voice_interaction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: transcript,
                    step: 'research',
                    company: company,
                    sector: selectedSector || '',
                    product: selectedProduct || ''
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show the thinking process if available
                    if (data.thinking_process && thinkingProcess && thinkingContainer) {
                        thinkingProcess.textContent = data.thinking_process;
                        thinkingContainer.style.display = 'block';
                    }
                    
                    // Play audio response if available
                    if (data.audio) {
                        playAudioResponse(data.audio);
                    }
                }
                isProcessingInterim = false;
            })
            .catch(error => {
                console.error('Error processing interim transcript:', error);
                isProcessingInterim = false;
            });
        }
    }, 1000); // Wait 1 second before processing to avoid too many API calls
}

// Function to process voice interaction with the API
function processVoiceInteraction(transcript) {
    const company = document.getElementById('companyInput')?.value;
    if (!company) {
        return;
    }
    
    // Add the user's message to the chat if chat container exists
    if (chatContainer) {
        addMessageToChat('user', transcript);
    }
    
    // Call the voice interaction API
    fetch('/api/voice_interaction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: transcript,
            step: 'research',
            company: company,
            sector: selectedSector || '',
            product: selectedProduct || ''
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Add the assistant's response to the chat
            addMessageToChat('assistant', data.text);
            
            // Show the thinking process if available
            if (data.thinking_process && thinkingProcess && thinkingContainer) {
                thinkingProcess.textContent = data.thinking_process;
                thinkingContainer.style.display = 'block';
            }
            
            // Play audio response if available
            if (data.audio) {
                playAudioResponse(data.audio);
            }
            
            // Update assistant message
            if (assistantMessage) {
                assistantMessage.style.display = 'block';
                assistantMessage.textContent = data.text;
            }
            
            // Update keywords if available
            if (data.keywords && data.keywords.length > 0) {
                updateKeywords(data.keywords);
            }
        }
    })
    .catch(error => {
        console.error('Error processing voice interaction:', error);
    });
}

// Function to play audio response
function playAudioResponse(audioBase64) {
    const audio = new Audio(`data:audio/mpeg;base64,${audioBase64}`);
    audio.play().catch(error => {
        console.error('Error playing audio:', error);
    });
}

// Function to update keywords display
function updateKeywords(keywords) {
    console.log("Updating keywords in DOM:", keywords);
    const keywordsContainer = document.getElementById('keywordsContainer');
    if (!keywordsContainer) return;
    
    // Clear existing keywords
    keywordsContainer.innerHTML = '';
    
    // Add heading if not already present
    if (!document.getElementById('keywordsHeading')) {
        const heading = document.createElement('h4');
        heading.id = 'keywordsHeading';
        heading.className = 'text-lg font-semibold mb-2';
        heading.textContent = 'Relevant Research Keywords';
        keywordsContainer.appendChild(heading);
    }
    
    // Create keywords container if not already present
    let keywordsList = document.getElementById('keywordsList');
    if (!keywordsList) {
        keywordsList = document.createElement('div');
        keywordsList.id = 'keywordsList';
        keywordsList.className = 'keywords-container';
        keywordsContainer.appendChild(keywordsList);
    } else {
        keywordsList.innerHTML = '';
    }
    
    // Add keywords
    keywords.forEach(keyword => {
        const keywordElement = document.createElement('span');
        keywordElement.className = 'keyword';
        keywordElement.textContent = keyword;
        keywordsList.appendChild(keywordElement);
    });
    
    // Show the keywords container
    keywordsContainer.style.display = 'block';
}

// Function to add message to chat
function addMessageToChat(role, message) {
    if (!chatContainer) return;
    
    // Add user message if this is an assistant response and no user message exists yet
    if (role === 'assistant' && chatContainer.querySelectorAll('.message.user').length === 0) {
        const userInput = document.getElementById('companyInput')?.value || 'Tell me about this company';
        const userMessage = document.createElement('div');
        userMessage.className = 'message user';
        userMessage.textContent = userInput;
        chatContainer.appendChild(userMessage);
    }
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}`;
    messageElement.textContent = message;
    chatContainer.appendChild(messageElement);
    
    // Scroll to the bottom of the chat
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Make chat container visible if it's hidden
    chatContainer.style.display = 'block';
}

async function processAudio() {
    try {
        updateUI('processing');

        // Create audio blob
        const audioBlob = new Blob(audioChunks, { type: audioConfig.type });
        const reader = new FileReader();

        reader.onload = async () => {
            try {
                const base64Audio = reader.result.split(',')[1];
                const company = document.getElementById('companyInput')?.value;

                if (!company) {
                    throw new Error('Please enter a company name first.');
                }

                // Prepare request data
                const requestData = {
                    audio: base64Audio,
                    format: 'webm',
                    codec: 'opus',
                    sampleRate: audioConfig.sampleRate
                };

                // Build query parameters
                const queryParams = new URLSearchParams({
                    company: company,
                    sector: selectedSector || '',
                    product: selectedProduct || ''
                });

                // Send audio to server
                const response = await fetch(`/process_audio?${queryParams}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                });

                // Handle response
                const data = await response.json();
                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Failed to process audio');
                }

                // Update UI with results
                updateUIWithResults(data);

            } catch (err) {
                console.error('Error in processAudio:', err);
                handleError(err);
            }
        };

        reader.onerror = () => {
            handleError(new Error('Failed to read audio data'));
        };

        reader.readAsDataURL(audioBlob);

    } catch (err) {
        console.error('Error in processAudio:', err);
        handleError(err);
    }
}

// UI Update Functions
function updateUI(state) {
    if (!recordButton || !status) return;

    switch (state) {
        case 'recording':
            recordButton.classList.add('recording');
            status.textContent = 'Recording... Click to stop';
            errorMessage.style.display = 'none';
            break;

        case 'processing':
            recordButton.classList.remove('recording');
            status.textContent = 'Processing audio...';
            break;

        case 'ready':
            recordButton.classList.remove('recording');
            status.textContent = 'Ready to record';
            break;

        case 'listening':
            recordButton.classList.remove('recording');
            status.textContent = 'Listening...';
            break;

        default:
            recordButton.classList.remove('recording');
            status.textContent = 'Click to start recording';
    }
}

function updateUIWithResults(data) {
    // Show transcript
    if (transcriptContainer) {
        transcriptContainer.style.display = 'block';
        transcriptContainer.textContent = `You said: ${data.transcript}`;
    }

    // Show assistant response
    if (assistantMessage) {
        assistantMessage.style.display = 'block';
        assistantMessage.textContent = data.response.text;
    }

    // Show insights
    if (insightsContainer && data.insights?.html) {
        insightsContainer.innerHTML = data.insights.html;
    }

    updateUI('ready');
}

function handleError(error) {
    console.error('Error:', error);
    
    if (errorMessage) {
        errorMessage.textContent = error.message || 'An error occurred. Please try again.';
        errorMessage.style.display = 'block';
    }
    
    updateUI('ready');
}

// Company and Product Selection Functions
function submitCompany() {
    const company = document.getElementById('companyInput')?.value;
    if (!company) {
        handleError(new Error('Please enter a company name'));
        return;
    }
    showStep(2);
}

function selectProduct(product) {
    selectedProduct = product;
    showStep(3);
}

function selectSector(sector) {
    selectedSector = sector;
    showStep(4);
}

function showStep(stepNumber) {
    document.querySelectorAll('.step-container').forEach(container => {
        container.classList.remove('active');
    });
    const stepElement = document.getElementById(`step${stepNumber}`);
    if (stepElement) {
        stepElement.classList.add('active');
    }
}

// Chat Voice Box Component
function initChatVoiceBox() {
    // Create chat voice box container if it doesn't exist
    if (!document.getElementById('chat-voice-box')) {
        chatVoiceBox = document.createElement('div');
        chatVoiceBox.id = 'chat-voice-box';
        chatVoiceBox.className = 'chat-voice-box';
        chatVoiceBox.innerHTML = `
            <div class="chat-voice-header">
                <h3>Voice Assistant</h3>
                <button id="close-chat-voice" class="close-btn">×</button>
            </div>
            <div id="chat-voice-messages" class="chat-voice-messages"></div>
            <div class="chat-voice-controls">
                <button id="chat-voice-btn" class="voice-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="23"></line>
                        <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                </button>
                <input type="text" id="chat-voice-input" placeholder="Type your response...">
                <button id="chat-voice-send" class="send-btn">Send</button>
            </div>
        `;
        
        document.body.appendChild(chatVoiceBox);
        
        // Add event listeners
        document.getElementById('close-chat-voice').addEventListener('click', toggleChatVoiceBox);
        document.getElementById('chat-voice-btn').addEventListener('click', toggleChatVoiceRecording);
        document.getElementById('chat-voice-send').addEventListener('click', sendChatVoiceResponse);
        document.getElementById('chat-voice-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatVoiceResponse();
            }
        });
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .chat-voice-box {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 350px;
                max-height: 500px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                display: flex;
                flex-direction: column;
                z-index: 1000;
                overflow: hidden;
                display: none;
            }
            
            .chat-voice-header {
                padding: 15px;
                background-color: #4F46E5;
                color: white;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .chat-voice-header h3 {
                margin: 0;
                font-size: 16px;
            }
            
            .close-btn {
                background: none;
                border: none;
                color: white;
                font-size: 20px;
                cursor: pointer;
            }
            
            .chat-voice-messages {
                padding: 15px;
                overflow-y: auto;
                flex-grow: 1;
                max-height: 350px;
            }
            
            .chat-voice-message {
                margin-bottom: 15px;
                padding: 10px 15px;
                border-radius: 18px;
                max-width: 80%;
                word-wrap: break-word;
            }
            
            .chat-voice-message.assistant {
                background-color: #F3F4F6;
                align-self: flex-start;
                border-bottom-left-radius: 4px;
            }
            
            .chat-voice-message.user {
                background-color: #4F46E5;
                color: white;
                align-self: flex-end;
                margin-left: auto;
                border-bottom-right-radius: 4px;
            }
            
            .chat-voice-controls {
                display: flex;
                padding: 10px;
                border-top: 1px solid #E5E7EB;
                background-color: #F9FAFB;
            }
            
            .chat-voice-controls input {
                flex-grow: 1;
                padding: 8px 12px;
                border: 1px solid #D1D5DB;
                border-radius: 20px;
                margin: 0 10px;
            }
            
            .voice-btn, .send-btn {
                background-color: #4F46E5;
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
            }
            
            .voice-btn.recording {
                background-color: #EF4444;
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
            }
        `;
        document.head.appendChild(style);
    }
}

// Toggle chat voice box visibility
function toggleChatVoiceBox() {
    if (!chatVoiceBox) {
        initChatVoiceBox();
    }
    
    if (chatVoiceBox.style.display === 'flex') {
        chatVoiceBox.style.display = 'none';
    } else {
        chatVoiceBox.style.display = 'flex';
        
        // If no question is currently being asked, ask the first question
        if (!currentQuestion && questionQueue.length === 0) {
            askInitialQuestion();
        }
    }
}

// Ask initial question
function askInitialQuestion() {
    // Add initial questions to the queue
    questionQueue = [
        "What company would you like to research today?",
        "What's your role or context for researching this company?",
        "Are there any specific aspects of the company you're interested in?"
    ];
    
    // Ask the first question
    askNextQuestion();
}

// Ask the next question in the queue
function askNextQuestion() {
    if (questionQueue.length === 0) {
        console.log("No more questions in queue");
        return;
    }
    
    // Get the next question from the queue
    currentQuestion = questionQueue.shift();
    console.log("Asking next question:", currentQuestion);
    
    // Add the question to the chat
    addChatVoiceMessage('assistant', currentQuestion);
    
    // Play the question as audio if TTS is enabled
    if (enableTTS) {
        // Convert the question to speech
        fetch('/api/text_to_speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: currentQuestion
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.audio) {
                // Play the audio
                playAudioResponse(data.audio);
                
                // Start listening for a response after a short delay
                // This ensures we don't start listening while the question is still being spoken
                setTimeout(() => {
                    startListening();
                }, 500);
            } else {
                console.error("Error converting text to speech:", data.error);
                // Start listening anyway since we can't play audio
                startListening();
            }
        })
        .catch(error => {
            console.error('Error converting text to speech:', error);
            // Start listening anyway since we can't play audio
            startListening();
        });
    } else {
        // Start listening immediately if TTS is disabled
        startListening();
    }
    
    // Save this question to the user journey for tracking
    const timestamp = new Date().toISOString();
    const data = {
        timestamp: timestamp,
        userInput: "SYSTEM_QUESTION",
        keywords: currentQuestion,
        question_type: "system_question"
    };
    
    // Send data to server to save in CSV
    fetch('/api/save_interaction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("System question saved to user journey");
        } else {
            console.error("Failed to save system question:", data.error);
        }
    })
    .catch(error => {
        console.error('Error saving system question:', error);
    });
}

// Toggle chat voice recording
function toggleChatVoiceRecording() {
    const voiceBtn = document.getElementById('chat-voice-btn');
    
    if (isListening || isRecording) {
        // Stop recording
        if (speechRecognition) {
            speechRecognition.stop();
        } else if (mediaRecorder) {
            stopRecording();
        }
        
        voiceBtn.classList.remove('recording');
    } else {
        // Start recording
        voiceBtn.classList.add('recording');
        
        // Initialize speech recognition for chat voice box
        if (!speechRecognition) {
            speechRecognition = initSpeechRecognition();
        }
        
        if (speechRecognition) {
            // Override onresult to handle chat voice box responses
            speechRecognition.onresult = function(event) {
                // Get interim results
                interimTranscript = '';
                finalTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript;
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                
                // Update input field with current transcript
                document.getElementById('chat-voice-input').value = finalTranscript || interimTranscript;
            };
            
            // Override onend to handle chat voice box responses
            speechRecognition.onend = function() {
                isListening = false;
                document.getElementById('chat-voice-btn').classList.remove('recording');
                
                // Process final transcript if available
                const transcript = document.getElementById('chat-voice-input').value.trim();
                if (transcript) {
                    sendChatVoiceResponse();
                }
            };
            
            speechRecognition.start();
        } else {
            // Fall back to MediaRecorder
            startRecording();
        }
    }
}

// Send chat voice response
function sendChatVoiceResponse() {
    const input = document.getElementById('chat-voice-input');
    const response = input.value.trim();
    
    if (!response) {
        return;
    }
    
    // Add user response to chat
    addChatVoiceMessage('user', response);
    
    // Clear input
    input.value = '';
    
    // Process the response
    processChatVoiceResponse(response);
    
    // Auto-trigger next question if user clicks send button
    if (document.activeElement === document.getElementById('chat-voice-send')) {
        setTimeout(() => {
            if (!isAnswering && questionQueue.length > 0) {
                askNextQuestion();
            }
        }, 300);
    }
}

// Process chat voice response
function processChatVoiceResponse(response) {
    // If we're already processing a response, queue this one
    if (isAnswering) {
        console.log("Already processing a response, skipping this one");
        return;
    }
    
    isAnswering = true;
    console.log("Processing chat voice response:", response);
    
    // Get all context from the form fields
    const productContext = document.getElementById('product-input')?.value || '';
    const marketContext = document.getElementById('market-input')?.value || '';
    const companySizeContext = document.getElementById('company-size-input')?.value || '';
    
    // Call the voice interaction API
    fetch('/api/voice_interaction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: response,
            question: currentQuestion,
            step: 'chat',
            product: productContext,
            sector: marketContext,
            company_size: companySizeContext
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received response data:", data);
        if (data.success) {
            // Add the assistant's response to the chat
            addChatVoiceMessage('assistant', data.text);
            
            // Update keywords immediately if available
            if (data.keywords && data.keywords.length > 0) {
                console.log("Updating keywords:", data.keywords);
                updateKeywords(data.keywords);
            } else {
                console.warn("No keywords returned from server");
                // Generate fallback keywords if none were returned
                const fallbackKeywords = ["B2B", "Sales", "Research", "Business Intelligence"];
                updateKeywords(fallbackKeywords);
            }
            
            // Save user interaction to CSV with all context
            const interactionData = {
                timestamp: new Date().toISOString(),
                userInput: response,
                keywords: data.keywords ? data.keywords.join(', ') : '',
                question: currentQuestion,
                product: productContext,
                market: marketContext,
                company_size: companySizeContext,
                assistant_response: data.text
            };
            
            // Send complete interaction data to server
            fetch('/api/save_interaction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(interactionData)
            })
            .then(response => response.json())
            .then(saveData => {
                if (saveData.success) {
                    console.log("User interaction saved to CSV with full context");
                } else {
                    console.error("Failed to save user interaction:", saveData.error);
                }
            })
            .catch(error => {
                console.error('Error saving user interaction:', error);
            });
            
            // If there are follow-up questions, add them to the queue
            if (data.follow_up_questions && data.follow_up_questions.length > 0) {
                console.log("Adding follow-up questions to queue:", data.follow_up_questions);
                questionQueue = [...questionQueue, ...data.follow_up_questions];
            }
            
            // Play audio response if available
            if (data.audio) {
                playAudioResponse(data.audio);
                
                // Wait for audio to finish (or a short time) then move to next question
                setTimeout(() => {
                    isAnswering = false;
                    
                    // Automatically trigger the next question immediately
                    if (questionQueue.length > 0) {
                        console.log("Auto-jumping to next question");
                        askNextQuestion();
                    }
                }, 100); // Very short delay to ensure UI updates before moving on
            } else {
                // No audio, move to next question immediately
                isAnswering = false;
                if (questionQueue.length > 0) {
                    console.log("Auto-jumping to next question (no audio)");
                    askNextQuestion();
                }
            }
        } else {
            console.error("Error in response:", data);
            isAnswering = false;
        }
    })
    .catch(error => {
        console.error('Error processing chat voice response:', error);
        isAnswering = false;
    });
}

// Add message to chat voice box
function addChatVoiceMessage(role, message) {
    const messagesContainer = document.getElementById('chat-voice-messages');
    if (!messagesContainer) return;
    
    const messageElement = document.createElement('div');
    messageElement.className = `chat-voice-message ${role}`;
    messageElement.textContent = message;
    
    messagesContainer.appendChild(messageElement);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Function to update keywords with improved visibility
function updateKeywords(keywords) {
    console.log("Updating keywords in DOM:", keywords);
    const keywordsContainer = document.getElementById('keywordsContainer');
    if (!keywordsContainer) {
        console.error("Keywords container not found");
        return;
    }
    
    // Clear existing keywords
    keywordsContainer.innerHTML = '';
    
    // Add heading if not already present
    if (!document.getElementById('keywordsHeading')) {
        const heading = document.createElement('h4');
        heading.id = 'keywordsHeading';
        heading.className = 'text-lg font-semibold mb-2';
        heading.textContent = 'Relevant Research Keywords';
        keywordsContainer.appendChild(heading);
    }
    
    // Create keywords container if not already present
    let keywordsList = document.getElementById('keywordsList');
    if (!keywordsList) {
        keywordsList = document.createElement('div');
        keywordsList.id = 'keywordsList';
        keywordsList.className = 'keywords-container';
        keywordsContainer.appendChild(keywordsList);
    } else {
        keywordsList.innerHTML = '';
    }
    
    // Add keywords with animation
    keywords.forEach((keyword, index) => {
        const keywordElement = document.createElement('span');
        keywordElement.className = 'keyword keyword-animate';
        keywordElement.textContent = keyword;
        keywordElement.style.animationDelay = `${index * 0.1}s`;
        keywordsList.appendChild(keywordElement);
    });
    
    // Show the keywords container with a highlight effect
    keywordsContainer.style.display = 'block';
    keywordsContainer.classList.add('highlight-container');
    setTimeout(() => {
        keywordsContainer.classList.remove('highlight-container');
    }, 1000);
}

// Function to save user interaction to CSV for journey tracking
function saveUserInteractionToCSV(userInput, keywords) {
    const timestamp = new Date().toISOString();
    const data = {
        timestamp: timestamp,
        userInput: userInput,
        keywords: keywords.join(', ')
    };
    
    // Send data to server to save in CSV
    fetch('/api/save_interaction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("User interaction saved to CSV");
        } else {
            console.error("Failed to save user interaction:", data.error);
        }
    })
    .catch(error => {
        console.error('Error saving user interaction:', error);
    });
}

// Add a button to open the chat voice box
function addChatVoiceBoxButton() {
    if (!document.getElementById('open-chat-voice')) {
        const button = document.createElement('button');
        button.id = 'open-chat-voice';
        button.className = 'open-chat-voice-btn';
        button.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
            <span>Ask me anything</span>
        `;
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            .open-chat-voice-btn {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background-color: #4F46E5;
                color: white;
                border: none;
                border-radius: 30px;
                padding: 12px 20px;
                display: flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
                z-index: 999;
                transition: all 0.3s ease;
            }
            
            .open-chat-voice-btn:hover {
                background-color: #4338CA;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(button);
        
        // Add event listener
        button.addEventListener('click', toggleChatVoiceBox);
    }
}

// Initialize chat voice box when the page is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chat voice box
    initChatVoiceBox();
    
    // Add button to open chat voice box
    addChatVoiceBoxButton();
    
    // Add event listener for Enter key in chat input
    const chatInput = document.getElementById('chat-voice-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendChatVoiceResponse();
            }
        });
    }
});
