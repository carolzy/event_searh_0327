// Flow state management
let currentStep = 1;
let flowData = {
    company: '',
    productUnit: '',
    sector: '',
    marketSegment: ''
};

// WebSocket connection for real-time voice interaction
let ws = null;
let isListening = false;

async function initializeFlow() {
    // Hide all steps except first
    document.querySelectorAll('.flow-step').forEach(step => {
        step.style.display = 'none';
    });
    document.getElementById('step1').style.display = 'block';
    
    // Update step indicators
    updateStepIndicators(1);
    
    // Initialize WebSocket connection
    initWebSocket();
}

function updateStepIndicators(currentStep) {
    const dots = document.querySelectorAll('.step-dot');
    dots.forEach((dot, index) => {
        if (index + 1 < currentStep) {
            dot.classList.remove('active');
            dot.classList.add('completed');
        } else if (index + 1 === currentStep) {
            dot.classList.add('active');
            dot.classList.remove('completed');
        } else {
            dot.classList.remove('active', 'completed');
        }
    });
}

async function submitCompany() {
    const company = document.getElementById('companyInput').value;
    if (!company) {
        showError('Please enter your company name');
        return;
    }
    
    showLoading('Analyzing company details...');
    
    try {
        const response = await fetch(`/analyze_company?company=${encodeURIComponent(company)}`, {
            method: 'GET'
        });
        
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Failed to analyze company');
        }
        
        flowData.company = company;
        
        // Populate product units with enhanced details
        const productList = document.getElementById('productList');
        productList.innerHTML = '';
        
        data.data.product_suites.forEach(product => {
            const div = document.createElement('div');
            div.className = 'product-option';
            div.innerHTML = `
                <input type="radio" name="product" value="${product.name}" id="product_${product.name.replace(/\s+/g, '_')}">
                <label for="product_${product.name.replace(/\s+/g, '_')}">
                    <strong>${product.name}</strong>
                    <p>${product.description}</p>
                    <ul>
                        ${product.key_features.map(feature => `<li>${feature}</li>`).join('')}
                    </ul>
                </label>
            `;
            productList.appendChild(div);
        });
        
        // Populate sectors with use cases
        const sectorList = document.getElementById('sectorList');
        sectorList.innerHTML = '';
        
        data.data.target_sectors.forEach(sector => {
            const div = document.createElement('div');
            div.className = 'sector-option';
            div.innerHTML = `
                <input type="radio" name="sector" value="${sector.name}" id="sector_${sector.name.replace(/\s+/g, '_')}">
                <label for="sector_${sector.name.replace(/\s+/g, '_')}">
                    <strong>${sector.name}</strong>
                    <ul>
                        ${sector.use_cases.map(useCase => `<li>${useCase}</li>`).join('')}
                    </ul>
                </label>
            `;
            sectorList.appendChild(div);
        });
        
        // Store investment areas and sales keywords for later use
        flowData.investmentAreas = data.data.investment_areas;
        flowData.salesKeywords = data.data.sales_keywords;
        flowData.differentiators = data.data.differentiators;
        
        // Move to next step
        nextStep();
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

async function submitProductUnit() {
    const selectedProduct = document.querySelector('input[name="product"]:checked');
    if (!selectedProduct) {
        showError('Please select a product unit');
        return;
    }
    
    flowData.productUnit = selectedProduct.value;
    nextStep();
}

async function submitSector() {
    const selectedSector = document.querySelector('input[name="sector"]:checked');
    if (!selectedSector) {
        showError('Please select a target sector');
        return;
    }
    
    flowData.sector = selectedSector.value;
    
    try {
        showLoading('Loading market segments...');
        const response = await fetch('/market_segments');
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Failed to load market segments');
        }
        
        const segmentList = document.getElementById('segmentList');
        segmentList.innerHTML = '';
        
        data.segments.forEach(segment => {
            const div = document.createElement('div');
            div.className = 'segment-option';
            div.innerHTML = `
                <input type="radio" name="segment" value="${segment.id}" id="segment_${segment.id}">
                <label for="segment_${segment.id}">
                    <strong>${segment.name}</strong>
                    <p>${segment.description}</p>
                    <ul>
                        ${segment.characteristics.map(char => `<li>${char}</li>`).join('')}
                    </ul>
                </label>
            `;
            segmentList.appendChild(div);
        });
        
        nextStep();
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

async function submitMarketSegment() {
    const selectedSegment = document.querySelector('input[name="segment"]:checked');
    if (!selectedSegment) {
        showError('Please select a market segment');
        return;
    }
    
    flowData.marketSegment = selectedSegment.value;
    
    // Show the voice interface
    document.querySelectorAll('.flow-step').forEach(step => {
        step.style.display = 'none';
    });
    document.getElementById('voiceInterface').style.display = 'block';
    
    // Initialize voice interface with enhanced context
    startVoiceInterface(flowData);
}

function nextStep() {
    document.getElementById(`step${currentStep}`).style.display = 'none';
    currentStep++;
    document.getElementById(`step${currentStep}`).style.display = 'block';
    updateStepIndicators(currentStep);
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 3000);
}

function showLoading(message) {
    const loadingDiv = document.getElementById('loadingMessage');
    loadingDiv.textContent = message;
    loadingDiv.style.display = 'block';
}

function hideLoading() {
    document.getElementById('loadingMessage').style.display = 'none';
}

// Initialize WebSocket connection
function initWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        // Start daily company flow
        startDailyCompany();
    };
    
    ws.onmessage = async (event) => {
        const response = JSON.parse(event.data);
        console.log('WebSocket message:', response);
        
        if (!response.success) {
            showError(response.error);
            return;
        }
        
        // Handle different response types
        if (response.company) {
            // Daily company response
            displayDailyCompany(response.company);
        } else if (response.transcript) {
            // Voice input response
            handleVoiceResponse(response.transcript);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showError('Connection error. Please refresh the page.');
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed');
        setTimeout(initWebSocket, 1000); // Reconnect after 1 second
    };
}

// Start daily company flow
async function startDailyCompany() {
    try {
        // Get today's company
        const response = await fetch('/daily_company');
        const data = await response.json();
        
        if (!data.success) {
            showError(data.error);
            return;
        }
        
        // Display company info
        displayDailyCompany(data.company);
        
        // Start voice interaction
        ws.send(JSON.stringify({
            action: 'start_daily_company'
        }));
        
    } catch (error) {
        console.error('Error starting daily company:', error);
        showError('Failed to get daily company. Please try again.');
    }
}

// Display daily company information
function displayDailyCompany(company) {
    const dailyCompanySection = document.getElementById('daily-company');
    if (!dailyCompanySection) return;
    
    dailyCompanySection.innerHTML = `
        <div class="company-header">
            <h2>${company.company}</h2>
            <div class="company-meta">
                <span class="industry">${company.industry}</span>
                <span class="date">${company.date}</span>
            </div>
        </div>
        <div class="company-content" id="company-content">
            <!-- Dynamic content will be added here -->
        </div>
        <div class="voice-controls">
            <button id="listen-btn" onclick="toggleListening()">
                <i class="fas fa-microphone"></i>
                Start Listening
            </button>
            <div class="voice-status" id="voice-status"></div>
        </div>
    `;
}

// Toggle voice listening
function toggleListening() {
    const listenBtn = document.getElementById('listen-btn');
    const voiceStatus = document.getElementById('voice-status');
    
    if (!isListening) {
        // Start listening
        isListening = true;
        listenBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Listening';
        voiceStatus.textContent = 'Listening...';
        
        ws.send(JSON.stringify({
            action: 'process_audio'
        }));
    } else {
        // Stop listening
        isListening = false;
        listenBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Listening';
        voiceStatus.textContent = '';
    }
}

// Handle voice response
function handleVoiceResponse(transcript) {
    const voiceStatus = document.getElementById('voice-status');
    voiceStatus.textContent = `You said: ${transcript}`;
}

// Handle voice input submission
function handleVoiceInput(text) {
    if (!text) return;
    
    // Show loading indicator
    document.getElementById('loading-indicator').style.display = 'block';
    
    // Get the current step
    const currentStep = document.getElementById('current-step').value || 'product';
    
    // Send the voice input to the server
    fetch('/api/voice_interaction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            step: currentStep
        })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading indicator
        document.getElementById('loading-indicator').style.display = 'none';
        
        if (data.success) {
            // Display the response text if available
            if (data.text) {
                const responseElement = document.getElementById('current-question');
                if (responseElement) {
                    responseElement.textContent = data.text;
                    responseElement.style.display = 'block';
                }
            }
            
            // Update keywords if available
            if (data.keywords && Array.isArray(data.keywords)) {
                updateKeywordsCloud(data.keywords);
            }
            
            // Update the current step
            if (data.next_step) {
                const stepInput = document.getElementById('current-step');
                if (stepInput) {
                    stepInput.value = data.next_step;
                }
                
                // Update UI based on step
                updateUIForStep(data.next_step);
            }
            
            // Play audio if available
            if (data.audio) {
                const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
                audio.play().catch(e => {
                    console.error("Error playing audio:", e);
                });
            }
            
            // Show recommendations button if requested
            if (data.show_recommendations_button) {
                showRecommendationsButton();
            }
            
            // Clear the input field
            const answerInput = document.getElementById('answer');
            if (answerInput) {
                answerInput.value = '';
                answerInput.focus();
            }
        } else {
            console.error("Error processing voice input:", data.error);
            // Show error message
            const errorElement = document.getElementById('error-message');
            if (errorElement) {
                errorElement.textContent = data.error || "Error processing your input. Please try again.";
                errorElement.style.display = 'block';
                
                // Hide error after 5 seconds
                setTimeout(() => {
                    errorElement.style.display = 'none';
                }, 5000);
            }
        }
    })
    .catch(error => {
        // Hide loading indicator
        document.getElementById('loading-indicator').style.display = 'none';
        console.error("Error submitting voice input:", error);
    });
}

// Update UI based on current step
function updateUIForStep(step) {
    console.log(`Updating UI for step: ${step}`);
    
    // Update hint text if available
    const hintElement = document.getElementById('hint-text');
    if (hintElement) {
        let hintText = '';
        
        switch(step) {
            case 'product':
                hintText = 'Tell us about your product or service';
                break;
            case 'market':
                hintText = 'What market or industry are you targeting?';
                break;
            case 'differentiation':
                hintText = 'What makes your product unique?';
                break;
            case 'company_size':
                hintText = 'What size companies are you targeting?';
                break;
            case 'location':
                hintText = 'Any specific geographic focus?';
                break;
            case 'linkedin_consent':
                hintText = 'Would you like to connect your LinkedIn?';
                break;
            case 'complete':
                hintText = 'Onboarding complete!';
                break;
            default:
                hintText = 'Please provide your answer';
        }
        
        hintElement.textContent = hintText;
    }
    
    // Show/hide elements based on step
    if (step === 'complete') {
        // Hide input area
        const inputArea = document.getElementById('input-area');
        if (inputArea) {
            inputArea.style.display = 'none';
        }
        
        // Show recommendations if available
        const recommendationsArea = document.getElementById('recommendations-area');
        if (recommendationsArea) {
            recommendationsArea.style.display = 'block';
        }
    }
}

// Play text-to-speech
function playTextToSpeech(text) {
    if (!text) return;
    
    // Show loading indicator
    document.getElementById('loading-indicator').style.display = 'block';
    
    fetch('/api/text_to_speech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading indicator
        document.getElementById('loading-indicator').style.display = 'none';
        
        if (data.status === 'quota_exceeded') {
            console.warn("ElevenLabs quota exceeded:", data.message);
            // Show a notification to the user
            showNotification("Voice functionality temporarily unavailable due to quota limits. Text responses will still work.", "warning");
            return;
        }
        
        if (data.audio) {
            const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
            audio.play();
        } else {
            console.error("No audio data returned from server");
        }
    })
    .catch(error => {
        // Hide loading indicator
        document.getElementById('loading-indicator').style.display = 'none';
        console.error("Error playing text-to-speech:", error);
    });
}

// Helper function to show notifications
function showNotification(message, type = "info") {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to the DOM
    document.body.appendChild(notification);
    
    // Show with animation
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateY(0)';
    }, 10);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

function showRecommendationsButton() {
    const recommendationsButton = document.getElementById('recommendations-button');
    if (recommendationsButton) {
        recommendationsButton.style.display = 'block';
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initializeFlow);
