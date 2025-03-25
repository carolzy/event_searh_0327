// Recommendations Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Show loading indicator
    const loadingIndicator = document.getElementById('loading-recommendations');
    
    // Hide all "no data" alerts initially
    document.querySelectorAll('[id^="no-"]').forEach(el => {
        el.classList.add('d-none');
    });
    
    // Fetch recommendations from the API
    fetchRecommendations();
    
    /**
     * Fetch recommendations from the API
     */
    async function fetchRecommendations() {
        try {
            const response = await fetch('/api/recommendations');
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const recommendations = await response.json();
            
            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            
            // Process and display recommendations
            displayRecommendations(recommendations);
            
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            
            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            
            // Show error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger';
            errorDiv.textContent = 'Failed to load recommendations. Please try again later.';
            document.querySelector('.card-body').appendChild(errorDiv);
        }
    }
    
    /**
     * Display recommendations in the appropriate tabs
     */
    function displayRecommendations(recommendations) {
        if (!recommendations || recommendations.length === 0) {
            showNoDataMessage('companies');
            return;
        }
        
        // Display company recommendations
        const companiesContent = document.getElementById('companies-content');
        if (companiesContent) {
            companiesContent.innerHTML = '';
            
            recommendations.forEach((company, index) => {
                const companyCard = createCompanyCard(company, index);
                companiesContent.appendChild(companyCard);
            });
        } else {
            showNoDataMessage('companies');
        }
        
        // Show no data messages for other tabs since we don't have that data yet
        showNoDataMessage('articles');
        showNoDataMessage('quotes');
        showNoDataMessage('leads');
        showNoDataMessage('events');
    }
    
    /**
     * Create a company card element
     */
    function createCompanyCard(company, index) {
        const card = document.createElement('div');
        card.className = 'card mb-3';
        
        // Format the match score if available
        const matchScore = company.match_score ? `<span class="badge bg-success">${company.match_score}% Match</span>` : '';
        
        // Create card content
        card.innerHTML = `
            <div class="card-header bg-light d-flex justify-content-between align-items-center">
                <h5 class="mb-0">${index + 1}. ${company.name || 'Unknown Company'}</h5>
                ${matchScore}
            </div>
            <div class="card-body">
                <p class="card-text">${company.description || 'No description available.'}</p>
                ${company.fit_reason ? `<p class="card-text"><strong>Why it's a good fit:</strong> ${company.fit_reason}</p>` : ''}
                ${company.website ? `<a href="${company.website}" target="_blank" class="btn btn-outline-primary btn-sm mt-2">
                    <i class="bi bi-globe"></i> Visit Website
                </a>` : ''}
                ${company.linkedin_url ? `<a href="${company.linkedin_url}" target="_blank" class="btn btn-outline-primary btn-sm mt-2 ms-2">
                    <i class="bi bi-linkedin"></i> LinkedIn
                </a>` : ''}
            </div>
        `;
        
        return card;
    }
    
    /**
     * Show "no data" message for a tab
     */
    function showNoDataMessage(tabId) {
        const noDataElement = document.getElementById(`no-${tabId}`);
        if (noDataElement) {
            noDataElement.classList.remove('d-none');
        }
    }
});
