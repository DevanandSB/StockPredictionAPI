document.addEventListener('DOMContentLoaded', function() {
    const companySelect = document.getElementById('company-select');
    const predictBtn = document.getElementById('predict-btn');
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    const errorAlert = document.getElementById('error-alert');
    const successAlert = document.getElementById('success-alert');
    const results = document.getElementById('results');
    const companyData = document.getElementById('company-data');

    let currentCompanyData = null;

    // Company selection handler
    companySelect.addEventListener('change', async function() {
        const symbol = this.value;

        if (!symbol) {
            predictBtn.disabled = true;
            companyData.classList.add('d-none');
            return;
        }

        // Show loading
        loading.classList.remove('d-none');
        loadingText.textContent = `Fetching data for ${symbol}...`;
        errorAlert.classList.add('d-none');
        companyData.classList.add('d-none');
        results.classList.add('d-none');

        try {
            // Fetch company data
            const response = await fetch(`/api/company/${symbol}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || 'Failed to fetch company data');
            }

            const data = await response.json();
            currentCompanyData = data;

            // Display the data
            displayCompanyData(data);

            // Enable predict button
            predictBtn.disabled = false;
            companyData.classList.remove('d-none');

        } catch (error) {
            errorAlert.classList.remove('d-none');
            errorAlert.textContent = 'Error: ' + error.message;
            console.error('Data fetch error:', error);
        } finally {
            loading.classList.add('d-none');
        }
    });

    // Predict button handler
    predictBtn.addEventListener('click', async function() {
        if (!currentCompanyData) return;

        // Show loading
        loading.classList.remove('d-none');
        loadingText.textContent = 'Making prediction...';
        errorAlert.classList.add('d-none');
        results.classList.add('d-none');

        try {
            // Make prediction - use POST method instead of GET
            // In your predict button handler
const response = await fetch('/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        fundamental_data: currentCompanyData.fundamentals,
        technical_data: currentCompanyData.technicals,
        sentiment_data: currentCompanyData.sentiment,
        prediction_type: "combined"
    })
});

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Prediction failed' }));
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const result = await response.json();

            // Display results
            displayResults(result);

            // Show success message
            successAlert.classList.remove('d-none');
            successAlert.textContent = `Prediction completed for ${currentCompanyData.company_name || companySelect.value}!`;

        } catch (error) {
            errorAlert.classList.remove('d-none');
            errorAlert.textContent = 'Error: ' + error.message;
            console.error('Prediction error:', error);
        } finally {
            loading.classList.add('d-none');
        }
    });

    // Display company data
    function displayCompanyData(data) {
        // Display fundamentals
        const fundamentalDiv = document.querySelector('.fundamental-data');
        fundamentalDiv.innerHTML = Object.entries(data.fundamentals || {})
            .map(([key, value]) => `
                <div class="mb-2">
                    <strong>${formatKey(key)}:</strong> 
                    <span class="float-end">${typeof value === 'number' ? value.toFixed(2) : value}</span>
                </div>
            `).join('');

        // Display technicals
        const technicalDiv = document.querySelector('.technical-data');
        technicalDiv.innerHTML = Object.entries(data.technicals || {})
            .map(([key, value]) => `
                <div class="mb-2">
                    <strong>${formatKey(key)}:</strong> 
                    <span class="float-end">${typeof value === 'number' ? value.toFixed(2) : value}</span>
                </div>
            `).join('');

        // Display sentiment
        const sentimentDiv = document.querySelector('.sentiment-data');
        sentimentDiv.innerHTML = Object.entries(data.sentiment || {})
            .filter(([key]) => key !== 'articles') // Exclude articles from main sentiment display
            .map(([key, value]) => {
                let displayValue = typeof value === 'number' ? value.toFixed(2) : value;
                let badgeClass = 'badge bg-secondary';

                if (key.includes('sentiment') || key.includes('rating')) {
                    if (value >= 0.7) badgeClass = 'badge bg-success';
                    else if (value >= 0.4) badgeClass = 'badge bg-warning';
                    else badgeClass = 'badge bg-danger';
                }

                return `
                    <div class="mb-2">
                        <strong>${formatKey(key)}:</strong> 
                        <span class="float-end ${badgeClass}">${displayValue}</span>
                    </div>
                `;
            }).join('');

        // Display news articles in a separate section
        displayNewsArticles(data.sentiment?.articles || []);
    }

    // Display news articles
    // In your displayCompanyData function, add:

// Add this function to display news properly
function displayNewsArticles(articles) {
    const newsContainer = document.getElementById('news-container');
    if (!newsContainer) return;

    if (!articles || articles.length === 0) {
        newsContainer.innerHTML = '<div class="text-center text-muted">No news articles available</div>';
        return;
    }

    newsContainer.innerHTML = articles.map(article => `
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">${article.title || 'No title'}</h6>
                <p class="card-text small text-muted">${article.preview || 'No description available'}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <small class="text-muted">${article.source || 'Unknown source'} • ${formatDate(article.published_at)}</small>
                    <span class="badge ${getSentimentBadgeClass(article.sentiment)}">
                        ${(article.sentiment * 100).toFixed(0)}% sentiment
                    </span>
                </div>
                ${article.url ? `<a href="${article.url}" target="_blank" class="btn btn-sm btn-outline-primary mt-2">Read more</a>` : ''}
            </div>
        </div>
    `).join('');
}

// Helper functions
function formatDate(dateString) {
    if (!dateString) return 'Unknown date';
    try {
        return new Date(dateString).toLocaleDateString();
    } catch {
        return dateString;
    }
}

function getSentimentBadgeClass(sentiment) {
    if (sentiment >= 0.7) return 'bg-success';
    if (sentiment >= 0.6) return 'bg-info';
    if (sentiment >= 0.4) return 'bg-warning';
    return 'bg-danger';
}
    // Display prediction results
    function displayResults(result) {
        document.getElementById('prediction-value').textContent = result.prediction > 0
            ? `+${result.prediction.toFixed(2)}%`
            : `${result.prediction.toFixed(2)}%`;

        document.getElementById('confidence-value').textContent = (result.confidence * 100).toFixed(2) + '%';
        document.getElementById('model-type').textContent = result.model_type || 'unknown';

        // Add interpretation
        const interpretationText = document.getElementById('interpretation-text');
        if (result.prediction > 5) {
            interpretationText.textContent = 'Strong buy signal! The model predicts significant positive performance.';
        } else if (result.prediction > 0) {
            interpretationText.textContent = 'Buy signal. The model predicts positive performance with reasonable confidence.';
        } else if (result.prediction > -5) {
            interpretationText.textContent = 'Neutral to slightly negative. The model suggests caution.';
        } else {
            interpretationText.textContent = 'Strong sell signal. The model predicts significant negative performance.';
        }

        // Show results
        results.classList.remove('d-none');
    }

    // Helper function to format keys
    function formatKey(key) {
        return key.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Helper function to format source
    function formatSource(source) {
        if (typeof source === 'string') return source;
        if (source && source.name) return source.name;
        return 'Unknown Source';
    }

    // Helper function to format date
    function formatDate(dateString) {
        if (!dateString) return 'Unknown date';
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString();
        } catch {
            return dateString;
        }
    }

    // Helper function to get sentiment badge class
    function getSentimentBadgeClass(sentiment) {
        if (sentiment >= 0.7) return 'bg-success';
        if (sentiment >= 0.6) return 'bg-info';
        if (sentiment >= 0.4) return 'bg-warning';
        return 'bg-danger';
    }
});