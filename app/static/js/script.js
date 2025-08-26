// app/static/js/script.js

document.addEventListener('DOMContentLoaded', function() {
    const companySelect = document.getElementById('company-select');
    const predictBtn = document.getElementById('predict-btn');
    const loading = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    const errorAlert = document.getElementById('error-alert');
    const successAlert = document.getElementById('success-alert');
    const resultsContainer = document.getElementById('results');
    const companyData = document.getElementById('company-data');

    let currentCompanyData = null;

    // --- Event Listeners ---
    companySelect.addEventListener('change', handleCompanySelect);
    predictBtn.addEventListener('click', handlePrediction);

    // --- Handlers ---
    async function handleCompanySelect() {
        const symbol = this.value;
        resetUIState();
        if (!symbol) {
            predictBtn.disabled = true;
            return;
        }
        setLoadingState(true, `Fetching data for ${symbol}...`);
        try {
            const response = await fetch(`/api/company/${symbol}`);
            if (!response.ok) throw new Error((await response.json()).detail || 'Failed to fetch company data');
            const data = await response.json();
            currentCompanyData = data;
            displayCompanyData(data);
            companyData.classList.remove('d-none');
            predictBtn.disabled = false;
        } catch (error) {
            showError('Data fetch error: ' + error.message);
        } finally {
            setLoadingState(false);
        }
    }

    async function handlePrediction() {
        if (!currentCompanyData) return;
        resetUIState(true);
        setLoadingState(true, 'Analyzing market data and making predictions...');
        try {
            const response = await fetch(`/api/predict-horizons/${companySelect.value}`, { method: 'POST' });
            if (!response.ok) throw new Error((await response.json()).detail || 'Prediction failed');
            const result = await response.json();
            displayHorizonResults(result);
            showSuccess(`Predictions generated for ${currentCompanyData.company_name || companySelect.value}!`);
        } catch (error) {
            showError('Prediction error: ' + error.message);
        } finally {
            setLoadingState(false);
        }
    }

    // --- UI Update Functions ---
    function displayCompanyData(data) {
        document.querySelector('.fundamental-data').innerHTML = renderKeyValuePairs(data.fundamentals);
        document.querySelector('.technical-data').innerHTML = renderKeyValuePairs(data.technicals);
        document.querySelector('.sentiment-data').innerHTML = renderKeyValuePairs(data.sentiment, true);
        displayNewsArticles(data.sentiment?.articles || []);
    }

    function renderKeyValuePairs(data, isSentiment = false) {
        if (!data || Object.keys(data).length === 0) return '<p class="text-muted small text-center">Data not available.</p>';
        return Object.entries(data)
            .filter(([key]) => key !== 'articles')
            .map(([key, value]) => {
                const keyLower = key.toLowerCase();
                let displayValue = (keyLower.includes('cap') || keyLower.includes('volume'))
                    ? formatLargeNumber(value)
                    : formatValue(value);

                if (isSentiment && (key.includes('sentiment') || key.includes('rating'))) {
                    displayValue = `<span class="badge ${getSentimentBadgeClass(value)}">${displayValue}</span>`;
                }
                return `<div class="mb-2"><strong>${formatKey(key)}:</strong><span class="float-end">${displayValue}</span></div>`;
            }).join('');
    }

    /**
     * THIS IS THE CORRECTED FUNCTION FOR DISPLAYING NEWS
     */
    function displayNewsArticles(articles) {
        const sentimentCardBody = document.querySelector('.sentiment-data')?.parentElement;
        if (!sentimentCardBody) return;

        let newsSection = sentimentCardBody.querySelector('#news-section');
        if (newsSection) newsSection.remove();

        newsSection = document.createElement('div');
        newsSection.id = 'news-section';

        const newsContainer = document.createElement('div');
        newsContainer.id = 'news-container';

        if (!articles || articles.length === 0) {
            newsSection.innerHTML = `<hr><p class="text-center text-muted small mt-3">No recent news articles found.</p>`;
        } else {
            newsSection.innerHTML = `<hr><h6 class="mb-3 mt-3">Relevant News</h6>`;
            // This .map() logic is now correctly implemented
            newsContainer.innerHTML = articles.map(article => `
                <div class="card mb-2">
                    <div class="card-body p-2">
                        <a href="${article.url}" target="_blank" rel="noopener noreferrer" class="text-dark text-decoration-none">
                            <h6 class="card-title" style="font-size: 0.85rem; margin-bottom: 0.25rem;">${article.title || 'No title'}</h6>
                        </a>
                        <div class="d-flex justify-content-between align-items-center mt-1">
                            <small class="text-muted">${article.source || 'Unknown source'}</small>
                            <span class="badge ${getSentimentBadgeClass(article.sentiment)}">
                                ${article.sentiment ? (article.sentiment).toFixed(2) : 'N/A'}
                            </span>
                        </div>
                    </div>
                </div>
            `).join('');
            newsSection.appendChild(newsContainer);
        }
        sentimentCardBody.appendChild(newsSection);
    }

    // --- Helper and State Management Functions ---
    function formatKey(key) { return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '); }

    function formatValue(value) {
        if (value === null || typeof value === 'undefined') return 'N/A';
        if (typeof value !== 'number') return value;
        return value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function formatLargeNumber(value) {
        if (value === null || typeof value === 'undefined') return 'N/A';
        if (typeof value !== 'number') return value;
        const crores = value / 10000000;
        if (crores >= 1) {
            return `₹ ${crores.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} Cr`;
        }
        return `₹ ${value.toLocaleString('en-IN')}`;
    }

    function getSentimentBadgeClass(sentiment) {
        if (sentiment == null) return 'bg-secondary';
        if (sentiment >= 0.65) return 'bg-success';
        if (sentiment >= 0.45) return 'bg-warning';
        return 'bg-danger';
    }

    // app/static/js/script.js

function displayHorizonResults(result) {
    const { predictions } = result;
    // This creates the full HTML table for the results.
    let tableHtml = `
        <h4 class="mb-3">Prediction Horizons</h4>
        <div class="table-responsive">
            <table class="table table-hover align-middle">
                <thead class="table-light">
                    <tr>
                        <th>Timeframe</th>
                        <th class="text-center">Prediction</th>
                        <th class="text-center">Confidence</th>
                        <th>Primary Basis</th>
                    </tr>
                </thead>
                <tbody>
    `;

    // Loop through each prediction (next_day, next_month, etc.)
    for (const [key, value] of Object.entries(predictions)) {
        const pred = value.prediction_percent;
        const predClass = pred > 1 ? 'text-success' : pred < -1 ? 'text-danger' : 'text-secondary';
        const predIcon = pred > 1 ? '▲' : pred < -1 ? '▼' : '▬';

        tableHtml += `
            <tr>
                <td><strong>${formatKey(key)}</strong></td>
                <td class="${predClass} text-center">
                    <h5 class="mb-0">${predIcon} ${pred}%</h5>
                </td>
                <td class="text-center">${(value.confidence * 100).toFixed(0)}%</td>
                <td><span class="badge bg-info text-dark">${value.basis}</span></td>
            </tr>
        `;
    }

    tableHtml += `</tbody></table></div>`;

    // Display the generated table in the results container
    resultsContainer.innerHTML = tableHtml;
    resultsContainer.classList.remove('d-none');
}

    function setLoadingState(isLoading, text = '') {
        loading.classList.toggle('d-none', !isLoading);
        loadingText.textContent = text;
    }

    function resetUIState(isPrediction = false) {
        errorAlert.classList.add('d-none');
        successAlert.classList.add('d-none');
        if (!isPrediction) {
            companyData.classList.add('d-none');
        }
        resultsContainer.classList.add('d-none');
    }

    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
        console.error(message);
    }

    function showSuccess(message) {
        successAlert.textContent = message;
        successAlert.classList.remove('d-none');
    }
});