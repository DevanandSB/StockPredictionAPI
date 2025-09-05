document.addEventListener('DOMContentLoaded', function () {
    // --- Element References ---
    const companySearch = document.getElementById('company-search');
    const searchResults = document.getElementById('search-results');
    const analyzeBtn = document.getElementById('analyze-btn');
    const companyDataContainer = document.getElementById('company-data');
    const resultsContainer = document.getElementById('results');
    const errorAlert = document.getElementById('error-alert');

    // Modal References
    const progressModal = new bootstrap.Modal(document.getElementById('progressModal'));
    const progressStatusText = document.getElementById('progress-status-text');
    const progressBar = document.getElementById('progress-bar');

    let selectedSymbol = null;
    let companies = [];
    let eventSource = null;

    // --- Dynamic Company Loading ---
    async function loadCompanies() {
        try {
            const response = await fetch('/api/companies');
            if (!response.ok) throw new Error('Failed to load company list.');
            companies = await response.json();
        } catch (error) {
            showError("Could not load company list.");
        }
    }

    // --- Event Listeners ---
    companySearch.addEventListener('input', handleSearchInput);
    companySearch.addEventListener('focus', handleSearchInput);
    analyzeBtn.addEventListener('click', () => {
        if (selectedSymbol) {
            handleAnalysis(selectedSymbol);
        }
    });
    document.addEventListener('click', (e) => {
        if (!companySearch.contains(e.target)) {
            searchResults.classList.remove('active');
        }
    });

    // --- Handlers ---
    function handleSearchInput() {
        const query = companySearch.value.toLowerCase();
        searchResults.innerHTML = '';
        const filteredCompanies = companies.filter(company =>
            company.name.toLowerCase().includes(query) ||
            company.symbol.toLowerCase().includes(query)
        );

        if (filteredCompanies.length > 0) {
            filteredCompanies.forEach(company => {
                const item = document.createElement('div');
                item.className = 'search-result-item';
                item.textContent = company.name;
                item.onclick = () => {
                    companySearch.value = company.name;
                    selectedSymbol = company.symbol;
                    searchResults.classList.remove('active');
                    handleCompanySelect(selectedSymbol);
                };
                searchResults.appendChild(item);
            });
            searchResults.classList.add('active');
        } else {
            searchResults.classList.remove('active');
        }
    }

    async function handleCompanySelect(symbol) {
        resetUI(true);
        updateProgress(0, `Fetching data for ${symbol}...`);
        progressModal.show();
        try {
            const response = await fetch(`/api/company/${symbol}`);
            if (!response.ok) throw new Error((await response.json()).detail || 'Failed to fetch company data');
            const data = await response.json();
            displayCompanyData(data);
            companyDataContainer.classList.remove('d-none');
            analyzeBtn.disabled = false;
        } catch (error) {
            showError(error.message);
        } finally {
            progressModal.hide();
        }
    }

    function handleAnalysis(symbol) {
        resultsContainer.classList.add('d-none');
        analyzeBtn.disabled = true;
        updateProgress(0, 'Initializing Analysis...');
        progressModal.show();

        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource(`/api/predict-stream/${symbol}`);

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.error) {
                showError(data.error);
                eventSource.close();
                progressModal.hide();
                analyzeBtn.disabled = false;
                return;
            }

            updateProgress(data.progress, data.status);

            if (data.status === "Done") {
                displayAnalysisResults(data.result);
                resultsContainer.classList.remove('d-none');
                eventSource.close();
                setTimeout(() => progressModal.hide(), 500);
                analyzeBtn.disabled = false;
            }
        };

        eventSource.onerror = function(err) {
            showError("Connection to the server was lost during analysis.");
            eventSource.close();
            progressModal.hide();
            analyzeBtn.disabled = false;
        };
    }

    function displayCompanyData(data) {
        displayKeyMetrics(data.fundamentals, data.technicals);
        displayFundamentals(data.fundamentals);
        displayTechnicals(data.technicals);
        displaySentiment(data.sentiment);
        displayNews(data.sentiment?.articles || []);
    }

    function displayKeyMetrics(fundamentals, technicals) {
        const metrics = [
            { icon: 'fas fa-money-bill-wave', value: `₹${formatValue(technicals.current_price)}`, label: 'Current Price' },
            { icon: 'fas fa-landmark', value: formatLargeNumber(fundamentals.market_cap), label: 'Market Cap' },
            { icon: 'fas fa-chart-pie', value: formatValue(fundamentals.pe_ratio), label: 'P/E Ratio' },
            { icon: 'fas fa-percentage', value: `${formatValue(fundamentals.roe)}%`, label: 'ROE' },
            { icon: 'fas fa-hand-holding-usd', value: formatValue(fundamentals.eps), label: 'EPS' },
            { icon: 'fas fa-balance-scale', value: formatValue(fundamentals.debt_to_equity), label: 'Debt/Equity' }
        ];
        document.getElementById('key-metrics').innerHTML = metrics.map(m => `
            <div class="col-6 col-md-4 col-lg-2 metric-card">
                <div class="metric-icon"><i class="${m.icon}"></i></div>
                <div class="metric-value">${m.value || 'N/A'}</div>
                <div class="metric-label">${m.label}</div>
            </div>`).join('');
    }

    function createDataRow(key, value, status = '') {
        return `<div class="data-row"><span class="key">${key}</span><strong class="value ${status}">${value || 'N/A'}</strong></div>`;
    }

    function displayFundamentals(d) {
        const container = document.getElementById('fundamentals-container');
        container.innerHTML = [
            createDataRow('P/E Ratio', formatValue(d.pe_ratio)),
            createDataRow('Book Value', `₹${formatValue(d.book_value)}`),
            createDataRow('Dividend Yield', `${formatValue(d.dividend_yield)}%`),
            createDataRow('ROE', `${formatValue(d.roe)}%`),
            createDataRow('ROCE', `${formatValue(d.roce)}%`),
            createDataRow('EPS', `₹${formatValue(d.eps)}`),
            createDataRow('Debt to Equity', formatValue(d.debt_to_equity)),
        ].join('');
    }

    function displayTechnicals(d) {
        const container = document.getElementById('technicals-container');
        const rsiStatus = d.rsi > 70 ? 'negative' : d.rsi < 30 ? 'positive' : '';
        const trend = d.current_price > d.sma_200 ? 'positive' : 'negative';
        container.innerHTML = [
            createDataRow('RSI', `${formatValue(d.rsi)} <small>(${(rsiStatus || 'neutral')})</small>`, rsiStatus),
            createDataRow('MACD', formatValue(d.macd)),
            createDataRow('50-Day MA', `₹${formatValue(d.sma_50)}`),
            createDataRow('200-Day MA', `₹${formatValue(d.sma_200)}`, trend),
            createDataRow('52-Week High', `₹${formatValue(d['52_week_high'])}`),
            createDataRow('52-Week Low', `₹${formatValue(d['52_week_low'])}`),
            createDataRow('Volume', formatLargeNumber(d.volume, false))
        ].join('');
    }

    function displaySentiment(s) {
        const container = document.getElementById('sentiment-container');
        if (!s) { container.innerHTML = '<p class="text-muted text-center">Sentiment data not available.</p>'; return; }
        const sentimentItems = [
            { label: 'News Sentiment', value: s.news_sentiment, icon: 'fa-newspaper' },
            { label: 'Social Media Buzz', value: s.social_media_sentiment, icon: 'fa-hashtag' },
            { label: 'Analyst Rating', value: s.analyst_rating / 10, icon: 'fa-user-tie' }
        ];
        container.innerHTML = sentimentItems.map(item => {
            const score = (item.value * 100).toFixed(0);
            const status = score >= 65 ? 'positive' : score >= 40 ? 'neutral' : 'negative';
            const displayValue = item.label === 'Analyst Rating' ? `${(item.value*10).toFixed(1)}/10` : `${score}%`;
            return `
                <div class="sentiment-item">
                    <div class="sentiment-icon ${status}"><i class="fas ${item.icon}"></i></div>
                    <div class="sentiment-details">
                        <div class="sentiment-label">
                            ${item.label}
                            <strong class="float-end ${status}">${displayValue}</strong>
                        </div>
                        <div class="progress">
                            <div class="progress-bar bg-${status}" role="progressbar" style="width: ${score}%"></div>
                        </div>
                    </div>
                </div>`;
        }).join('');
    }

    function displayNews(articles) {
        const container = document.getElementById('news-container');
        if (!articles || articles.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No important news found.</p>'; return;
        }
        container.innerHTML = articles.map(a => {
            const sentiment = a.sentiment || 0.5;
            const icon = sentiment >= 0.6 ? 'fa-arrow-trend-up positive' : sentiment >= 0.4 ? 'fa-minus neutral' : 'fa-arrow-trend-down negative';
            const sourceName = a.source.replace('www.', '').split('.')[0];
            return `
                <a href="${a.url}" target="_blank" class="news-card">
                    <div class="news-sentiment-icon"><i class="fas ${icon}"></i></div>
                    <div>
                        <div class="news-title">${a.title || 'No title'}</div>
                        <div class="news-meta">
                            <strong>${sourceName.toUpperCase()}</strong> - ${new Date(a.publishedAt).toLocaleDateString()}
                        </div>
                    </div>
                </a>`;
        }).join('');
    }

    function displayAnalysisResults(prediction) {
        const formatPercent = (p) => {
            const sign = p > 0 ? '+' : '';
            const colorClass = p > 0 ? 'positive' : p < 0 ? 'negative' : 'neutral';
            return `<strong class="${colorClass}">${sign}${p}%</strong>`;
        };

        let resultHtml = `
            <div class="card">
                <div class="card-header"><h5 class="mb-0"><i class="fas fa-bullseye me-2"></i>AI Prediction</h5></div>
                <div class="card-body">
                    <p class="text-muted text-center mb-4">
                        ${prediction.basis} &mdash; Current Price: <strong>₹${prediction.current_price}</strong>
                    </p>
                    <div class="row">
        `;
        const orderedHorizons = [
            "Next Day", "Next Week", "Next 15 Days", "Next 30 Days",
            "Next 3 Months", "Next 6 Months", "Next Year"
        ];
        for (const horizon of orderedHorizons) {
            if (prediction.predictions[horizon]) {
                const pred = prediction.predictions[horizon];
                resultHtml += `
                    <div class="col-md-6 col-lg-4 mb-3">
                        <div class="prediction-card h-100">
                            <div class="horizon">${horizon}</div>
                            <div class="range">₹${pred.lower_bound} – ₹${pred.upper_bound}</div>
                            <div class="expected-price text-muted">Expected: ₹${pred.expected_price}</div>
                            <div class="percentage-range">
                                ${formatPercent(pred.lower_change_percent)} to ${formatPercent(pred.upper_change_percent)}
                            </div>
                        </div>
                    </div>`;
            }
        }
        resultHtml += '</div></div></div>';
        resultsContainer.innerHTML = resultHtml;
    }

    function updateProgress(progress, status) {
        const p = Math.min(100, Math.round(progress));
        progressBar.style.width = p + '%';
        progressBar.textContent = p + '%';
        progressBar.setAttribute('aria-valuenow', p);
        progressStatusText.textContent = status;
    }

    const resetUI = (isSearching = false) => {
        if (!isSearching) companySearch.value = '';
        analyzeBtn.disabled = true;
        errorAlert.classList.add('d-none');
        companyDataContainer.classList.add('d-none');
        resultsContainer.classList.add('d-none');
    };

    const showError = (message) => {
        errorAlert.textContent = `Error: ${message}`;
        errorAlert.classList.remove('d-none');
    };

    const formatValue = (val) => val != null ? val.toFixed(2) : 'N/A';

    const formatLargeNumber = (num, currency = true) => {
        if (num == null) return 'N/A';
        const prefix = currency ? '₹' : '';
        if (num >= 1e7) return `${prefix}${(num / 1e7).toFixed(2)} Cr`;
        if (num >= 1e5) return `${prefix}${(num / 1e5).toFixed(2)} L`;
        return `${prefix}${parseInt(num).toLocaleString('en-IN')}`;
    };

    // --- Initializer ---
    loadCompanies();
});