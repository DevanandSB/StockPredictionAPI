document.addEventListener('DOMContentLoaded', function () {
    // --- Element References ---
    const exploreBtn = document.getElementById('explore-btn');
    const appSection = document.getElementById('app-section');
    const companySearch = document.getElementById('company-search');
    const searchResults = document.getElementById('search-results');
    const analyzeBtn = document.getElementById('analyze-btn');
    const companyDataContainer = document.getElementById('company-data');
    const resultsContainer = document.getElementById('results');
    const errorAlert = document.getElementById('error-alert');
    const chartContainer = document.getElementById('chart-container');
    const predictionCardsContainer = document.getElementById('prediction-cards-container');
    const loadingIndicator = document.getElementById('loading-indicator');

    // Modal References
    const progressModal = new bootstrap.Modal(document.getElementById('progressModal'));
    const progressStatusText = document.getElementById('progress-status-text');
    const progressBar = document.getElementById('progress-bar');

    let selectedSymbol = null;
    let companies = [];
    let eventSource = null;
    let chartInstance = null;

    // --- Smooth Scroll for Hero Button ---
    if (exploreBtn && appSection) {
        exploreBtn.addEventListener('click', (e) => {
            e.preventDefault();
            appSection.scrollIntoView({ behavior: 'smooth' });
        });
    }

    /**
     * Generates a random number from a standard normal distribution
     * using the Box-Muller transform.
     */
    function gaussianRandom() {
        let u = 0, v = 0;
        while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    // --- Dynamic Company Loading ---
    async function loadCompanies() {
        try {
            const response = await fetch('/api/companies');
            if (!response.ok) throw new Error('Failed to load company list.');
            companies = await response.json();
            console.log('Loaded companies:', companies.length);
        } catch (error) {
            console.error('Error loading companies:', error);
            showError("Could not load company list.");
        }
    }

    // --- Event Listeners ---
    companySearch.addEventListener('input', handleSearchInput);
    companySearch.addEventListener('focus', handleSearchInput);
    document.addEventListener('click', (e) => {
        if (!companySearch.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.remove('active');
        }
    });

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            if (selectedSymbol) {
                handleAnalysis(selectedSymbol);
            } else {
                showError("Please select a company first.");
            }
        });
    }

    // --- Handlers ---
    function handleSearchInput() {
        const query = companySearch.value.toLowerCase();
        searchResults.innerHTML = '';

        if (query.length < 2) {
            searchResults.classList.remove('active');
            return;
        }

        const filteredCompanies = companies.filter(company =>
            company.name.toLowerCase().includes(query) ||
            company.symbol.toLowerCase().includes(query)
        );

        if (filteredCompanies.length > 0) {
            filteredCompanies.forEach(company => {
                const item = document.createElement('div');
                item.className = 'search-result-item';
                item.innerHTML = `
                    <strong>${company.name}</strong>
                    <small class="text-muted">${company.symbol}</small>
                `;
                item.onclick = () => {
                    selectCompany(company);
                };
                searchResults.appendChild(item);
            });
            searchResults.classList.add('active');
        } else {
            searchResults.classList.remove('active');
        }
    }

    function selectCompany(company) {
        console.log('Company selected:', company);
        companySearch.value = company.name;
        selectedSymbol = company.symbol;
        searchResults.classList.remove('active');

        // Enable Analyze button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }

        // AUTO-FETCH DATA IMMEDIATELY
        handleCompanySelect(selectedSymbol);
    }

    async function handleCompanySelect(symbol) {
        console.log('Fetching data for:', symbol);
        resetUI(true);
        showLoading();
        updateProgress(0, `Fetching data for ${symbol}...`);
        progressModal.show();

        try {
            const response = await fetch(`/api/company/${symbol}`);
            console.log('API response status:', response.status);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Failed to fetch data (Status: ${response.status})`);
            }

            const data = await response.json();
            console.log('API data received:', data);

            if (data.error) {
                throw new Error(data.error);
            }

            displayCompanyData(data);
            companyDataContainer.classList.remove('d-none');
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
            }

        } catch (error) {
            console.error('Error in handleCompanySelect:', error);
            showError(error.message || 'Failed to fetch company data');
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
            }
        } finally {
            hideLoading();
            progressModal.hide();
        }
    }

    function handleAnalysis(symbol) {
        if (!symbol) {
            showError("Please select a company first.");
            return;
        }

        console.log('Starting analysis for:', symbol);
        resultsContainer.classList.add('d-none');
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
        }
        updateProgress(0, 'Initializing Analysis...');
        progressModal.show();

        // Close any existing connection
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        // Create new EventSource connection
        eventSource = new EventSource(`/api/predict-stream/${symbol}`);
        console.log('EventSource created for:', symbol);

        eventSource.onmessage = function(event) {
            console.log('Received SSE message:', event.data);

            try {
                let jsonData;
                if (event.data.startsWith('data: ')) {
                    jsonData = JSON.parse(event.data.substring(6));
                } else {
                    jsonData = JSON.parse(event.data);
                }

                console.log('Parsed data:', jsonData);

                if (jsonData.error) {
                    console.error('Prediction error:', jsonData.error);
                    showError(jsonData.error);
                    closeEventSource();
                    return;
                }

                if (jsonData.status && jsonData.progress !== undefined) {
                    updateProgress(jsonData.progress, jsonData.status);

                    if (jsonData.status === 'Done' && jsonData.result) {
                        console.log('Prediction complete:', jsonData.result);
                        displayAnalysisResults(jsonData.result);
                        resultsContainer.classList.remove('d-none');
                        closeEventSource();
                    }
                }

            } catch (error) {
                console.error('Error parsing SSE data:', error, 'Raw data:', event.data);
                showError('Error processing analysis data. Please try again.');
                closeEventSource();
            }
        };

        eventSource.onerror = function(error) {
            console.error('EventSource error:', error);
            if (progressBar.style.width !== '100%') {
                showError('Connection lost during analysis. Please try again.');
            }
            closeEventSource();
        };

        setTimeout(() => {
            if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
                console.log('Analysis timeout reached');
                showError('Analysis timed out. Please try again.');
                closeEventSource();
            }
        }, 180000);

        function closeEventSource() {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            progressModal.hide();
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
            }
        }
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
            const sourceName = a.source || 'Unknown';
            return `
                <a href="${a.url}" target="_blank" class="news-card">
                    <div class="news-sentiment-icon"><i class="fas ${icon}"></i></div>
                    <div>
                        <div class="news-title">${a.title || 'No title'}</div>
                        <div class="news-meta">
                            <strong>${sourceName.toUpperCase()}</strong> - ${new Date(a.published_at).toLocaleDateString()}
                        </div>
                    </div>
                </a>`;
        }).join('');
    }

    function displayAnalysisResults(prediction) {
        // --- 1. Render Prediction Cards ---
        const formatPercent = (p) => {
            const sign = p > 0 ? '+' : '';
            const colorClass = p > 0 ? 'positive' : p < 0 ? 'negative' : 'neutral';
            return `<strong class="${colorClass}">${sign}${p}%</strong>`;
        };

        let cardsHtml = `
            <div class="card">
                <div class="card-header"><h5 class="mb-0"><i class="fas fa-bullseye me-2"></i>AI Prediction Summary</h5></div>
                <div class="card-body">
                    <p class="text-muted text-center mb-4">
                        ${prediction.basis} &mdash; Current Price: <strong>₹${prediction.current_price}</strong>
                    </p>
                    <div class="row">
        `;

        const orderedHorizons = [
            "Next Day", "Next Week", "Next 15 Days", "Next 30 Days",
            "Next 3 Months", "Next 6 Months", "Next Year", "Next 2 Years"
        ];

        for (const horizon of orderedHorizons) {
            if (prediction.predictions && prediction.predictions[horizon]) {
                const pred = prediction.predictions[horizon];
                cardsHtml += `
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

        cardsHtml += '</div></div></div>';
        predictionCardsContainer.innerHTML = cardsHtml;

        // --- 2. Create Professional TradingView-like Chart ---
        chartContainer.innerHTML = `
            <div class="card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Price Prediction Chart</h5>
                    <div class="chart-controls">
                        <button id="reset-chart" class="btn btn-sm btn-outline-secondary">
                            <i class="fas fa-sync-alt me-1"></i>Default View
                        </button>
                        <button id="show-all-chart" class="btn btn-sm btn-outline-secondary">
                            <i class="fas fa-expand-alt me-1"></i>Show All
                        </button>
                        <button id="fullscreen-chart" class="btn btn-sm btn-outline-secondary">
                            <i class="fas fa-expand me-1"></i>Fullscreen
                        </button>
                    </div>
                </div>
                <div class="card-body p-0 position-relative">
                    <div id="price-chart" style="height: 500px; width: 100%;"></div>
                    <div id="chart-loading" class="chart-loading-overlay">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading chart...</span>
                        </div>
                    </div>
                </div>
                <div class="card-footer text-muted">
                    <small>Scroll to zoom • Drag to pan • Default view shows 90 days history + 30 days prediction</small>
                </div>
            </div>
        `;

        // Create the professional chart
        createProfessionalChart(prediction);
    }

    function createProfessionalChart(prediction) {
        const chartElement = document.getElementById('price-chart');
        const loadingOverlay = document.getElementById('chart-loading');

        loadingOverlay.style.display = 'flex';

        if (chartInstance) {
            chartInstance.remove();
            chartInstance = null;
        }

        setTimeout(() => {
            try {
                // *** MODIFIED: Changed chart options for dark theme ***
                chartInstance = LightweightCharts.createChart(chartElement, {
                    layout: {
                        background: { type: 'solid', color: '#121212' }, // Dark background
                        textColor: '#cccccc', // Light gray text
                    },
                    grid: {
                        vertLines: { color: 'rgba(255, 255, 255, 0.1)' }, // Lighter grid lines
                        horzLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    },
                    timeScale: { timeVisible: true, secondsVisible: false, borderColor: 'rgba(197, 203, 206, 0.5)' },
                    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
                    localization: { priceFormatter: price => `₹${price.toFixed(2)}` }
                });

                // --- DATA PREPARATION ---
                const historicalData = prepareHistoricalData(prediction.historical_data);

                const historicalSeries = chartInstance.addLineSeries({
                    color: 'rgba(0, 123, 255, 1)', // Blue for historical
                    lineWidth: 2,
                    title: 'Historical Price',
                });
                historicalSeries.setData(historicalData);

                // --- PREDICTION LOGIC ---
                if (historicalData.length > 0) {
                    const lastHistoricalPoint = historicalData[historicalData.length - 1];
                    const predictionPoints = getPredictionPoints(prediction);

                    const predictionCurve = preparePredictionData(prediction, historicalData);

                    // Add the prediction series
                    const predictionSeries = chartInstance.addLineSeries({
                        color: 'rgba(255, 138, 0, 1)', // Orange for prediction
                        lineWidth: 2,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        title: 'Predicted Path',
                    });
                    predictionSeries.setData(predictionCurve);

                    // Add markers for clarity
                    const markerData = predictionPoints.map(p => ({
                        time: p.time,
                        position: 'aboveBar',
                        color: '#e91e63',
                        shape: 'circle',
                        text: `${p.horizon}\n₹${p.value.toFixed(2)}`,
                        size: 1,
                    }));
                    markerData.unshift({
                        time: lastHistoricalPoint.time,
                        position: 'aboveBar',
                        color: '#2962FF',
                        shape: 'arrowUp',
                        text: `Today\n₹${lastHistoricalPoint.value.toFixed(2)}`,
                    });
                    predictionSeries.setMarkers(markerData);
                }

                // Set default view
                const today = new Date();
                const ninetyDaysAgo = new Date(today);
                ninetyDaysAgo.setDate(today.getDate() - 90);
                const thirtyDaysFuture = new Date(today);
                thirtyDaysFuture.setDate(today.getDate() + 30);

                chartInstance.timeScale().setVisibleRange({
                    from: Math.floor(ninetyDaysAgo.getTime() / 1000),
                    to: Math.floor(thirtyDaysFuture.getTime() / 1000),
                });

                // --- EVENT LISTENERS ---
                document.getElementById('reset-chart').addEventListener('click', () => chartInstance.timeScale().fitContent());
                document.getElementById('show-all-chart').addEventListener('click', () => chartInstance.timeScale().fitContent());
                document.getElementById('fullscreen-chart').addEventListener('click', () => chartElement.requestFullscreen());

                loadingOverlay.style.display = 'none';

            } catch (error) {
                console.error('Error creating professional chart:', error);
                loadingOverlay.style.display = 'none';
                chartElement.innerHTML = `<div class="alert alert-danger m-3">Chart error: ${error.message}</div>`;
            }
        }, 100);
    }

    function prepareHistoricalData(historical_data) {
        if (!historical_data || historical_data.length === 0) return [];
        const sortedData = historical_data.sort((a, b) => new Date(a.time) - new Date(b.time));
        let sampleRate = 1;
        if (sortedData.length > 5000) sampleRate = 10;
        else if (sortedData.length > 2000) sampleRate = 5;
        else if (sortedData.length > 1000) sampleRate = 2;
        const downsampledData = [];
        for (let i = 0; i < sortedData.length; i += sampleRate) {
            downsampledData.push({
                time: Math.floor(new Date(sortedData[i].time).getTime() / 1000),
                value: parseFloat(sortedData[i].value)
            });
        }
        return downsampledData;
    }

    function preparePredictionData(prediction, historicalData) {
        const data = [];
        let startDate, startPrice;

        if (historicalData && historicalData.length > 0) {
            const lastHistoricalPoint = historicalData[historicalData.length - 1];
            startDate = new Date(lastHistoricalPoint.time * 1000);
            startPrice = lastHistoricalPoint.value;
            data.push({ time: lastHistoricalPoint.time, value: startPrice });
        } else {
            startDate = new Date();
            startPrice = prediction.current_price;
            if (startDate.getDay() === 0 || startDate.getDay() === 6) {
                startDate.setDate(startDate.getDate() + (startDate.getDay() === 0 ? 1 : 2));
            }
            data.push({ time: Math.floor(startDate.getTime() / 1000), value: startPrice });
        }

        if (!prediction.predictions) return data;
        const predictionPoints = getPredictionPoints(prediction);
        if (predictionPoints.length === 0) return data;
        const realisticPredictionData = createRealisticPredictionCurve(
            predictionPoints, startDate, startPrice, prediction.annual_volatility
        );
        return data.concat(realisticPredictionData);
    }

    function getPredictionPoints(prediction) {
        const points = [];
        const today = new Date();
        const horizons = {
            "Next Day": 1, "Next Week": 5, "Next 15 Days": 15, "Next 30 Days": 21,
            "Next 3 Months": 63, "Next 6 Months": 126, "Next Year": 252, "Next 2 Years": 504
        };
        for (const [horizon, tradingDays] of Object.entries(horizons)) {
            if (prediction.predictions[horizon]) {
                const pred = prediction.predictions[horizon];
                const futureDate = getFutureTradingDate(today, tradingDays);
                points.push({
                    time: Math.floor(futureDate.getTime() / 1000),
                    value: pred.expected_price,
                    horizon: horizon,
                    tradingDays: tradingDays
                });
            }
        }
        return points.sort((a, b) => a.tradingDays - b.tradingDays);
    }

    function getFutureTradingDate(startDate, tradingDays) {
        let currentDate = new Date(startDate);
        let daysCount = 0;
        while (daysCount < tradingDays) {
            currentDate.setDate(currentDate.getDate() + 1);
            if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) {
                daysCount++;
            }
        }
        return currentDate;
    }

    function createRealisticPredictionCurve(predictionPoints, startDate, startPrice, annualVolatility) {
        const curve = [];
        if (predictionPoints.length === 0) return curve;

        const targetPoint = predictionPoints[predictionPoints.length - 1];
        const simulationDays = targetPoint.tradingDays;
        const targetPrice = targetPoint.value;

        if (simulationDays <= 0) return curve;

        const effectiveAnnualVolatility = (annualVolatility && annualVolatility > 0) ? annualVolatility : 25.0;
        const dailyVolatility = (effectiveAnnualVolatility / 100) / Math.sqrt(252);
        const totalLogReturn = Math.log(targetPrice / startPrice);

        let currentPrice = startPrice;
        let currentDate = new Date(startDate);

        for (let i = 1; i <= simulationDays; i++) {
            currentDate = getFutureTradingDate(currentDate, 1);
            const drift = -0.5 * Math.pow(dailyVolatility, 2);
            const randomShock = dailyVolatility * gaussianRandom();
            let simulatedPrice = currentPrice * Math.exp(drift + randomShock);

            const timeRemaining = simulationDays - i;
            if (timeRemaining > 0) {
                const idealPriceOnTrend = startPrice * Math.exp(totalLogReturn * (i / simulationDays));
                const correctionDrift = Math.log(idealPriceOnTrend / simulatedPrice) * (1 / timeRemaining);
                simulatedPrice = simulatedPrice * Math.exp(correctionDrift);
            }
            currentPrice = simulatedPrice;

            curve.push({
                time: Math.floor(currentDate.getTime() / 1000),
                value: parseFloat(currentPrice.toFixed(2))
            });
        }

        if (curve.length > 0) {
            curve[curve.length - 1].value = parseFloat(targetPrice.toFixed(2));
        }

        return curve;
    }

    function updateProgress(progress, status) {
        const p = Math.min(100, Math.round(progress));
        progressBar.style.width = p + '%';
        progressBar.textContent = p + '%';
        progressBar.setAttribute('aria-valuenow', p);
        progressStatusText.textContent = status;
    }

    function showLoading() {
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }
    }

    function hideLoading() {
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }

    const resetUI = (isSearching = false) => {
        if (!isSearching) companySearch.value = '';
        if (analyzeBtn) analyzeBtn.disabled = true;
        errorAlert.classList.add('d-none');
        companyDataContainer.classList.add('d-none');
        resultsContainer.classList.add('d-none');
        if (chartInstance) {
            chartInstance.remove();
            chartInstance = null;
        }
        chartContainer.innerHTML = '';
        predictionCardsContainer.innerHTML = '';
        hideLoading();
    };

    const showError = (message) => {
        const errorMessageSpan = document.getElementById('error-message');
        if (errorMessageSpan) {
            errorMessageSpan.textContent = `Error: ${message}`;
        }
        errorAlert.classList.remove('d-none');
        progressModal.hide();
        if (analyzeBtn) analyzeBtn.disabled = false;
        hideLoading();
    };

    const formatValue = (val) => val != null ? val.toFixed(2) : 'N/A';

    const formatLargeNumber = (num, currency = true) => {
        if (num == null) return 'N/A';
        const prefix = currency ? '₹' : '';
        if (num >= 1e7) return `${prefix}${(num / 1e7).toFixed(2)} Cr`;
        if (num >= 1e5) return `${prefix}${(num / 1e5).toFixed(2)} L`;
        return `${prefix}${parseInt(num).toLocaleString('en-IN')}`;
    };

    // --- Page Initialization Logic ---
function initParticles() {
    particlesJS('particles-js', {
        particles: {
            number: { value: 100, density: { enable: true, value_area: 800 } },
            color: { value: "#8B5FBF" },
            shape: { type: "circle" },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: "#8B5FBF", opacity: 0.3, width: 1 },
            move: { enable: true, speed: 2, direction: "none", random: true, straight: false, out_mode: "out", bounce: false }
        },
        interactivity: {
            detect_on: "canvas",
            events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" }, resize: true }
        },
        retina_detect: true
    });
}

function initScrollAnimations() {
    window.addEventListener('scroll', () => {
        const header = document.querySelector('header');
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
}

    // --- Initializer ---
    loadCompanies();
    initParticles();
    initScrollAnimations();
});