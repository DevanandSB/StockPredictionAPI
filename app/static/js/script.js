document.addEventListener('DOMContentLoaded', function () {
    // --- Element References ---
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

    // Add event listener for Analyze button
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
                    <small>Scroll to zoom • Drag to pan • Default view shows 20 days history + 10 days prediction</small>
                </div>
            </div>
        `;

        // Create the professional chart
        createProfessionalChart(prediction);
    }

    function createProfessionalChart(prediction) {
        const chartElement = document.getElementById('price-chart');
        const loadingOverlay = document.getElementById('chart-loading');

        // Show loading
        loadingOverlay.style.display = 'flex';

        // Clear any existing chart
        if (chartInstance) {
            chartInstance.remove();
            chartInstance = null;
        }

        setTimeout(() => {
            try {
                // Create professional chart instance
                chartInstance = LightweightCharts.createChart(chartElement, {
                    layout: {
                        background: { type: 'solid', color: '#ffffff' },
                        textColor: '#333333',
                        fontSize: 12,
                    },
                    grid: {
                        vertLines: { color: 'rgba(42, 46, 57, 0.1)' },
                        horzLines: { color: 'rgba(42, 46, 57, 0.1)' },
                    },
                    timeScale: {
                        timeVisible: true,
                        secondsVisible: false,
                        borderColor: 'rgba(197, 203, 206, 0.8)',
                        tickMarkFormatter: (time) => {
                            const date = new Date(time * 1000);
                            return date.toLocaleDateString();
                        },
                        barSpacing: 8,
                    },
                    rightPriceScale: {
                        borderColor: 'rgba(197, 203, 206, 0.8)',
                        entireTextOnly: true,
                        autoScale: true,
                    },
                    crosshair: {
                        mode: LightweightCharts.CrosshairMode.Normal,
                        vertLine: {
                            color: '#758696',
                            width: 1,
                            style: LightweightCharts.LineStyle.Dashed,
                            labelBackgroundColor: '#758696',
                        },
                        horzLine: {
                            color: '#758696',
                            width: 1,
                            style: LightweightCharts.LineStyle.Dashed,
                            labelBackgroundColor: '#758696',
                        },
                    },
                    localization: {
                        priceFormatter: (price) => `₹${price.toFixed(2)}`,
                        timeFormatter: (time) => {
                            const date = new Date(time * 1000);
                            return date.toLocaleDateString();
                        },
                    },
                });

                // Prepare data - Use REAL historical data from backend with downsampling
                const historicalData = prepareHistoricalData(prediction.historical_data);
                const predictionData = preparePredictionData(prediction);

                console.log('Historical data points:', historicalData.length);
                console.log('Prediction data points:', predictionData.length);
                console.log('Actual model predictions:', prediction.predictions);

                // Create historical series (Line)
                const historicalSeries = chartInstance.addLineSeries({
                    color: 'rgba(38, 166, 154, 1)',
                    lineWidth: 2,
                    title: 'Historical Price',
                    priceLineVisible: false,
                });

                // Create prediction series (Line) - No area to avoid visual discontinuities
                const predictionSeries = chartInstance.addLineSeries({
                    color: 'rgba(41, 98, 255, 1)',
                    lineWidth: 3,
                    lineStyle: LightweightCharts.LineStyle.Solid,
                    title: 'Predicted Price',
                    priceLineVisible: false,
                });

                // Set data
                historicalSeries.setData(historicalData);

                // Only set prediction data if we have valid predictions
                if (predictionData.length > 0) {
                    predictionSeries.setData(predictionData);
                }

                // Set default view: 20 days historic + 10 days prediction
                const today = new Date();
                const twentyDaysAgo = new Date(today);
                twentyDaysAgo.setDate(today.getDate() - 20);

                const tenDaysFuture = new Date(today);
                tenDaysFuture.setDate(today.getDate() + 10);

                chartInstance.timeScale().setVisibleRange({
                    from: Math.floor(twentyDaysAgo.getTime() / 1000),
                    to: Math.floor(tenDaysFuture.getTime() / 1000)
                });

                // Enable auto-scaling for price axis when user zooms/scrolls
                chartInstance.applyOptions({
                    handleScale: {
                        axisPressedMouseMove: {
                            time: true,
                            price: true,
                        },
                    },
                    handleScroll: {
                        vertTouchDrag: true,
                        horzTouchDrag: true,
                        mouseWheel: true,
                        pressedMouseMove: true,
                    }
                });

                // Add crosshair move handler
                chartInstance.subscribeCrosshairMove((param) => {
                    if (!param.point) return;

                    let tooltipContent = '';
                    if (param.time) {
                        const date = new Date(param.time * 1000);
                        tooltipContent += `<strong>${date.toLocaleDateString()}</strong><br>`;
                    }

                    if (param.seriesPrices) {
                        const prices = param.seriesPrices;

                        // Historical data
                        if (prices.get(historicalSeries)) {
                            const price = prices.get(historicalSeries);
                            tooltipContent += `<div>Historical: ₹${price.toFixed(2)}</div>`;
                        }

                        // Prediction data
                        if (prices.get(predictionSeries)) {
                            const price = prices.get(predictionSeries);
                            tooltipContent += `<div>Predicted: ₹${price.toFixed(2)}</div>`;
                        }
                    }

                    updateChartTooltip(tooltipContent, param.point.x, param.point.y);
                });

                // Hide tooltip when crosshair leaves
                chartInstance.subscribeClick(() => {
                    hideChartTooltip();
                });

                // Add reset button functionality - back to 20 days historic + 10 days prediction
                document.getElementById('reset-chart').addEventListener('click', () => {
                    const today = new Date();
                    const twentyDaysAgo = new Date(today);
                    twentyDaysAgo.setDate(today.getDate() - 20);

                    const tenDaysFuture = new Date(today);
                    tenDaysFuture.setDate(today.getDate() + 10);

                    chartInstance.timeScale().setVisibleRange({
                        from: Math.floor(twentyDaysAgo.getTime() / 1000),
                        to: Math.floor(tenDaysFuture.getTime() / 1000)
                    });
                });

                // Add "Show All" button functionality
                document.getElementById('show-all-chart').addEventListener('click', () => {
                    if (historicalData.length > 0 && predictionData.length > 0) {
                        chartInstance.timeScale().setVisibleRange({
                            from: historicalData[0].time,
                            to: predictionData[predictionData.length - 1].time
                        });
                    }
                });

                // Add fullscreen functionality
                document.getElementById('fullscreen-chart').addEventListener('click', () => {
                    chartElement.requestFullscreen().catch(err => {
                        console.error('Fullscreen error:', err);
                    });
                });

                // Handle fullscreen changes
                document.addEventListener('fullscreenchange', () => {
                    if (document.fullscreenElement) {
                        chartInstance.applyOptions({
                            width: window.innerWidth,
                            height: window.innerHeight
                        });
                    } else {
                        chartInstance.applyOptions({
                            width: chartElement.clientWidth,
                            height: 500
                        });

                        // Reset to default view when exiting fullscreen
                        const today = new Date();
                        const twentyDaysAgo = new Date(today);
                        twentyDaysAgo.setDate(today.getDate() - 20);

                        const tenDaysFuture = new Date(today);
                        tenDaysFuture.setDate(today.getDate() + 10);

                        chartInstance.timeScale().setVisibleRange({
                            from: Math.floor(twentyDaysAgo.getTime() / 1000),
                            to: Math.floor(tenDaysFuture.getTime() / 1000)
                        });
                    }
                });

                // Resize chart when window resizes
                window.addEventListener('resize', () => {
                    if (!document.fullscreenElement) {
                        chartInstance.applyOptions({
                            width: chartElement.clientWidth
                        });
                    }
                });

                // Hide loading
                loadingOverlay.style.display = 'none';

            } catch (error) {
                console.error('Error creating professional chart:', error);
                loadingOverlay.style.display = 'none';
                chartElement.innerHTML = `
                    <div class="alert alert-danger m-3">
                        <i class="fas fa-exclamation-triangle"></i> 
                        Chart error: ${error.message}
                    </div>
                `;
            }
        }, 100);
    }

    function prepareHistoricalData(historical_data) {
        // Process REAL historical data from backend with intelligent downsampling
        if (!historical_data || historical_data.length === 0) {
            console.warn('No historical data provided');
            return [];
        }

        // Sort data by date
        const sortedData = historical_data.sort((a, b) => new Date(a.time) - new Date(b.time));

        // Intelligent downsampling based on data length
        let sampleRate = 1;
        if (sortedData.length > 5000) {
            sampleRate = 10; // For very long histories (20+ years)
        } else if (sortedData.length > 2000) {
            sampleRate = 5; // For long histories (8-20 years)
        } else if (sortedData.length > 1000) {
            sampleRate = 2; // For medium histories (4-8 years)
        }

        const downsampledData = [];

        for (let i = 0; i < sortedData.length; i += sampleRate) {
            downsampledData.push({
                time: Math.floor(new Date(sortedData[i].time).getTime() / 1000),
                value: parseFloat(sortedData[i].value)
            });
        }

        console.log(`Downsampled from ${sortedData.length} to ${downsampledData.length} points (rate: ${sampleRate})`);
        return downsampledData;
    }

    function preparePredictionData(prediction) {
        const data = [];
        const today = new Date();
        const currentPrice = prediction.current_price;

        if (!prediction.predictions) return data;

        // Get the key prediction points from the AI model
        const predictionPoints = getPredictionPoints(prediction);

        if (predictionPoints.length === 0) return data;

        // Start from current price (connect to historical data)
        // Ensure we use a trading day (skip if today is weekend)
        let startChartDate = new Date(today);
        if (startChartDate.getDay() === 0 || startChartDate.getDay() === 6) {
            // If today is weekend, move to next trading day (Monday)
            startChartDate.setDate(startChartDate.getDate() + (startChartDate.getDay() === 0 ? 1 : 2));
        }

        data.push({
            time: Math.floor(startChartDate.getTime() / 1000),
            value: currentPrice
        });

        // Create realistic prediction curve using Brownian motion with drift
        const realisticPredictionData = createRealisticPredictionCurve(predictionPoints, startChartDate, currentPrice, prediction.annual_volatility);

        return data.concat(realisticPredictionData);
    }

    function getPredictionPoints(prediction) {
        const points = [];
        const today = new Date();

        // Define the prediction horizons and their TRADING day offsets (approx)
        const horizons = {
            "Next Day": 1,
            "Next Week": 5,
            "Next 15 Days": 15,
            "Next 30 Days": 21,    // ~30 calendar days = ~21 trading days
            "Next 3 Months": 63,   // 3 months = ~63 trading days
            "Next 6 Months": 126,  // 6 months = ~126 trading days
            "Next Year": 252,      // 1 year = 252 trading days
            "Next 2 Years": 504    // 2 years = 504 trading days
        };

        // Create prediction points from the actual model output
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

            // Skip weekends (Saturday = 6, Sunday = 0)
            if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) {
                daysCount++;
            }
        }

        return currentDate;
    }

    function createRealisticPredictionCurve(predictionPoints, startDate, startPrice, annualVolatility) {
        const curve = [];

        if (predictionPoints.length === 0) return curve;

        // Calculate daily drift from the 2-year prediction
        const twoYearPoint = predictionPoints.find(p => p.horizon === "Next 2 Years");
        const dailyDrift = twoYearPoint ?
            Math.log(twoYearPoint.value / startPrice) / 500 : // Use ~500 trading days for 2 years (252 trading days/year)
            0.0005; // Default small growth

        // Convert annual volatility to daily volatility
        const dailyVolatility = (annualVolatility / 100) / Math.sqrt(252);

        let currentPrice = startPrice;
        let currentDate = new Date(startDate);
        let tradingDays = 0;
        const maxTradingDays = 504; // 2 years * 252 trading days/year

        // Create 2 years of trading days prediction data
        while (tradingDays < maxTradingDays) {
            // Move to next trading day (skip weekends)
            currentDate.setDate(currentDate.getDate() + 1);

            // Skip weekends (Saturday = 6, Sunday = 0)
            if (currentDate.getDay() === 0 || currentDate.getDay() === 6) {
                continue;
            }

            tradingDays++;

            // Brownian motion with drift: dS = μSdt + σSdW
            const drift = dailyDrift * currentPrice;
            const shock = dailyVolatility * currentPrice * gaussianRandom();

            currentPrice = currentPrice + drift + shock;

            // Ensure price stays within reasonable bounds
            const maxPrice = startPrice * 3.0;
            const minPrice = startPrice * 0.3;
            currentPrice = Math.max(minPrice, Math.min(maxPrice, currentPrice));

            curve.push({
                time: Math.floor(currentDate.getTime() / 1000),
                value: currentPrice
            });
        }

        return curve;
    }

    function gaussianRandom() {
        // Box-Muller transform for Gaussian random numbers
        let u = 0, v = 0;
        while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function updateChartTooltip(content, x, y) {
        let tooltip = document.getElementById('chart-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'chart-tooltip';
            tooltip.className = 'chart-tooltip';
            document.body.appendChild(tooltip);
        }

        if (content) {
            tooltip.innerHTML = content;
            tooltip.style.display = 'block';

            const rect = tooltip.getBoundingClientRect();
            const left = Math.min(x + 10, window.innerWidth - rect.width - 10);
            const top = Math.min(y - rect.height / 2, window.innerHeight - rect.height - 10);

            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
        } else {
            tooltip.style.display = 'none';
        }
    }

    function hideChartTooltip() {
        const tooltip = document.getElementById('chart-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
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
        hideChartTooltip();
    };

    const showError = (message) => {
        errorAlert.textContent = `Error: ${message}`;
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

    // Debug function
    function debugSearch() {
        console.log('Companies loaded:', companies.length);
        console.log('Selected symbol:', selectedSymbol);
        console.log('Search value:', companySearch.value);

        if (selectedSymbol) {
            fetch(`/api/company/${selectedSymbol}`)
                .then(response => {
                    console.log('Manual API test - Status:', response.status);
                    return response.json();
                })
                .then(data => console.log('Manual API test - Data:', data))
                .catch(error => console.error('Manual API test - Error:', error));
        }
    }

    // Make debug function available globally
    window.debugSearch = debugSearch;

    // --- Initializer ---
    loadCompanies();
});