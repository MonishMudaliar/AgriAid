// Global variables - Now using separate HTML files
let currentPage = 'home';

// Page navigation functions - Removed since using separate files
// Navigation is now handled by HTML href links

// Crop Recommendation - Event listener is now added in DOMContentLoaded

function displayCropResults(result) {
    const resultsDiv = document.getElementById('crop-results');
    const modelDetailsDiv = document.getElementById('model-details');
    const cropDetailsDiv = document.getElementById('crop-details');
    const topCropDiv = document.getElementById('top-crop');
    const confidenceDiv = document.getElementById('confidence');
    const recommendedCropsDiv = document.getElementById('recommended-crops');
    const probBarsDiv = document.getElementById('prob-bars');
    
    // Show results
    resultsDiv.style.display = 'block';
    
    // Model information
    modelDetailsDiv.innerHTML = `
        <div style="background: rgba(74, 124, 89, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(74, 124, 89, 0.3);">
            <strong>Model Used:</strong> ${result.model_used}<br>
            <strong>Accuracy:</strong> ${result.model_accuracy}<br>
            <strong>Confidence:</strong> ${(result.prediction_confidence * 100).toFixed(1)}%
        </div>
    `;
    
    // Input parameters
    cropDetailsDiv.innerHTML = `
        <h4 style="color: #4a7c59; margin-bottom: 15px;">📊 Input Parameters</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
            <div style="background: rgba(74, 124, 89, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(74, 124, 89, 0.3);">
                <strong>N:</strong> ${result.input_parameters.N} mg/kg
            </div>
            <div style="background: rgba(74, 124, 89, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(74, 124, 89, 0.3);">
                <strong>P:</strong> ${result.input_parameters.P} mg/kg
            </div>
            <div style="background: rgba(74, 124, 89, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(74, 124, 89, 0.3);">
                <strong>K:</strong> ${result.input_parameters.K} mg/kg
            </div>
            <div style="background: rgba(74, 124, 89, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(74, 124, 89, 0.3);">
                <strong>pH:</strong> ${result.input_parameters.ph}
            </div>
            <div style="background: rgba(74, 124, 89, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(74, 124, 89, 0.3);">
                <strong>Rainfall:</strong> ${result.input_parameters.rainfall} mm
            </div>
        </div>
    `;
    
    // Top recommendation
    topCropDiv.textContent = result.predicted_crop;
    confidenceDiv.textContent = `Confidence: ${(result.prediction_confidence * 100).toFixed(1)}%`;
    
    // All recommendations
    let cropsHTML = '';
    result.recommended_crops.forEach(crop => {
        const prob = result.crop_probabilities[crop] || 0;
        cropsHTML += `
            <div style="background: rgba(74, 124, 89, 0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(74, 124, 89, 0.3); text-align: center;">
                <h4 style="color: #4a7c59; margin-bottom: 10px;">${crop}</h4>
                <p style="color: #6b9c7a; font-size: 1.2rem; font-weight: bold;">${(prob * 100).toFixed(1)}%</p>
            </div>
        `;
    });
    recommendedCropsDiv.innerHTML = cropsHTML;
    
    // Probability bars
    let probHTML = '';
    Object.entries(result.crop_probabilities).forEach(([crop, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        probHTML += `
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: #4a7c59;">${crop}</span>
                    <span style="color: #6b9c7a;">${percentage}%</span>
                </div>
                <div style="background: rgba(74, 124, 89, 0.2); height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #4a7c59, #6b9c7a); height: 100%; width: ${percentage}%; transition: width 1s ease;"></div>
                </div>
            </div>
        `;
    });
    probBarsDiv.innerHTML = probHTML;
}

// Weather functions
async function fetchWeatherData() {
    try {
        const response = await fetch('/api/weather-prediction');
        const data = await response.json();
        displayWeatherData(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('current-weather').innerHTML = '<p style="color: #e74c3c;">Error loading weather data</p>';
        document.getElementById('forecast-grid').innerHTML = '<p style="color: #e74c3c;">Error loading forecast data</p>';
    }
}

function displayWeatherData(data) {
    // Current weather
    const currentWeatherDiv = document.getElementById('current-weather');
    currentWeatherDiv.innerHTML = `
        <div class="weather-card">
            <h4>🌡️ Temperature</h4>
            <p>${data.current_weather.temperature}°C</p>
        </div>
        <div class="weather-card">
            <h4>💧 Humidity</h4>
            <p>${data.current_weather.humidity}%</p>
        </div>
        <div class="weather-card">
            <h4>🌧️ Rainfall</h4>
            <p>${data.current_weather.rainfall}mm</p>
        </div>
        <div class="weather-card">
            <h4>💨 Wind Speed</h4>
            <p>${data.current_weather.wind_speed} km/h</p>
        </div>
    `;
    
    // Forecast
    const forecastDiv = document.getElementById('forecast-grid');
    let forecastHTML = '';
    data.predictions.forEach(day => {
        forecastHTML += `
            <div class="weather-card">
                <h4>${day.date}</h4>
                <p style="font-size: 1.5rem; margin: 10px 0;">${day.condition}</p>
                <p><strong>${day.temperature}°C</strong></p>
                <p>Humidity: ${day.humidity}%</p>
                <p>Rain: ${day.rainfall}mm</p>
            </div>
        `;
    });
    forecastDiv.innerHTML = forecastHTML;
}

// Market functions
async function fetchMarketData() {
    try {
        const response = await fetch('/api/market-trends');
        const data = await response.json();
        displayMarketData(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('market-tbody').innerHTML = '<tr><td colspan="4" style="color: #e74c3c; text-align: center;">Error loading market data</td></tr>';
    }
}

function displayMarketData(data) {
    const tbody = document.getElementById('market-tbody');
    const lastUpdated = document.getElementById('last-updated');
    const insightsGrid = document.getElementById('insights-grid');
    
    // Update last updated time
    lastUpdated.textContent = `Last updated: ${new Date(data.last_updated).toLocaleString()}`;
    
    // Market table
    let tableHTML = '';
    data.trends.forEach(trend => {
        const trendColor = trend.trend === 'Rising' ? '#27ae60' : trend.trend === 'Falling' ? '#e74c3c' : '#f39c12';
        const demandColor = trend.demand === 'High' ? '#27ae60' : trend.demand === 'Medium' ? '#f39c12' : '#e74c3c';
        
        tableHTML += `
            <tr>
                <td>${trend.crop}</td>
                <td><strong>₹${trend.price}</strong></td>
                <td style="color: ${trendColor};">${trend.trend}</td>
                <td style="color: ${demandColor};">${trend.demand}</td>
            </tr>
        `;
    });
    tbody.innerHTML = tableHTML;
    
    // Market insights
    let insightsHTML = '';
    data.trends.forEach(trend => {
        if (trend.trend === 'Rising') {
            insightsHTML += `
                <div style="background: rgba(39, 174, 96, 0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(39, 174, 96, 0.3);">
                    <h4 style="color: #27ae60;">📈 ${trend.crop} - Rising Trend</h4>
                    <p style="color: #c8e6c9;">Good time to sell ${trend.crop.toLowerCase()} at current prices.</p>
                </div>
            `;
        } else if (trend.trend === 'Falling') {
            insightsHTML += `
                <div style="background: rgba(231, 76, 60, 0.1); padding: 20px; border-radius: 15px; border: 2px solid rgba(231, 76, 60, 0.3);">
                    <h4 style="color: #e74c3c;">📉 ${trend.crop} - Falling Trend</h4>
                    <p style="color: #c8e6c9;">Consider holding ${trend.crop.toLowerCase()} or selling at better prices.</p>
                </div>
            `;
        }
    });
    insightsGrid.innerHTML = insightsHTML;
}

// E-commerce functions
function toggleAddProductForm() {
    const form = document.getElementById('add-product-form');
    const btn = document.getElementById('add-product-btn');
    
    if (form.style.display === 'none') {
        form.style.display = 'block';
        btn.textContent = '❌ Cancel';
    } else {
        form.style.display = 'none';
        btn.textContent = '➕ Add New Product';
    }
}

async function fetchProducts() {
    try {
        const response = await fetch('/api/products');
        const data = await response.json();
        displayProducts(data.products);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('products-grid').innerHTML = '<p style="color: #e74c3c;">Error loading products</p>';
    }
}

function displayProducts(products) {
    const productsGrid = document.getElementById('products-grid');
    const productsCount = document.getElementById('products-count');
    
    productsCount.textContent = products.length;
    
    let productsHTML = '';
    products.forEach(product => {
        productsHTML += `
            <div class="product-card">
                <h3>${product.name}</h3>
                <div class="product-price">₹${product.price}</div>
                <div class="product-details">
                    <p><strong>Seller:</strong> ${product.seller}</p>
                    <p><strong>Quantity:</strong> ${product.quantity}</p>
                    <p><strong>Location:</strong> ${product.location}</p>
                </div>
            </div>
        `;
    });
    productsGrid.innerHTML = productsHTML;
}

// Add product form submission - Event listener is now added in DOMContentLoaded

// Chatbot functions
function toggleChatbot() {
    const chatbotWindow = document.getElementById('chatbot-window');
    if (chatbotWindow.style.display === 'none') {
        chatbotWindow.style.display = 'flex';
    } else {
        chatbotWindow.style.display = 'none';
    }
}

function handleChatbotKeyPress(event) {
    if (event.key === 'Enter') {
        const input = event.target;
        const message = input.value.trim();
        
        if (message) {
            addUserMessage(message);
            input.value = '';
            
            // Simulate bot response
            setTimeout(() => {
                const botResponse = getBotResponse(message);
                addBotMessage(botResponse);
            }, 1000);
        }
    }
}

function addUserMessage(message) {
    const messagesDiv = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.textContent = message;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addBotMessage(message) {
    const messagesDiv = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.textContent = message;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function getBotResponse(message) {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        return "Hello! I'm your agricultural assistant. How can I help you today?";
    } else if (lowerMessage.includes('crop') || lowerMessage.includes('recommend')) {
        return "I can help you with crop recommendations! Just provide your soil parameters (N, P, K, pH, rainfall) and I'll suggest the best crops.";
    } else if (lowerMessage.includes('weather')) {
        return "I can provide weather forecasts to help you plan your farming activities. Check the weather page for detailed forecasts.";
    } else if (lowerMessage.includes('market') || lowerMessage.includes('price')) {
        return "I can show you current market trends and prices for various crops. Visit the market trends page for the latest information.";
    } else if (lowerMessage.includes('sell') || lowerMessage.includes('buy')) {
        return "You can buy and sell products on our marketplace! Just go to the e-commerce page and add your products or browse available items.";
    } else {
        return "I'm here to help with your farming needs! You can ask me about crops, weather, market trends, or how to sell your products.";
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    console.log('AgriTech Hub loaded successfully!');
    
    // Check if we're on a page that needs data loading
    if (document.getElementById('crop-form')) {
        console.log('Crop recommendation page loaded');
        // Add event listener for crop form
        document.getElementById('crop-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                N: parseInt(formData.get('N')),
                P: parseInt(formData.get('P')),
                K: parseInt(formData.get('K')),
                ph: parseFloat(formData.get('ph')),
                rainfall: parseFloat(formData.get('rainfall')),
                model_choice: formData.get('model_choice')
            };

            try {
                const response = await fetch('/api/crop-recommendation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                displayCropResults(result);
            } catch (error) {
                console.error('Error:', error);
                alert('Error getting crop recommendation. Please try again.');
            }
        });
    }
    if (document.getElementById('weather-results')) {
        console.log('Weather page loaded');
        fetchWeatherData();
    }
    if (document.getElementById('market-results')) {
        console.log('Market page loaded');
        fetchMarketData();
    }
    if (document.getElementById('products-results')) {
        console.log('E-commerce page loaded');
        fetchProducts();
        
        // Add event listener for add product form
        document.getElementById('add-product-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const product = {
                name: document.getElementById('product-name').value,
                price: parseInt(document.getElementById('product-price').value),
                seller: document.getElementById('product-seller').value,
                quantity: document.getElementById('product-quantity').value,
                location: document.getElementById('product-location').value
            };

            try {
                const response = await fetch('/api/products', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(product)
                });

                const result = await response.json();
                if (result.message) {
                    alert('Product added successfully!');
                    this.reset();
                    toggleAddProductForm();
                    fetchProducts();
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error adding product. Please try again.');
            }
        });
    }
});
