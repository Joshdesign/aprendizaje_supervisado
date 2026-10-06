// ==========================================================================
// Obsidian Theme - Client-Side JS for Titanic ML Dashboard
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Setup & Event Listeners
    initTabNavigation();
    initFormControls();
    loadDiagnostics();
    loadDatasetInsights();
});

// Tab Navigation logic
function initTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTabId = btn.getAttribute('data-tab');

            // Remove active states
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Set active states
            btn.classList.add('active');
            document.getElementById(targetTabId).classList.add('active');
        });
    });
}

// Form sliders & dynamic value updates
function initFormControls() {
    const ageSlider = document.getElementById('age');
    const ageVal = document.getElementById('age-val');
    ageSlider.addEventListener('input', (e) => {
        ageVal.textContent = e.target.value;
    });

    const fareSlider = document.getElementById('fare');
    const fareVal = document.getElementById('fare-val');
    fareSlider.addEventListener('input', (e) => {
        fareVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Form submission
    const form = document.getElementById('predictor-form');
    form.addEventListener('submit', handlePredictionSubmit);
}

// Helper to set Fare from shortcuts
window.setFare = function(val) {
    const fareSlider = document.getElementById('fare');
    const fareVal = document.getElementById('fare-val');
    fareSlider.value = val;
    fareVal.textContent = parseFloat(val).toFixed(2);
};

// Handle submission of the passenger manifest
async function handlePredictionSubmit(e) {
    e.preventDefault();

    const pclass = parseInt(document.getElementById('pclass').value);
    const sex = document.getElementById('sex').value;
    const age = parseFloat(document.getElementById('age').value);
    const fare = parseFloat(document.getElementById('fare').value);
    const sibsp = parseInt(document.getElementById('sibsp').value);
    const parch = parseInt(document.getElementById('parch').value);
    const embarked = document.getElementById('embarked').value;

    const payload = { Pclass: pclass, Sex: sex, Age: age, Fare: fare, SibSp: sibsp, Parch: parch, Embarked: embarked };

    // UI state: Show loading or hide placeholder
    const placeholder = document.getElementById('result-placeholder');
    const content = document.getElementById('result-content');
    
    placeholder.classList.add('hidden');
    content.classList.remove('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.success) {
            updatePredictionResults(data.prediction, data.narrative);
        } else {
            alert("Error al procesar la predicción: " + data.error);
        }
    } catch (err) {
        console.error("Prediction error:", err);
        alert("Ocurrió un error al contactar al servidor.");
    }
}

// Animate progress circle and text values
function updatePredictionResults(pred, narrative) {
    // 1. Random Forest
    const rfProb = pred.rf.probability;
    const rfSurvived = pred.rf.survived;
    animateProgressCircle('rf-progress', 'rf-prob-text', rfProb);
    updateResultBadge('rf-badge', rfSurvived);

    // 2. Logistic Regression
    const lrProb = pred.lr.probability;
    const lrSurvived = pred.lr.survived;
    animateProgressCircle('lr-progress', 'lr-prob-text', lrProb);
    updateResultBadge('lr-badge', lrSurvived);

    // 3. Narrative verdict
    const narrativeText = document.getElementById('verdict-narrative');
    // Format bold items from markdown returned
    let formattedText = narrative.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formattedText = formattedText.replace(/### (.*?)\n/g, '<h3>$1</h3>');
    formattedText = formattedText.replace(/\n\n/g, '<br><br>');
    formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    narrativeText.innerHTML = formattedText;
}

function animateProgressCircle(circleId, textId, targetPercent) {
    const circle = document.getElementById(circleId);
    const text = document.getElementById(textId);
    
    const radius = circle.r.baseVal.value;
    const circumference = 2 * Math.PI * radius; // Approx 251.2
    
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    
    // Counting animation
    let startVal = 0;
    const duration = 800; // ms
    const startTime = performance.now();
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing out cubic
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const currentVal = Math.round(startVal + (targetPercent * 100 - startVal) * easeProgress);
        
        text.textContent = `${currentVal}%`;
        
        // Update SVG circle stroke dashoffset
        const offset = circumference - (currentVal / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

function updateResultBadge(badgeId, survived) {
    const badge = document.getElementById(badgeId);
    badge.className = 'badge'; // reset
    if (survived) {
        badge.classList.add('badge-survived');
        badge.textContent = 'Sobrevive';
    } else {
        badge.classList.add('badge-lost');
        badge.textContent = 'Fallece';
    }
}

// 2. Fetch and Render Model Diagnostics
async function loadDiagnostics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        if (data.success) {
            // Update Text Metrics
            populateModelMetrics('rf', data.rf);
            populateModelMetrics('lr', data.lr);

            // Draw Charts
            renderFeatureChart('rf-features-chart', data.rf.feature_importances, '#c5a059', 'Importancia Relativa');
            renderFeatureChart('lr-features-chart', data.lr.feature_importances, '#3fc1c9', 'Magnitud del Coeficiente', true);
        }
    } catch (err) {
        console.error("Error loading model metrics:", err);
    }
}

function populateModelMetrics(prefix, metrics) {
    document.getElementById(`${prefix}-accuracy`).textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById(`${prefix}-precision`).textContent = `${(metrics.precision * 100).toFixed(1)}%`;
    document.getElementById(`${prefix}-recall`).textContent = `${(metrics.recall * 100).toFixed(1)}%`;
    document.getElementById(`${prefix}-f1`).textContent = `${(metrics.f1 * 100).toFixed(1)}%`;

    // Confusion Matrix cells
    document.getElementById(`${prefix}-cm-tn`).textContent = metrics.confusion_matrix.tn;
    document.getElementById(`${prefix}-cm-fp`).textContent = metrics.confusion_matrix.fp;
    document.getElementById(`${prefix}-cm-fn`).textContent = metrics.confusion_matrix.fn;
    document.getElementById(`${prefix}-cm-tp`).textContent = metrics.confusion_matrix.tp;
}

// Feature Importance rendering using Chart.js
function renderFeatureChart(canvasId, features, mainColor, datasetLabel, isCoefficient = false) {
    // Sort features by absolute value if coefficient, or value if importance
    const sortedFeatures = [...features].sort((a, b) => {
        const valA = isCoefficient ? Math.abs(a.value) : a.value;
        const valB = isCoefficient ? Math.abs(b.value) : b.value;
        return valB - valA;
    });

    const labels = sortedFeatures.map(f => translateFeatureName(f.name));
    const values = sortedFeatures.map(f => f.value);

    // Mapeador de colores de barra para coeficientes positivos/negativos
    let backgroundColors = mainColor;
    if (isCoefficient) {
        backgroundColors = values.map(v => v >= 0 ? '#2ec4b6' : '#ff4d6d');
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: datasetLabel,
                data: values,
                backgroundColor: backgroundColors,
                borderColor: isCoefficient ? 'transparent' : mainColor,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#161921',
                    titleColor: '#e2e8f0',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', size: 11 } }
                }
            }
        }
    });
}

function translateFeatureName(name) {
    const translations = {
        'Sex_female': 'Género: Mujer',
        'Sex_male': 'Género: Hombre',
        'Pclass': 'Clase de Boleto',
        'Age': 'Edad',
        'Fare': 'Tarifa pagada',
        'SibSp': 'Hermanos/Esposo(a)',
        'Parch': 'Padres/Hijos',
        'Embarked_C': 'Embarque: Cherbourg',
        'Embarked_Q': 'Embarque: Queenstown',
        'Embarked_S': 'Embarque: Southampton'
    };
    return translations[name] || name;
}

// 3. Fetch and Render Historical Dataset Insights
async function loadDatasetInsights() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        if (data.success) {
            // Populate stats summary bar
            document.getElementById('stat-total').textContent = data.summary.total;
            document.getElementById('stat-survived').textContent = data.summary.survived;
            document.getElementById('stat-rate').textContent = `${(data.summary.rate * 100).toFixed(1)}%`;

            // Draw Charts
            renderGenderChart(data.sex);
            renderClassChart(data.class);
            renderAgeChart(data.age);
            renderEmbarkedChart(data.embarked);
        }
    } catch (err) {
        console.error("Error loading historical insights:", err);
    }
}

// Gender chart
function renderGenderChart(sexData) {
    const ctx = document.getElementById('gender-chart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Mujeres', 'Hombres'],
            datasets: [{
                data: [sexData.female * 100, sexData.male * 100],
                backgroundColor: ['#2ec4b6', '#ff4d6d'],
                borderWidth: 2,
                borderColor: '#161921'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', size: 11 } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) { return ` Tasa Supervivencia: ${context.raw.toFixed(1)}%`; }
                    }
                }
            }
        }
    });
}

// Class chart
function renderClassChart(classData) {
    const ctx = document.getElementById('class-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['1a Clase', '2a Clase', '3a Clase'],
            datasets: [{
                data: [classData['1'] * 100, classData['2'] * 100, classData['3'] * 100],
                backgroundColor: ['rgba(197, 160, 89, 0.8)', 'rgba(197, 160, 89, 0.5)', 'rgba(197, 160, 89, 0.2)'],
                borderColor: '#c5a059',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) { return ` Tasa Supervivencia: ${context.raw.toFixed(1)}%`; }
                    }
                }
            },
            scales: {
                x: { ticks: { color: '#e2e8f0' }, grid: { display: false } },
                y: {
                    max: 100,
                    ticks: { color: '#94a3b8', callback: value => `${value}%` },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// Age Group Chart
function renderAgeChart(ageData) {
    const labels = Object.keys(ageData);
    const rates = labels.map(lbl => ageData[lbl].rate * 100);
    const totals = labels.map(lbl => ageData[lbl].total);

    const ctx = document.getElementById('age-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    type: 'line',
                    label: 'Tasa de Supervivencia (%)',
                    data: rates,
                    borderColor: '#3fc1c9',
                    backgroundColor: 'rgba(63, 193, 201, 0.1)',
                    yAxisID: 'yRate',
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 4
                },
                {
                    label: 'Pasajeros Totales',
                    data: totals,
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    yAxisID: 'yCount',
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans' } }
                }
            },
            scales: {
                x: { ticks: { color: '#e2e8f0' }, grid: { display: false } },
                yRate: {
                    type: 'linear',
                    position: 'left',
                    max: 100,
                    title: { display: true, text: 'Supervivencia %', color: '#3fc1c9' },
                    ticks: { color: '#94a3b8', callback: value => `${value}%` },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                yCount: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'N° de Pasajeros', color: '#94a3b8' },
                    ticks: { color: '#94a3b8' },
                    grid: { display: false }
                }
            }
        }
    });
}

// Embarkation Chart
function renderEmbarkedChart(embarkedData) {
    const labels = Object.keys(embarkedData);
    const rates = labels.map(k => embarkedData[k] * 100);

    const ctx = document.getElementById('embarked-chart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: rates,
                backgroundColor: ['#2ec4b6', '#3fc1c9', '#ff4d6d'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) { return ` Tasa Supervivencia: ${context.raw.toFixed(1)}%`; }
                    }
                }
            },
            scales: {
                x: { ticks: { color: '#e2e8f0' }, grid: { display: false } },
                y: {
                    max: 100,
                    ticks: { color: '#94a3b8', callback: value => `${value}%` },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}
