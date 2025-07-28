// Fake Job Posting Detector Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('jobForm');
    const resultsCard = document.getElementById('resultsCard');
    const loadingCard = document.getElementById('loadingCard');
    const infoCard = document.getElementById('infoCard');
    const loadExampleBtn = document.getElementById('loadExample');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading
        showLoading();
        
        // Collect form data
        const formData = new FormData(form);
        const jobData = {
            title: formData.get('title'),
            location: formData.get('location') || 'Unknown',
            department: formData.get('department') || 'Unknown',
            salary_range: formData.get('salary_range') || 'Unknown',
            company_profile: formData.get('company_profile') || '',
            description: formData.get('description'),
            requirements: formData.get('requirements') || '',
            benefits: formData.get('benefits') || '',
            telecommuting: formData.get('telecommuting') ? 1 : 0,
            has_company_logo: formData.get('has_company_logo') ? 1 : 0,
            has_questions: formData.get('has_questions') ? 1 : 0,
            employment_type: formData.get('employment_type') || 'Unknown',
            required_experience: formData.get('required_experience') || 'Unknown',
            required_education: formData.get('required_education') || 'Unknown',
            industry: formData.get('industry') || 'Unknown',
            function: formData.get('function') || 'Unknown'
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(jobData)
            });

            const result = await response.json();
            
            if (response.ok) {
                displayResults(result);
            } else {
                showError(result.error || 'An error occurred during analysis.');
            }
        } catch (error) {
            showError('Network error. Please try again.');
        } finally {
            hideLoading();
        }
    });

    // Load example button handler
    loadExampleBtn.addEventListener('click', function() {
        loadRandomExample();
    });

    function showLoading() {
        loadingCard.style.display = 'block';
        resultsCard.style.display = 'none';
        infoCard.style.display = 'none';
    }

    function hideLoading() {
        loadingCard.style.display = 'none';
        infoCard.style.display = 'block';
    }

    function displayResults(result) {
        const predictionResult = document.getElementById('predictionResult');
        const confidenceBar = document.getElementById('confidenceBar');
        const featureAnalysis = document.getElementById('featureAnalysis');

        // Determine result styling
        const isFraudulent = result.is_fraudulent;
        const confidence = result.confidence;
        const confidencePercent = Math.round(confidence * 100);

        // Create prediction result HTML
        let resultHTML = `
            <div class="result-card ${isFraudulent ? 'result-fraudulent' : 'result-legitimate'} fade-in">
                <div class="d-flex align-items-center mb-3">
                    <i class="fas ${isFraudulent ? 'fa-exclamation-triangle text-danger' : 'fa-check-circle text-success'} fa-2x me-3"></i>
                    <div>
                        <h5 class="mb-1">${isFraudulent ? 'Potentially Fraudulent' : 'Likely Legitimate'}</h5>
                        <p class="mb-0 text-muted">Confidence: ${confidencePercent}%</p>
                    </div>
                </div>
            </div>
        `;

        // Create confidence bar
        let confidenceHTML = `
            <div class="confidence-bar">
                <div class="confidence-fill ${isFraudulent ? 'confidence-fraudulent' : 'confidence-legitimate'}" 
                     style="width: ${confidencePercent}%"></div>
            </div>
            <div class="d-flex justify-content-between mt-2">
                <small class="text-muted">0%</small>
                <small class="text-muted">${confidencePercent}%</small>
                <small class="text-muted">100%</small>
            </div>
        `;

        // Create feature analysis
        let analysisHTML = '<h6 class="mb-3"><i class="fas fa-chart-line me-2"></i>Feature Analysis</h6>';
        
        if (result.feature_analysis) {
            const analysis = result.feature_analysis;
            
            // Suspicious keywords
            if (analysis.suspicious_keywords && analysis.suspicious_keywords.length > 0) {
                analysisHTML += '<div class="mb-3"><strong class="text-warning">⚠️ Suspicious Keywords Found:</strong>';
                analysis.suspicious_keywords.forEach(keyword => {
                    analysisHTML += `<div class="feature-item suspicious-keyword">• ${keyword}</div>`;
                });
                analysisHTML += '</div>';
            }

            // Red flags
            if (analysis.red_flags && analysis.red_flags.length > 0) {
                analysisHTML += '<div class="mb-3"><strong class="text-danger">🚩 Red Flags:</strong>';
                analysis.red_flags.forEach(flag => {
                    analysisHTML += `<div class="feature-item red-flag">• ${flag}</div>`;
                });
                analysisHTML += '</div>';
            }

            // Green flags
            if (analysis.green_flags && analysis.green_flags.length > 0) {
                analysisHTML += '<div class="mb-3"><strong class="text-success">✅ Positive Indicators:</strong>';
                analysis.green_flags.forEach(flag => {
                    analysisHTML += `<div class="feature-item green-flag">• ${flag}</div>`;
                });
                analysisHTML += '</div>';
            }

            if (analysis.suspicious_keywords.length === 0 && 
                analysis.red_flags.length === 0 && 
                analysis.green_flags.length === 0) {
                analysisHTML += '<div class="text-muted">No specific indicators found.</div>';
            }
        }

        // Update DOM
        predictionResult.innerHTML = resultHTML;
        confidenceBar.innerHTML = confidenceHTML;
        featureAnalysis.innerHTML = analysisHTML;

        // Show results
        resultsCard.style.display = 'block';
        infoCard.style.display = 'none';

        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    function showError(message) {
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerHTML = `
            <div class="alert alert-danger fade-in">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${message}
            </div>
        `;
        resultsCard.style.display = 'block';
        infoCard.style.display = 'none';
    }

    async function loadRandomExample() {
        try {
            const response = await fetch('/examples');
            const examples = await response.json();
            
            if (examples.length > 0) {
                const randomExample = examples[Math.floor(Math.random() * examples.length)];
                populateForm(randomExample);
            }
        } catch (error) {
            console.error('Error loading example:', error);
        }
    }

    function populateForm(example) {
        // Populate form fields
        document.getElementById('title').value = example.title || '';
        document.getElementById('location').value = example.location || '';
        document.getElementById('department').value = example.department || '';
        document.getElementById('salary_range').value = example.salary_range || '';
        document.getElementById('company_profile').value = example.company_profile || '';
        document.getElementById('description').value = example.description || '';
        document.getElementById('requirements').value = example.requirements || '';
        document.getElementById('benefits').value = example.benefits || '';
        document.getElementById('employment_type').value = example.employment_type || 'Full-time';
        document.getElementById('required_experience').value = example.required_experience || 'Not Applicable';
        document.getElementById('required_education').value = example.required_education || 'High School or equivalent';
        document.getElementById('industry').value = example.industry || '';
        document.getElementById('function').value = example.function || '';
        
        // Set checkboxes
        document.getElementById('telecommuting').checked = example.telecommuting === 1;
        document.getElementById('has_company_logo').checked = example.has_company_logo === 1;
        document.getElementById('has_questions').checked = example.has_questions === 1;

        // Show success message
        showToast('Example loaded successfully!', 'success');
    }

    function showToast(message, type = 'info') {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'success' ? 'success' : 'info'} position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>
            ${message}
        `;

        document.body.appendChild(toast);

        // Remove toast after 3 seconds
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    // Add some interactive features
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    });

    // Add character count for important fields
    const titleInput = document.getElementById('title');
    const descriptionTextarea = document.getElementById('description');
    
    function addCharacterCount(element, maxLength = 100) {
        const counter = document.createElement('small');
        counter.className = 'text-muted character-count';
        counter.style.display = 'block';
        counter.style.marginTop = '5px';
        
        element.parentNode.appendChild(counter);
        
        function updateCount() {
            const count = element.value.length;
            counter.textContent = `${count}/${maxLength} characters`;
            counter.style.color = count > maxLength * 0.8 ? '#dc3545' : '#6c757d';
        }
        
        element.addEventListener('input', updateCount);
        updateCount();
    }

    if (titleInput) addCharacterCount(titleInput, 100);
    if (descriptionTextarea) addCharacterCount(descriptionTextarea, 2000);
}); 