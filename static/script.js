const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const previewSection = document.getElementById('preview-section');
const originalImage = document.getElementById('original-image');
const resultImage = document.getElementById('result-image');
const loadingSpinner = document.getElementById('loading-spinner');
const resultStats = document.getElementById('result-stats');
const percentageText = document.getElementById('corrosion-percentage');
const confidenceText = document.getElementById('ai-confidence');
const progressFill = document.getElementById('progress-fill');
const statusText = document.getElementById('corrosion-status');
const resetBtn = document.getElementById('reset-btn');

// ROI Elements
const roiCanvas = document.getElementById('roi-canvas');
const roiControls = document.getElementById('roi-controls');
const drawBtn = document.getElementById('draw-btn');
const heatmapBtn = document.getElementById('heatmap-btn');
const calibrateBtn = document.getElementById('calibrate-btn');
const clearRoiBtn = document.getElementById('clear-roi-btn');
const ctx = roiCanvas ? roiCanvas.getContext('2d') : null;

// Calibration Elements
const calibInputBox = document.getElementById('calibration-input-box');
const calibRealLengthInput = document.getElementById('calib-real-length');
const calibSaveBtn = document.getElementById('calib-save-btn');
const calibCancelBtn = document.getElementById('calib-cancel-btn');
const corrosionCm2Text = document.getElementById('corrosion-cm2');

let isDrawingMode = false;
let isCalibrationMode = false;
let useHeatmap = false;
let polygonPoints = [];
let arucoSizeCm = 0;
let imgNaturalWidth = 0;
let imgNaturalHeight = 0;
let currentFile = null;

// Handle drag and drop
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropArea.classList.add('highlight');
}

function unhighlight() {
    dropArea.classList.remove('highlight');
}

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    currentFile = file; // Simpan file ke global state
    
    // Show preview UI
    dropArea.classList.add('hidden');
    previewSection.classList.remove('hidden');
    
    // Display original image
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
        originalImage.onload = () => {
            // Setup canvas
            roiCanvas.width = originalImage.clientWidth;
            roiCanvas.height = originalImage.clientHeight;
            imgNaturalWidth = originalImage.naturalWidth;
            imgNaturalHeight = originalImage.naturalHeight;
            roiControls.classList.remove('hidden');
    document.querySelector('.container').classList.add('preview-mode');
            
            // Start analyzing initially without ROI
            analyzeImage(file);
        }
    }
    reader.readAsDataURL(file);
}

async function analyzeImage(file) {
    // Reset UI state
    resultImage.classList.add('hidden');
    resultStats.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_heatmap', useHeatmap);
    
    if (arucoSizeCm > 0) {
        formData.append('aruco_size_cm', arucoSizeCm);
    }
    
    // Add polygon points if any (normalize to 0.0 - 1.0)
    if (polygonPoints.length > 2) {
        const normalizedPoints = polygonPoints.map(p => ({
            x: p.x / roiCanvas.width,
            y: p.y / roiCanvas.height
        }));
        formData.append('roi_polygon', JSON.stringify(normalizedPoints));
    }

    try {
        // Send to FastAPI backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            if (isCalibrationMode) {
                calibrateBtn.click(); // turn off if error
            }
            resetBtn.click();
            return;
        }
        
        // Hide loading
        loadingSpinner.classList.add('hidden');
        
        // Show result image
        resultImage.src = 'data:image/jpeg;base64,' + data.image_base64;
        resultImage.classList.remove('hidden');
        
        // Update stats
        updateStats(data.percentage, data.confidence, data.severe_pct, data.light_pct, data.absolute_area_cm2);

    } catch (error) {
        console.error(error);
        alert('Gagal menganalisis gambar. Pastikan backend sudah berjalan.');
        resetUI();
    }
}

function updateStats(percentage, confidence = 0, severePct = 0, lightPct = 0, absoluteAreaCm2 = 0) {
    resultStats.classList.remove('hidden');
    
    const end = parseFloat(percentage);
    const endConf = parseFloat(confidence);
    
    // ISO 4628-3 Rust Grade Mapping
    let statusMsg = '';
    let statusColor = '';
    
    if (end < 0.05) {
        statusMsg = 'Grade: Ri 0 (Sangat Aman)';
        statusColor = 'var(--accent-safe)';
    } else if (end < 0.5) {
        statusMsg = 'Grade: Ri 1-2 (Korosi Sangat Ringan)';
        statusColor = 'var(--accent-safe)';
    } else if (end < 1.0) {
        statusMsg = 'Grade: Ri 3 (Korosi Sedang)';
        statusColor = 'var(--accent-warning)';
    } else if (end < 8.0) {
        statusMsg = 'Grade: Ri 4 (Korosi Parah)';
        statusColor = 'var(--accent-danger)';
    } else {
        statusMsg = 'Grade: Ri 5 (Kritis / Sangat Parah)';
        statusColor = 'var(--accent-danger)';
    }
    
    progressFill.style.backgroundColor = statusColor;
    statusText.textContent = statusMsg + ' - Ref: ISO 4628-3';
    statusText.style.color = statusColor;
    
    // Animate numbers
    let current = 0;
    let currentConf = 0;
    const duration = 1000;
    const stepTime = 20;
    const steps = duration / stepTime;
    let increment = end / steps;
    let incrementConf = endConf / steps;
    
    if (end === 0) increment = 0;
    if (endConf === 0) incrementConf = 0;

    const timer = setInterval(() => {
        current += increment;
        currentConf += incrementConf;
        
        if (current >= end) current = end;
        if (currentConf >= endConf) currentConf = endConf;
        
        percentageText.textContent = current.toFixed(2) + '%';
        if (confidenceText) confidenceText.textContent = currentConf.toFixed(1) + '%';
        
        if (current === end && currentConf === endConf) {
            clearInterval(timer);
        }
    }, stepTime);

    // Update progress bar
    progressFill.style.width = end + '%';
    
    // Update Breakdown & Action Plan
    const breakdownBox = document.getElementById('breakdown-box');
    const lightSpan = document.getElementById('light-pct');
    const severeSpan = document.getElementById('severe-pct');
    const actionPlanSpan = document.getElementById('action-plan');
    
    if (breakdownBox && lightSpan && severeSpan) {
        breakdownBox.classList.remove('hidden');
        lightSpan.textContent = parseFloat(lightPct || 0).toFixed(1) + '%';
        severeSpan.textContent = parseFloat(severePct || 0).toFixed(1) + '%';
        
        if (actionPlanSpan) {
            if (end >= 1.0 || severePct > 20) {
                actionPlanSpan.textContent = "Segera lakukan inspeksi ketebalan (Ultrasonic Testing / UT). Persiapkan Surface Preparation ke standar ISO 8501 Sa 2.5.";
                actionPlanSpan.style.color = "var(--accent-danger)";
            } else if (end >= 0.05) {
                actionPlanSpan.textContent = "Lakukan pemantauan berkala dan perbaikan coating (Coating Repair).";
                actionPlanSpan.style.color = "var(--accent-warning)";
            } else {
                actionPlanSpan.textContent = "Tidak ada tindakan khusus. Aman.";
                actionPlanSpan.style.color = "var(--accent-safe)";
            }
        }
    }
    
    // Update cm2
    if (corrosionCm2Text) {
        if (absoluteAreaCm2 && absoluteAreaCm2 > 0) {
            corrosionCm2Text.textContent = `(~ ${parseFloat(absoluteAreaCm2).toFixed(1)} cm²)`;
        } else {
            corrosionCm2Text.textContent = '';
        }
    }
}

resetBtn.addEventListener('click', resetUI);

function resetUI() {
    dropArea.classList.remove('hidden');
    previewSection.classList.add('hidden');
    roiControls.classList.add('hidden');
    
    // Kembalikan ukuran card ke default (lebih sempit)
    document.querySelector('.container').classList.remove('preview-mode');
    
    fileInput.value = '';
    polygonPoints = [];
    isDrawingMode = false;
    roiCanvas.style.pointerEvents = 'none';
    drawBtn.textContent = 'Tandai Objek (Pen Tool)';
    drawBtn.style.color = '';
    clearRoiBtn.classList.add('hidden');
    if (ctx) ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
    currentFile = null;
}

// ROI Drawing Logic
if (drawBtn) {
    drawBtn.addEventListener('click', () => {
        isDrawingMode = !isDrawingMode;
        if (isDrawingMode) {
            roiCanvas.style.pointerEvents = 'auto';
            roiCanvas.style.cursor = 'crosshair';
            drawBtn.textContent = 'Selesai & Analisis';
            drawBtn.style.color = 'var(--accent-safe)';
            // resultImage.classList.add('hidden'); // Dihapus karena gambar ada di kolom terpisah
        } else {
            roiCanvas.style.pointerEvents = 'none';
            drawBtn.textContent = 'Ubah Tanda Area';
            drawBtn.style.color = '';
            
            // Re-analyze when done drawing
            if (currentFile) {
                analyzeImage(currentFile);
            }
        }
    });
}

if (clearRoiBtn) {
    clearRoiBtn.addEventListener('click', () => {
        polygonPoints = [];
        ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        clearRoiBtn.classList.add('hidden');
        if (!isDrawingMode && currentFile) {
            analyzeImage(currentFile); // Re-analyze without mask
        }
    });
}

if (roiCanvas) {
    roiCanvas.addEventListener('mousedown', (e) => {
        if (!isDrawingMode) return;
        
        const rect = roiCanvas.getBoundingClientRect();
        const scaleX = roiCanvas.width / rect.width;
        const scaleY = roiCanvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        polygonPoints.push({x, y});
        drawPolygon();
        
        if (polygonPoints.length > 2) {
            clearRoiBtn.classList.remove('hidden');
        }
    });
}

if (heatmapBtn) {
    heatmapBtn.addEventListener('click', () => {
        useHeatmap = !useHeatmap;
        const textSpan = heatmapBtn.querySelector('.btn-text');
        const iconSpan = heatmapBtn.querySelector('.btn-icon');
        
        if (useHeatmap) {
            textSpan.textContent = 'Heatmap Aktif';
            textSpan.style.color = 'var(--accent-warning)';
            iconSpan.style.background = 'var(--accent-warning)';
            iconSpan.style.color = '#fff';
        } else {
            textSpan.textContent = 'Mode Heatmap';
            textSpan.style.color = '';
            iconSpan.style.background = '';
            iconSpan.style.color = '';
        }
        
        if (!isDrawingMode && currentFile) {
            analyzeImage(currentFile);
        }
    });
}

// Calibration Logic
if (calibrateBtn) {
    calibrateBtn.addEventListener('click', () => {
        isCalibrationMode = !isCalibrationMode;
        isDrawingMode = false; // Turn off ROI drawing
        drawBtn.textContent = 'Tandai Objek';
        drawBtn.style.color = '';
        
        const textSpan = calibrateBtn.querySelector('.btn-text');
        const iconSpan = calibrateBtn.querySelector('.btn-icon');
        
        if (isCalibrationMode) {
            textSpan.textContent = 'ArUco Aktif';
            textSpan.style.color = 'var(--accent-warning)';
            iconSpan.style.background = 'var(--accent-warning)';
            iconSpan.style.color = '#fff';
            calibInputBox.classList.remove('hidden');
        } else {
            textSpan.textContent = 'Kalibrasi ArUco';
            textSpan.style.color = '';
            iconSpan.style.background = '';
            iconSpan.style.color = '';
            calibInputBox.classList.add('hidden');
            arucoSizeCm = 0; // reset
            if (currentFile) analyzeImage(currentFile); // re-analyze without calibration
        }
    });
}

if (calibCancelBtn) {
    calibCancelBtn.addEventListener('click', () => {
        calibInputBox.classList.add('hidden');
        if (isCalibrationMode) calibrateBtn.click();
    });
}

if (calibSaveBtn) {
    calibSaveBtn.addEventListener('click', () => {
        const val = parseFloat(calibRealLengthInput.value);
        if (isNaN(val) || val <= 0) {
            alert('Masukkan ukuran sisi ArUco yang valid (contoh: 5)!');
            return;
        }
        arucoSizeCm = val;
        calibInputBox.classList.add('hidden');
        
        if (currentFile) {
            analyzeImage(currentFile);
        }
    });
}

function drawPolygon() {
    if (!ctx || polygonPoints.length === 0) return;
    
    ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
    
    ctx.beginPath();
    ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
    
    for (let i = 1; i < polygonPoints.length; i++) {
        ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
    }
    
    // Draw connecting lines
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw dots
    ctx.fillStyle = '#ff0000';
    polygonPoints.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
    });
    
    // If not drawing mode, fill the polygon
    if (!isDrawingMode && polygonPoints.length > 2) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        ctx.fill();
    } else if (polygonPoints.length > 2) {
        // Still draw a faint fill while drawing
        ctx.closePath();
        ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
        ctx.fill();
    }
}
