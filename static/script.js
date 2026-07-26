let currentTab = 'record'; 
let cameraRunning = true;
let cameraStream = null;

function switchTab(tabId, title) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');
    
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.getElementById(`${tabId}-tab`).classList.add('active');
    
    document.getElementById('page-title').innerText = title;
    currentTab = tabId;
    
    const workspace = document.querySelector('.workspace');
    if (tabId === 'text2sign') {
        workspace.classList.add('no-camera');
        cameraRunning = false;
        // FULLY stop camera to free microphone
        stopCamera();
    } else {
        workspace.classList.remove('no-camera');
        if (!cameraRunning) {
            cameraRunning = true;
            startCamera();
        }
    }
}

// ====== MEDIAPIPE LOGIC ======
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
let currentLandmarks = [];

function getNormalizedLandmarks(landmarks) {
    let data = [];
    const baseX = landmarks[0].x;
    const baseY = landmarks[0].y;
    let tempX = [], tempY = [];
    
    for (let lm of landmarks) {
        tempX.push(lm.x - baseX);
        tempY.push(lm.y - baseY);
    }
    
    let maxVal = 0;
    for(let i = 0; i < tempX.length; i++) {
        if(Math.abs(tempX[i]) > maxVal) maxVal = Math.abs(tempX[i]);
        if(Math.abs(tempY[i]) > maxVal) maxVal = Math.abs(tempY[i]);
    }
    if(maxVal === 0) maxVal = 1; 
    
    for(let i = 0; i < tempX.length; i++) {
        data.push(tempX[i] / maxVal);
        data.push(tempY[i] / maxVal);
    }
    return data;
}

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    canvasCtx.translate(canvasElement.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    let allLandmarks = [];
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#10b981', lineWidth: 4});
            drawLandmarks(canvasCtx, landmarks, {color: '#6366f1', lineWidth: 2, radius: 4});
            allLandmarks.push(...getNormalizedLandmarks(landmarks));
        }
    }
    canvasCtx.restore();
    
    while (allLandmarks.length < 84) { allLandmarks.push(0.0); }
    currentLandmarks = allLandmarks;

    if (currentTab === 'run' && results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        runPrediction(allLandmarks);
    }
}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
hands.onResults(onResults);

// ====== CAMERA START/STOP FUNCTIONS ======
function stopCamera() {
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
    .then(stream => {
        cameraStream = stream;
        videoElement.srcObject = stream;
        videoElement.play();
        
        function processFrame() {
            if (!cameraRunning) return;
            hands.send({ image: videoElement }).then(() => {
                requestAnimationFrame(processFrame);
            });
        }
        requestAnimationFrame(processFrame);
    })
    .catch(err => console.error("Camera error:", err));
}

// Start camera on page load
startCamera();


// ====== API LOGIC ======
document.getElementById('record-btn').addEventListener('click', async () => {
    const label = document.getElementById('label-input').value.trim();
    if (!label) return alert("Please enter a label first!");
    const msgEl = document.getElementById('record-status');
    try {
        const res = await fetch('/api/record', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ label: label, landmarks: currentLandmarks })
        });
        const data = await res.json();
        msgEl.innerText = data.message;
        setTimeout(() => msgEl.innerText = "", 2000);
    } catch(err) { console.error(err); }
});

document.getElementById('train-btn').addEventListener('click', async () => {
    const msgEl = document.getElementById('train-status');
    msgEl.innerText = "Training model... please wait.";
    msgEl.style.color = "var(--text-main)";
    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        msgEl.style.color = data.error ? "#ef4444" : "var(--accent)";
        msgEl.innerText = data.error || data.message;
    } catch(err) { console.error(err); }
});

let lastStable = "";
let stableCount = 0;
let currentPredictionLabel = "";

async function runPrediction(landmarks) {
    try {
        const res = await fetch('/api/predict', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ landmarks: landmarks })
        });
        const data = await res.json();
        const conf = data.confidence || 0;
        let pred = data.prediction || "";
        
        if (pred === lastStable && pred !== "") { stableCount++; } 
        else { stableCount = 0; }
        lastStable = pred;
        
        if (stableCount > 3) {
            currentPredictionLabel = pred;
            document.getElementById('current-prediction').innerText = pred;
            document.getElementById('confidence-fill').style.width = `${Math.min(conf * 100, 100)}%`;
        } else if (pred === "") {
             document.getElementById('current-prediction').innerText = "--";
             document.getElementById('confidence-fill').style.width = `0%`;
             currentPredictionLabel = "";
        }
    } catch(err) { console.error(err); }
}

// ====== SIGN TO TEXT BUILDER ======
let builtText = "";
const textDisplay = document.getElementById('built-text');
const updateText = () => textDisplay.innerText = builtText;
document.getElementById('add-btn').addEventListener('click', () => { if (currentPredictionLabel) { builtText += currentPredictionLabel; updateText(); }});
document.getElementById('space-btn').addEventListener('click', () => { builtText += " "; updateText(); });
document.getElementById('del-btn').addEventListener('click', () => { builtText = builtText.slice(0, -1); updateText(); });
document.getElementById('clear-btn').addEventListener('click', () => { builtText = ""; updateText(); });
document.getElementById('speak-btn').addEventListener('click', () => {
    if (builtText.trim() === "") return;
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(builtText)); 
});


// ====== LIVE F1 SCORE (FEEDBACK) ======
let sessionMetrics = { macro_f1: null, weighted_f1: null, n: 0 };

function updateF1Display() {
    const el = document.getElementById('f1-display');
    if (!el) return;
    if (!sessionMetrics.n) {
        el.innerText = "Session F1: -- (give feedback to start tracking)";
    } else {
        el.innerText = `Session F1: ${sessionMetrics.macro_f1.toFixed(2)} macro / ` +
                        `${sessionMetrics.weighted_f1.toFixed(2)} weighted (n=${sessionMetrics.n})`;
    }
}

async function sendFeedback(trueLabel, predictedLabel) {
    try {
        const res = await fetch('/api/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ true_label: trueLabel, predicted_label: predictedLabel })
        });
        const data = await res.json();
        if (data.error) { console.error(data.error); return; }
        sessionMetrics = { macro_f1: data.macro_f1, weighted_f1: data.weighted_f1, n: data.n };
        updateF1Display();
    } catch (err) { console.error(err); }
}

document.getElementById('feedback-correct-btn').addEventListener('click', () => {
    if (!currentPredictionLabel) {
        alert("No stable prediction to confirm yet — hold a sign until it shows up above.");
        return;
    }
    // Correct: what was predicted is exactly what was intended.
    sendFeedback(currentPredictionLabel, currentPredictionLabel);
});

document.getElementById('feedback-wrong-btn').addEventListener('click', () => {
    if (!currentPredictionLabel) {
        alert("No stable prediction to correct yet — hold a sign until it shows up above.");
        return;
    }
    const trueLabel = prompt(`The system predicted "${currentPredictionLabel}". What sign did you actually mean?`);
    if (!trueLabel || trueLabel.trim() === "") return;
    sendFeedback(trueLabel.trim(), currentPredictionLabel);
});

// Restore the running score from previous sessions (feedback_log.csv) on load.
(async () => {
    try {
        const res = await fetch('/api/feedback-stats');
        const data = await res.json();
        sessionMetrics = { macro_f1: data.macro_f1, weighted_f1: data.weighted_f1, n: data.n };
        updateF1Display();
    } catch (err) { console.error(err); }
})();


// ====== TEXT TO SIGN LOGIC (VIDEO PLAYLIST) ======
let videoQueue = [];
const videoPlayer = document.getElementById('sign-video-player');

function playNextVideo() {
    if(videoQueue.length === 0) return;
    const nextVid = videoQueue.shift();
    videoPlayer.src = nextVid; 
    videoPlayer.play();
}

videoPlayer.addEventListener('ended', playNextVideo);

document.getElementById('t2s-translate-btn').addEventListener('click', async () => {
    const text = document.getElementById('t2s-input').value;
    if(!text) return;
    
    const res = await fetch('/api/text-to-sign', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    });
    const data = await res.json();
    
    if(data.videos && data.videos.length > 0) {
        videoQueue = data.videos;
        playNextVideo();
    } else {
        alert("No videos found for this text!");
    }
});


// ====== FIXED: VOICE RECOGNITION (robust, auto-restarting) ======
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let micActive = false;        // true while a recognition session is actually running
let micShouldRun = false;     // true while the USER wants the mic on (drives auto-restart)
let restartAttempts = 0;
const MAX_RESTART_ATTEMPTS = 5;

function setMicStatus(text, isError) {
    const el = document.getElementById('mic-status');
    if (!el) return;
    el.innerText = text;
    el.style.color = isError ? "#ef4444" : "var(--text-muted)";
}

// --- Hard prerequisite checks, run immediately and surfaced on screen ---
if (!window.isSecureContext) {
    setMicStatus(
        "⚠ Mic/voice will NOT work: this page must be opened via " +
        "http://localhost:5000 or https://, not a network IP like " +
        "http://192.168.x.x. Change the address bar and reload.", true
    );
} else if (!SpeechRecognition) {
    setMicStatus("⚠ This browser doesn't support voice input. Use Chrome or Edge.", true);
} else if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setMicStatus("⚠ This browser has no microphone API available.", true);
}

// Request mic permission on page load so Chrome remembers it
if (window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        // Got permission, now release it immediately
        stream.getTracks().forEach(track => track.stop());
        console.log("✅ Microphone permission granted");
    })
    .catch(err => {
        console.warn("Mic permission not granted:", err);
        setMicStatus(`⚠ Microphone permission problem: ${err.name}. ` +
                     `Click the lock icon in the address bar → allow Microphone.`, true);
    });
}

function setMicUI(listening) {
    const btn = document.getElementById('mic-btn');
    if (listening) {
        btn.innerText = "🔴 Listening...";
        btn.style.background = "rgba(239, 68, 68, 0.3)";
    } else {
        btn.innerText = "🎙 Mic";
        btn.style.background = "rgba(99, 102, 241, 0.2)";
    }
}

function createRecognition() {
    const rec = new SpeechRecognition();
    rec.lang = 'en-US';
    // interimResults=true gives live partial text while you're still talking,
    // so you can SEE the mic is actually capturing audio instead of it
    // looking dead until (or unless) a final result arrives.
    rec.interimResults = true;
    rec.continuous = true;
    rec.maxAlternatives = 1;

    rec.onstart = () => {
        micActive = true;
        restartAttempts = 0;
        setMicUI(true);
        setMicStatus("Listening — speak now.", false);
        console.log("🎤 Mic started - speak now!");
    };

    rec.onaudiostart = () => {
        setMicStatus("Mic is capturing audio…", false);
    };

    rec.onspeechstart = () => {
        setMicStatus("Speech detected, transcribing…", false);
    };

    rec.onresult = (e) => {
        let finalTranscript = "";
        let interimTranscript = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
            const transcript = e.results[i][0].transcript;
            if (e.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        if (interimTranscript) {
            document.getElementById('t2s-input').value = interimTranscript;
        }
        if (finalTranscript) {
            console.log("✅ You said:", finalTranscript);
            document.getElementById('t2s-input').value = finalTranscript;
            micShouldRun = false; // got what we needed — stop listening cleanly
            rec.stop();
            setMicStatus(`Heard: "${finalTranscript}"`, false);
            document.getElementById('t2s-translate-btn').click();
        }
    };

    rec.onerror = (e) => {
        console.warn("Speech recognition error:", e.error);
        if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
            micShouldRun = false;
            setMicStatus("⚠ Microphone access denied. Click the lock icon in the " +
                         "address bar → allow Microphone, then click Mic again.", true);
        } else if (e.error === 'network') {
            setMicStatus("⚠ Network error — speech recognition needs an active " +
                         "internet connection (it streams audio to Google's servers). " +
                         "Retrying…", true);
        } else if (e.error === 'audio-capture') {
            setMicStatus("⚠ No microphone found, or it's in use by another app. " +
                         "Check your OS sound settings.", true);
        } else if (e.error === 'no-speech') {
            setMicStatus("No speech detected yet — still listening…", false);
        } else {
            setMicStatus(`⚠ Recognition error: ${e.error}. Retrying…`, true);
        }
        // 'no-speech', 'audio-capture', 'network', 'aborted' are transient —
        // onend (below) restarts automatically instead of giving up here.
    };

    rec.onend = () => {
        micActive = false;
        if (micShouldRun && restartAttempts < MAX_RESTART_ATTEMPTS) {
            restartAttempts++;
            // Small delay avoids a tight error loop if the mic/permissions
            // are genuinely unavailable, while still feeling instantaneous.
            setTimeout(() => {
                if (micShouldRun) {
                    try {
                        recognition.start();
                    } catch (err) {
                        console.error("Restart failed:", err);
                        setMicStatus(`⚠ Could not restart mic: ${err.message}`, true);
                    }
                }
            }, 250);
        } else {
            if (restartAttempts >= MAX_RESTART_ATTEMPTS) {
                setMicStatus("⚠ Gave up after repeated failures. Check your internet " +
                             "connection, OS microphone permissions, and that you're on " +
                             "http://localhost:5000 (not a LAN IP).", true);
            } else {
                setMicStatus("", false);
            }
            micShouldRun = false;
            setMicUI(false);
        }
    };

    return rec;
}

if (SpeechRecognition) {
    recognition = createRecognition();
}

document.getElementById('mic-btn').addEventListener('click', () => {
    if (!recognition) {
        setMicStatus("⚠ This browser doesn't support voice input. Use Chrome or Edge.", true);
        return;
    }
    if (!window.isSecureContext) {
        setMicStatus(
            "⚠ Mic will NOT work here: open this page via http://localhost:5000, " +
            "not a network IP.", true
        );
        return;
    }

    if (micShouldRun) {
        // User wants to turn the mic off.
        micShouldRun = false;
        recognition.stop();
        return;
    }

    micShouldRun = true;
    restartAttempts = 0;
    setMicStatus("Starting mic…", false);
    try {
        recognition.start();
    } catch (err) {
        // Chrome can throw InvalidStateError if the previous session didn't
        // fully tear down yet — recreate a fresh instance and retry once.
        recognition = createRecognition();
        setTimeout(() => {
            try { recognition.start(); } catch (e) {
                console.error(e);
                setMicStatus(`⚠ Could not start mic: ${e.message}`, true);
            }
        }, 300);
    }
});


// ====== VIDEO UPLOAD LOGIC ======
document.getElementById('video-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    let word = prompt("What word or phrase does this sign language video represent?\n(e.g., 'hello' or 'how are you')");
    
    if (!word || word.trim() === "") {
        alert("Upload cancelled. You must provide a word.");
        return;
    }
    
    word = word.trim().toLowerCase().replace(/ /g, "_");
    const fileName = word + ".mp4";
    
    const formData = new FormData();
    formData.append('video', file, fileName);
    
    try {
        const res = await fetch('/api/upload-video', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        alert(data.message);
    } catch(err) {
        console.error(err);
        alert("Upload failed. Make sure your server is running.");
    }
    
    e.target.value = '';
});


// ====== MAGIC SPARKLE CURSOR TRAIL (purely decorative, self-contained) ======
(function () {
    let lastSparkTime = 0;
    const SPARK_INTERVAL_MS = 45; // throttle so we don't flood the DOM

    document.addEventListener('mousemove', (e) => {
        const now = Date.now();
        if (now - lastSparkTime < SPARK_INTERVAL_MS) return;
        lastSparkTime = now;

        const spark = document.createElement('div');
        spark.className = 'spark';
        spark.style.left = `${e.clientX + (Math.random() * 10 - 5)}px`;
        spark.style.top = `${e.clientY + (Math.random() * 10 - 5)}px`;
        document.body.appendChild(spark);

        // Clean up after the CSS fade-out animation finishes.
        setTimeout(() => spark.remove(), 750);
    });
})();