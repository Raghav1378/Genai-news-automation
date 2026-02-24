/*  ═══════════════════════════════════════════════════════════
    NeuralPulse v3 — Client-Side Application Logic
    Features: Search, Fact-Check, Trends, Voice, Memory,
              PDF Export, Email Briefings, Cross-Lingual
    ═══════════════════════════════════════════════════════════ */

const API = '';

// ── API Key Gate ─────────────────────────────────────── */
function getStoredKeys() {
    return {
        groq: sessionStorage.getItem('np-groq-key') || '',
        gemini: sessionStorage.getItem('np-gemini-key') || '',
    };
}

function hasValidKeys() {
    const keys = getStoredKeys();
    return !!(keys.groq || keys.gemini);
}

async function validateAndEnter() {
    const groqKey = document.getElementById('gateGroqKey').value.trim();
    const geminiKey = document.getElementById('gateGeminiKey').value.trim();
    const status = document.getElementById('keyStatus');
    const btn = document.getElementById('keySubmitBtn');
    const btnText = document.getElementById('keySubmitText');

    if (!groqKey && !geminiKey) {
        status.className = 'key-status error';
        status.textContent = '⚠️ Enter at least one API key';
        return;
    }

    btn.disabled = true;
    btnText.textContent = '⏳ Validating keys...';
    status.className = 'key-status loading';
    status.textContent = 'Testing your API keys...';

    try {
        const res = await fetch(`${API}/api/validate-keys`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ groq_key: groqKey, gemini_key: geminiKey }),
        });
        const data = await res.json();

        if (data.valid) {
            // Store keys in sessionStorage
            if (groqKey) sessionStorage.setItem('np-groq-key', groqKey);
            if (geminiKey) sessionStorage.setItem('np-gemini-key', geminiKey);

            const parts = [];
            if (data.groq) parts.push('✅ Groq');
            if (data.gemini) parts.push('✅ Gemini');
            if (!data.groq && groqKey) parts.push('❌ Groq (invalid)');
            if (!data.gemini && geminiKey) parts.push('❌ Gemini (invalid)');
            status.className = 'key-status success';
            status.textContent = parts.join('  ·  ') + '  — Entering...';

            setTimeout(() => {
                document.getElementById('keyGate').classList.add('hidden');
                initApp();
            }, 800);
        } else {
            status.className = 'key-status error';
            let msg = '❌ Keys are invalid. ';
            if (data.groq_error) msg += `Groq: ${data.groq_error}. `;
            if (data.gemini_error) msg += `Gemini: ${data.gemini_error}`;
            status.textContent = msg;
        }
    } catch (err) {
        status.className = 'key-status error';
        status.textContent = '❌ Server unreachable: ' + err.message;
    }
    btn.disabled = false;
    btnText.textContent = '🔓 Enter System';
}

/**
 * Wrapper around fetch() that auto-injects API key headers.
 * Use this for ALL /api/* calls.
 */
function apiFetch(url, options = {}) {
    const keys = getStoredKeys();
    const headers = { ...(options.headers || {}) };
    if (keys.groq) headers['X-Groq-Key'] = keys.groq;
    if (keys.gemini) headers['X-Gemini-Key'] = keys.gemini;
    return fetch(url, { ...options, headers });
}

// ── DOM ──────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);

const sidebar = $('sidebar'), sidebarToggle = $('sidebarToggle'), mobileMenuBtn = $('mobileMenuBtn');
const newSearchBtn = $('newSearchBtn'), clearHistoryBtn = $('clearHistoryBtn');
const historyList = $('historyList'), historyEmpty = $('historyEmpty');
const heroState = $('heroState'), loadingState = $('loadingState'), resultState = $('resultState');
const searchInput = $('searchInput'), searchBtn = $('searchBtn');
const deepResearchToggle = $('deepResearchToggle');
const languageSelect = $('languageSelect');

// Result elements
const resultQuery = $('resultQuery');
const backBtn = $('backBtn');
const ttsBtn = $('ttsBtn'), pdfBtn = $('pdfBtn');
const voiceBriefingBtn = $('voiceBriefingBtn');
const gaugeFill = $('gaugeFill'), gaugeValue = $('gaugeValue'), confidenceLabel = $('confidenceLabel');
const sentimentIcon = $('sentimentIcon'), sentimentLabel = $('sentimentLabel'), sentimentBar = $('sentimentBar');
const biasIcon = $('biasIcon'), biasLabel = $('biasLabel'), biasBar = $('biasBar');
const analysisBody = $('analysisBody');
const entitiesGrid = $('entitiesGrid');
const groqBody = $('groqBody'), geminiBody = $('geminiBody');
const groqConf = $('groqConf'), geminiConf = $('geminiConf');
const sourcesGrid = $('sourcesGrid'), sourceCount = $('sourceCount');
const factcheckGrid = $('factcheckGrid');
const biasBalanceSection = $('biasBalanceSection'), biasBalanceBody = $('biasBalanceBody');

// Loading steps
const step1 = $('step1'), step2 = $('step2'), step3 = $('step3'), step4 = $('step4');

// Settings
const settingsBtn = $('settingsBtn'), settingsModal = $('settingsModal'), closeSettingsBtn = $('closeSettingsBtn');
const blacklistInput = $('blacklistInput'), addBlacklistBtn = $('addBlacklistBtn'), blacklistList = $('blacklistList');

// Watchlist
const addWatchBtn = $('addWatchBtn'), watchlistItems = $('watchlistItems'), watchlistEmpty = $('watchlistEmpty');
const generateBriefingBtn = $('generateBriefingBtn');

// Memory
const memoryInput = $('memoryInput'), memorySearchBtn = $('memorySearchBtn'), memoryResults = $('memoryResults');

// Voice player
const voicePlayer = $('voicePlayer'), voicePlayBtn = $('voicePlayBtn');
const voiceText = $('voiceText'), voiceCloseBtn = $('voiceCloseBtn');

// Briefing modal
const briefingModal = $('briefingModal'), closeBriefingBtn = $('closeBriefingBtn');
const briefingPreview = $('briefingPreview'), sendBriefingBtn = $('sendBriefingBtn');

// Trends
const trendsChart = $('trendsChart'), trendsHint = $('trendsHint');

let currentResult = null;
let currentSearchId = null;
let isSpeaking = false;
let trendChartInstance = null;
let briefingHtml = '';

// ── Init ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    // Check if keys already exist in session
    if (hasValidKeys()) {
        document.getElementById('keyGate').classList.add('hidden');
        initApp();
    }
    // Gate inputs: allow Enter to submit
    const gateInputs = document.querySelectorAll('#keyGate input');
    gateInputs.forEach(inp => inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') validateAndEnter(); }));
});

function initApp() {
    loadHistory();
    loadWatchlist();
    loadLanguages();
    setupEvents();
}

function setupEvents() {
    // Sidebar
    sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.remove('collapsed');
        sidebar.classList.toggle('open');
    });
    newSearchBtn.addEventListener('click', () => { showHero(); stopTTS(); });
    backBtn.addEventListener('click', () => { showHero(); stopTTS(); });

    // Search
    searchBtn.addEventListener('click', runSearch);
    searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(); });
    document.querySelectorAll('.tag').forEach(tag => {
        tag.addEventListener('click', () => {
            searchInput.value = tag.dataset.query;
            runSearch();
        });
    });

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            const target = $(`tab-${tab.dataset.tab}`);
            if (target) target.classList.add('active');

            // Load trends on tab click
            if (tab.dataset.tab === 'trends') loadTrends();
        });
    });

    // Actions
    ttsBtn.addEventListener('click', toggleTTS);
    pdfBtn.addEventListener('click', downloadPDF);
    voiceBriefingBtn.addEventListener('click', generateVoiceBriefing);
    voicePlayBtn.addEventListener('click', playVoiceBriefing);
    voiceCloseBtn.addEventListener('click', () => { voicePlayer.classList.add('hidden'); stopTTS(); });

    // Settings
    settingsBtn.addEventListener('click', () => { settingsModal.classList.remove('hidden'); loadBlacklist(); loadEmailSettings(); });
    closeSettingsBtn.addEventListener('click', () => settingsModal.classList.add('hidden'));
    settingsModal.addEventListener('click', (e) => { if (e.target === settingsModal) settingsModal.classList.add('hidden'); });
    addBlacklistBtn.addEventListener('click', addDomainToBlacklist);

    // Email settings
    const saveEmailBtn = $('saveEmailSettingsBtn');
    if (saveEmailBtn) saveEmailBtn.addEventListener('click', saveEmailSettings);

    // Watchlist
    addWatchBtn.addEventListener('click', () => {
        const topic = prompt('Enter a topic to watch:');
        if (topic) addWatchTopic(topic);
    });

    // Briefing
    generateBriefingBtn.addEventListener('click', generateBriefing);
    closeBriefingBtn.addEventListener('click', () => briefingModal.classList.add('hidden'));
    sendBriefingBtn.addEventListener('click', sendEmailBriefing);

    // Memory
    memorySearchBtn.addEventListener('click', searchMemory);
    memoryInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') searchMemory(); });

    // Theme toggle
    const themeToggleBtn = $('themeToggleBtn');
    if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);

    // History
    clearHistoryBtn.addEventListener('click', async () => {
        if (!confirm('Clear all history?')) return;
        await apiFetch(`${API}/api/history`, { method: 'DELETE' });
        loadHistory();
    });
}

// ── State Transitions ────────────────────────────────── */
function showHero() { heroState.classList.remove('hidden'); loadingState.classList.add('hidden'); resultState.classList.add('hidden'); }
function showLoading() { heroState.classList.add('hidden'); loadingState.classList.remove('hidden'); resultState.classList.add('hidden'); }
function showResult() { heroState.classList.add('hidden'); loadingState.classList.add('hidden'); resultState.classList.remove('hidden'); }

function advanceStep(n) {
    [step1, step2, step3, step4].forEach((s, i) => {
        s.classList.toggle('active', i < n);
        s.classList.toggle('done', i < n - 1);
    });
}

// ── Search ───────────────────────────────────────────── */
async function runSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    const deep = deepResearchToggle.checked;
    const lang = languageSelect ? languageSelect.value : 'en';

    showLoading();
    advanceStep(1);

    try {
        setTimeout(() => advanceStep(2), 1500);
        setTimeout(() => advanceStep(3), 4000);
        setTimeout(() => advanceStep(4), 7000);

        const res = await apiFetch(`${API}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, deep_research: deep, language: lang }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        currentResult = data;
        currentSearchId = data.id;
        displayResult(data);
        loadHistory();
    } catch (err) {
        alert('Search failed: ' + err.message);
        showHero();
    }
}

// ── Display ──────────────────────────────────────────── */
function displayResult(data) {
    showResult();
    resultQuery.textContent = data.query;
    setConfidence(data.confidence);
    setSentiment(data.sentiment, data.sentiment_score);
    setBias(data.bias, data.bias_score);

    // Analysis
    analysisBody.innerHTML = renderMarkdown(data.analysis);

    // Bias balance
    if (data.bias_balance && data.bias_balance.length > 5) {
        biasBalanceSection.classList.remove('hidden');
        biasBalanceBody.innerHTML = `<p>${esc(data.bias_balance)}</p>`;
    } else {
        biasBalanceSection.classList.add('hidden');
    }

    // Entities
    renderEntities(data.entities);

    // Fact check
    renderFactCheck(data.fact_check_results || []);

    // Model comparison
    renderComparison(data);

    // Sources
    renderSources(data.sources);

    // Reset voice player
    voicePlayer.classList.add('hidden');

    updateActiveHistoryItem();
}

// ── Confidence ───────────────────────────────────────── */
function setConfidence(score) {
    const pct = Math.min(100, Math.max(0, score));
    const circumference = 2 * Math.PI * 52;
    gaugeFill.style.strokeDasharray = circumference;
    gaugeFill.style.strokeDashoffset = circumference - (pct / 100) * circumference;
    gaugeValue.textContent = Math.round(pct);
    const color = pct >= 70 ? 'var(--emerald)' : pct >= 40 ? 'var(--amber)' : 'var(--rose)';
    gaugeFill.style.stroke = color;
    confidenceLabel.textContent = pct >= 70 ? 'High confidence' : pct >= 40 ? 'Moderate' : 'Low confidence';
}

// ── Sentiment ────────────────────────────────────────── */
function setSentiment(sentiment, score) {
    const icons = { positive: '😊', negative: '😟', neutral: '😐', mixed: '🤔' };
    sentimentIcon.textContent = icons[sentiment] || '😐';
    sentimentLabel.textContent = sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
    sentimentBar.style.width = `${score || 50}%`;
    const colors = { positive: 'var(--emerald)', negative: 'var(--rose)', neutral: 'var(--slate-400)', mixed: 'var(--amber)' };
    sentimentBar.style.background = colors[sentiment] || 'var(--slate-400)';
}

// ── Bias ─────────────────────────────────────────────── */
function setBias(bias, score) {
    const icons = { left: '⬅️', right: '➡️', center: '⚖️', corporate: '🏢', mixed: '🔀' };
    biasIcon.textContent = icons[bias] || '⚖️';
    biasLabel.textContent = bias.charAt(0).toUpperCase() + bias.slice(1);
    biasBar.style.width = `${Math.min(score || 0, 100)}%`;
    biasBar.style.background = score > 50 ? 'var(--rose)' : score > 25 ? 'var(--amber)' : 'var(--emerald)';
}

// ── Entities ─────────────────────────────────────────── */
function renderEntities(entities) {
    if (!entities || !entities.length) {
        entitiesGrid.innerHTML = '<p class="empty-text">No entities extracted.</p>';
        return;
    }
    const typeColors = { person: '#6366f1', org: '#10b981', location: '#f59e0b', technology: '#3b82f6' };
    entitiesGrid.innerHTML = entities.map(e => `
        <div class="entity-card">
            <div class="entity-type" style="background:${typeColors[e.type] || '#64748b'}20;color:${typeColors[e.type] || '#64748b'}">${e.type}</div>
            <div class="entity-name">${esc(e.name)}</div>
            <div class="entity-role">${esc(e.role || '')}</div>
        </div>
    `).join('');
}

// ── Fact Check ───────────────────────────────────────── */
function renderFactCheck(claims) {
    if (!claims || !claims.length) {
        factcheckGrid.innerHTML = '<p class="empty-text">No fact-check results available.</p>';
        return;
    }
    const statusConfig = {
        verified: { icon: '✅', color: '#10b981', label: 'Verified' },
        unconfirmed: { icon: '⚠️', color: '#f59e0b', label: 'Unconfirmed' },
        contradictory: { icon: '❌', color: '#ef4444', label: 'Contradictory' },
    };
    factcheckGrid.innerHTML = claims.map(c => {
        const cfg = statusConfig[c.status] || statusConfig.unconfirmed;
        return `
            <div class="factcheck-card">
                <div class="fc-status" style="background:${cfg.color}15;color:${cfg.color};border:1px solid ${cfg.color}30">
                    ${cfg.icon} ${cfg.label}
                </div>
                <div class="fc-claim">${esc(c.claim)}</div>
                <div class="fc-evidence">${esc(c.evidence || '')}</div>
            </div>
        `;
    }).join('');
}

// ── Model Comparison ─────────────────────────────────── */
function renderComparison(data) {
    groqBody.innerHTML = renderMarkdown(data.analysis);
    groqConf.textContent = `${Math.round(data.confidence)}%`;
    geminiBody.innerHTML = data.gemini_analysis ? renderMarkdown(data.gemini_analysis) : '<p class="empty-text">Gemini analysis unavailable.</p>';
    geminiConf.textContent = data.gemini_confidence ? `${Math.round(data.gemini_confidence)}%` : '—%';
}

// ── Sources ──────────────────────────────────────────── */
function renderSources(sources) {
    sourceCount.textContent = sources ? sources.length : 0;
    if (!sources || !sources.length) {
        sourcesGrid.innerHTML = '<p class="empty-text">No sources available.</p>';
        return;
    }
    sourcesGrid.innerHTML = sources.map((s, i) => `
        <a class="source-card" href="${s.url}" target="_blank" rel="noopener">
            <div class="source-rank">#${i + 1}</div>
            <div class="source-info">
                <div class="source-title">${esc(s.title)}</div>
                <div class="source-url">${extractDomain(s.url)}</div>
            </div>
            <div class="source-score">${(s.score * 100).toFixed(0)}%</div>
        </a>
    `).join('');
}

// ── TTS (Text-to-Speech) ─────────────────────────────── */
function toggleTTS() {
    if (isSpeaking) { stopTTS(); return; }
    const text = analysisBody.innerText;
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 3000));
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.onend = () => { isSpeaking = false; ttsBtn.querySelector('span').textContent = 'Read'; };
    speechSynthesis.speak(utterance);
    isSpeaking = true;
    ttsBtn.querySelector('span').textContent = 'Stop';
}
function stopTTS() { speechSynthesis.cancel(); isSpeaking = false; if (ttsBtn) ttsBtn.querySelector('span').textContent = 'Read'; }

// ── PDF Export (Server-side) ─────────────────────────── */
async function downloadPDF() {
    if (!currentSearchId) return;
    pdfBtn.querySelector('span').textContent = '...';
    try {
        const res = await apiFetch(`${API}/api/export/pdf/${currentSearchId}`);
        if (!res.ok) throw new Error('PDF generation failed');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `NeuralPulse_Report_${currentSearchId}.pdf`;
        link.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        alert('PDF export failed: ' + err.message);
    }
    pdfBtn.querySelector('span').textContent = 'PDF';
}

// ── Voice Briefing ───────────────────────────────────── */
let voiceScript = '';
async function generateVoiceBriefing() {
    if (!currentSearchId) return;
    voicePlayer.classList.remove('hidden');
    voiceText.textContent = 'Generating briefing script…';
    voicePlayBtn.textContent = '⏳';
    try {
        const res = await apiFetch(`${API}/api/voice/briefing/${currentSearchId}`, { method: 'POST' });
        const data = await res.json();
        voiceScript = data.script || '';
        voiceText.textContent = voiceScript.slice(0, 200) + (voiceScript.length > 200 ? '…' : '');
        voicePlayBtn.textContent = '▶';
    } catch (err) {
        voiceText.textContent = 'Failed to generate briefing.';
        voicePlayBtn.textContent = '▶';
    }
}

function playVoiceBriefing() {
    if (!voiceScript) return;
    if (isSpeaking) { stopTTS(); voicePlayBtn.textContent = '▶'; return; }
    const utterance = new SpeechSynthesisUtterance(voiceScript);
    utterance.rate = 1.05;
    utterance.pitch = 1;
    utterance.onend = () => { isSpeaking = false; voicePlayBtn.textContent = '▶'; };
    speechSynthesis.speak(utterance);
    isSpeaking = true;
    voicePlayBtn.textContent = '⏸';
}

// ── Trends Chart ─────────────────────────────────────── */
async function loadTrends() {
    try {
        const res = await apiFetch(`${API}/api/trends`);
        const data = await res.json();
        if (!data.count || data.count < 2) {
            trendsHint.textContent = 'Run multiple searches to see trends over time.';
            return;
        }
        trendsHint.textContent = '';
        renderTrendChart(data);
    } catch (err) {
        trendsHint.textContent = 'Failed to load trends.';
    }
}

function renderTrendChart(data) {
    if (trendChartInstance) trendChartInstance.destroy();
    const ctx = trendsChart.getContext('2d');
    const labels = data.dates.map(d => {
        const dt = new Date(d);
        return `${dt.getMonth() + 1}/${dt.getDate()} ${dt.getHours()}:${String(dt.getMinutes()).padStart(2, '0')}`;
    });

    trendChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Confidence',
                    data: data.confidences,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99,102,241,0.1)',
                    tension: 0.4,
                    fill: true,
                },
                {
                    label: 'Sentiment',
                    data: data.sentiment_scores,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16,185,129,0.1)',
                    tension: 0.4,
                    fill: true,
                },
                {
                    label: 'Bias',
                    data: data.bias_scores,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245,158,11,0.1)',
                    tension: 0.4,
                    fill: true,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } },
                tooltip: {
                    callbacks: {
                        afterLabel: (ctx) => data.queries[ctx.dataIndex] ? `Query: ${data.queries[ctx.dataIndex].slice(0, 50)}` : '',
                    },
                },
            },
            scales: {
                x: { ticks: { color: '#64748b', maxRotation: 45 }, grid: { color: '#1e293b' } },
                y: { min: 0, max: 100, ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
            },
        },
    });
}

// ── Memory Search ────────────────────────────────────── */
async function searchMemory() {
    const q = memoryInput.value.trim();
    if (!q) return;
    memoryResults.innerHTML = '<p class="loading-text">Searching…</p>';
    try {
        const res = await apiFetch(`${API}/api/memory/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: q, top_k: 5 }),
        });
        const data = await res.json();
        if (data.error) {
            memoryResults.innerHTML = `<p class="empty-text">${esc(data.error)}</p>`;
            return;
        }
        if (!data.results || !data.results.length) {
            memoryResults.innerHTML = '<p class="empty-text">No matches found.</p>';
            return;
        }
        memoryResults.innerHTML = data.results.map(r => `
            <div class="memory-item" onclick="loadHistoryItem(${r.search_id})">
                <div class="memory-query">${esc(r.query)}</div>
                <div class="memory-meta">
                    <span class="memory-sim">${(r.similarity * 100).toFixed(0)}% match</span>
                    <span class="memory-sent">${r.sentiment}</span>
                </div>
            </div>
        `).join('');
    } catch (err) {
        memoryResults.innerHTML = '<p class="empty-text">Memory unavailable.</p>';
    }
}

// ── Email Briefing ───────────────────────────────────── */
async function generateBriefing() {
    generateBriefingBtn.textContent = '⏳ Generating…';
    try {
        const res = await apiFetch(`${API}/api/briefing/generate`, { method: 'POST' });
        const data = await res.json();
        if (data.html) {
            briefingHtml = data.html;
            briefingPreview.innerHTML = `<iframe srcdoc="${data.html.replace(/"/g, '&quot;')}" style="width:100%;height:500px;border:none;border-radius:8px;"></iframe>`;
            briefingModal.classList.remove('hidden');
        }
    } catch (err) {
        alert('Briefing generation failed: ' + err.message);
    }
    generateBriefingBtn.textContent = '📧 Generate Briefing';
}

async function sendEmailBriefing() {
    sendBriefingBtn.textContent = '📤 Sending…';
    try {
        const res = await apiFetch(`${API}/api/briefing/generate?send_email=true`, { method: 'POST' });
        const data = await res.json();
        if (data.email_status?.success) {
            alert('✅ Briefing sent successfully!');
        } else {
            alert('⚠️ ' + (data.email_status?.error || data.email_status || 'Email not configured'));
        }
    } catch (err) {
        alert('Send failed: ' + err.message);
    }
    sendBriefingBtn.textContent = '📤 Send Email';
}

// ── Email Settings ───────────────────────────────────── */
async function loadEmailSettings() {
    try {
        const res = await apiFetch(`${API}/api/email-settings`);
        const data = await res.json();
        $('smtpHost').value = data.smtp_host || 'smtp.gmail.com';
        $('smtpPort').value = data.smtp_port || 587;
        $('senderEmail').value = data.email || '';
        $('appPassword').value = '';
        $('recipientEmail').value = data.recipient || '';
        $('emailActive').checked = data.is_active || false;
        if (data.app_password && data.app_password !== '') {
            $('appPassword').placeholder = '••••••••  (saved)';
        }
    } catch (err) { /* ignore */ }
}

async function saveEmailSettings() {
    const body = {
        smtp_host: $('smtpHost').value,
        smtp_port: parseInt($('smtpPort').value) || 587,
        email: $('senderEmail').value,
        app_password: $('appPassword').value || '••••••••',
        recipient: $('recipientEmail').value,
        is_active: $('emailActive').checked,
    };
    try {
        await apiFetch(`${API}/api/email-settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        alert('✅ Email settings saved!');
    } catch (err) {
        alert('Failed to save: ' + err.message);
    }
}

// ── Languages ────────────────────────────────────────── */
async function loadLanguages() {
    try {
        const res = await apiFetch(`${API}/api/languages`);
        const langs = await res.json();
        const flags = { en: '🌐', ja: '🇯🇵', zh: '🇨🇳', ar: '🇸🇦', es: '🇪🇸', fr: '🇫🇷', de: '🇩🇪', hi: '🇮🇳', ko: '🇰🇷', pt: '🇧🇷', ru: '🇷🇺' };
        languageSelect.innerHTML = '';
        for (const [code, name] of Object.entries(langs)) {
            const opt = document.createElement('option');
            opt.value = code;
            opt.textContent = `${flags[code] || '🌐'} ${code.toUpperCase()}`;
            opt.title = name;
            languageSelect.appendChild(opt);
        }
    } catch (err) { /* default EN stays */ }
}

// ── History ──────────────────────────────────────────── */
async function loadHistory() {
    try {
        const res = await apiFetch(`${API}/api/history`);
        const items = await res.json();
        renderHistory(items);
    } catch (err) { /* ignore */ }
}

function renderHistory(items) {
    if (!items || !items.length) {
        historyEmpty.classList.remove('hidden');
        historyList.querySelectorAll('.history-item').forEach(el => el.remove());
        return;
    }
    historyEmpty.classList.add('hidden');
    const existing = historyList.querySelectorAll('.history-item');
    existing.forEach(el => el.remove());

    items.forEach(item => {
        const el = document.createElement('div');
        el.className = 'history-item';
        el.dataset.id = item.id;
        const time = new Date(item.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        const sentColors = { positive: '#10b981', negative: '#ef4444', neutral: '#64748b', mixed: '#f59e0b' };
        el.innerHTML = `
            <div class="hi-query">${esc(item.query)}</div>
            <div class="hi-meta">
                <span class="hi-conf">${Math.round(item.confidence)}%</span>
                <span class="hi-sent" style="color:${sentColors[item.sentiment] || '#64748b'}">${item.sentiment || 'neutral'}</span>
                <span class="hi-time">${time}</span>
            </div>
        `;
        el.addEventListener('click', () => loadHistoryItem(item.id));
        historyList.appendChild(el);
    });
}

async function loadHistoryItem(id) {
    try {
        const res = await apiFetch(`${API}/api/history/${id}`);
        const item = await res.json();
        currentResult = {
            ...item,
            analysis: item.result_summary,
        };
        currentSearchId = item.id;
        displayResult(currentResult);
    } catch (err) { /* ignore */ }
}

function updateActiveHistoryItem() {
    historyList.querySelectorAll('.history-item').forEach(el => {
        el.classList.toggle('active', currentSearchId && el.dataset.id == currentSearchId);
    });
}

// ── Blacklist ────────────────────────────────────────── */
async function loadBlacklist() {
    const res = await apiFetch(`${API}/api/blacklist`);
    const items = await res.json();
    blacklistList.innerHTML = items.map(d => `
        <div class="domain-item">
            <span>${esc(d.domain)}</span>
            <button onclick="removeBlacklist(${d.id})">✕</button>
        </div>
    `).join('') || '<p class="empty-text">No blacklisted domains</p>';
}
async function addDomainToBlacklist() {
    const domain = blacklistInput.value.trim();
    if (!domain) return;
    await apiFetch(`${API}/api/blacklist`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ domain }) });
    blacklistInput.value = '';
    loadBlacklist();
}
async function removeBlacklist(id) {
    await apiFetch(`${API}/api/blacklist/${id}`, { method: 'DELETE' });
    loadBlacklist();
}

// ── Watchlist ────────────────────────────────────────── */
async function loadWatchlist() {
    try {
        const res = await apiFetch(`${API}/api/watchlist`);
        const items = await res.json();
        if (!items.length) {
            watchlistEmpty.classList.remove('hidden');
            watchlistItems.querySelectorAll('.watch-item').forEach(el => el.remove());
            return;
        }
        watchlistEmpty.classList.add('hidden');
        watchlistItems.querySelectorAll('.watch-item').forEach(el => el.remove());
        items.forEach(item => {
            const el = document.createElement('div');
            el.className = 'watch-item';
            el.innerHTML = `
                <span class="watch-topic" onclick="searchInput.value='${esc(item.topic)}';runSearch();">${esc(item.topic)}</span>
                <button class="watch-del" onclick="deleteWatch(${item.id})">✕</button>
            `;
            watchlistItems.appendChild(el);
        });
    } catch (err) { /* ignore */ }
}

async function addWatchTopic(topic) {
    await apiFetch(`${API}/api/watchlist`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ topic }) });
    loadWatchlist();
}

async function deleteWatch(id) {
    await apiFetch(`${API}/api/watchlist/${id}`, { method: 'DELETE' });
    loadWatchlist();
}

// ── Utilities ────────────────────────────────────────── */
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
function extractDomain(u) { try { return new URL(u).hostname; } catch { return u; } }

function renderMarkdown(text) {
    if (!text) return '';
    // Simple markdown renderer
    let html = esc(text);
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Lists
    html = html.replace(/^[-•] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    // Line breaks
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br/>');
    // Wrap in paragraph
    html = '<p>' + html + '</p>';
    // Links
    html = html.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>');
    // Horizontal rules
    html = html.replace(/---/g, '<hr/>');
    return html;
}

// ── Theme Toggle ─────────────────────────────────────── */
function initTheme() {
    const saved = localStorage.getItem('neuralpulse-theme') || 'dark';
    applyTheme(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    applyTheme(next);
    localStorage.setItem('neuralpulse-theme', next);
}

function applyTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
    // Update button label
    const label = document.querySelector('.theme-label');
    if (label) label.textContent = theme === 'light' ? 'Dark Mode' : 'Light Mode';
}
