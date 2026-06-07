/* ═══════════════════════════════════════════════════════════
   chat.js — Shaan Raza AI Chatbot Frontend Logic
   Handles: chat flow, booking, admin panel, evaluation display
   ═══════════════════════════════════════════════════════════ */

'use strict';

const API_BASE = (window.API_BASE_URL && window.API_BASE_URL !== "API_URL_PLACEHOLDER") ? window.API_BASE_URL : "https://web-production-11aa7.up.railway.app";

// ─── State ────────────────────────────────────────────────────
const STATE = {
  sessionId: generateUUID(),
  isTyping: false,
  messageCount: 0,
  availabilityCache: null,
  selectedSlot: { date: null, time: null },
  hasBookedInCurrentSession: false,
};

// ─── DOM refs ─────────────────────────────────────────────────
const $msg    = () => document.getElementById('messagesArea');
const $input  = () => document.getElementById('chatInput');
const $send   = () => document.getElementById('sendBtn');
const $status = () => document.getElementById('statusDot');

// ─────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initAutoResize();
  initKeyboardShortcuts();
  checkStatus();
  updateInterviewsCounter();
  setInterval(checkStatus, 15000);

  // Safety timeout: reveal chat after 15s even if status is not ready
  setTimeout(() => {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
      console.warn('Warming up timed out. Displaying chat interface anyway.');
      overlay.style.opacity = '0';
      overlay.style.visibility = 'hidden';
      setTimeout(() => overlay.remove(), 400);
      
      const dot = $status();
      if (dot && (dot.textContent === 'Loading KB…' || dot.className.includes('loading'))) {
        dot.textContent = 'Ready (Delayed)';
        dot.className = 'status-dot';
      }
    }
  }, 15000);

  document.getElementById('sendBtn').addEventListener('click', handleSend);
  document.getElementById('clearBtn').addEventListener('click', clearConversation);
  document.getElementById('bookCTABtn').addEventListener('click', () => {
    // Only open booking modal if not already booked in current session
    if (!STATE.hasBookedInCurrentSession) {
      openBookingModal();
    }
  });
  document.getElementById('adminBtn').addEventListener('click', openAdminModal);
  document.getElementById('evalBtn').addEventListener('click', openEvalModal);

  // Close modals on overlay click
  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', e => {
      if (e.target === overlay) closeModal(overlay.id);
    });
  });
});

// ─────────────────────────────────────────────────────────────
// Status Check
// ─────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch(API_BASE + '/api/status');
    const data = await res.json();
    const dot = $status();

    const isReady = data.ready || data.rag_loaded;
    if (isReady) {
      dot.textContent = data.llm_ready ? 'AI Ready' : 'RAG Ready (No LLM Key)';
      dot.className = 'status-dot';
      
      // Hide and remove loading overlay once ready
      const overlay = document.getElementById('loadingOverlay');
      if (overlay) {
        overlay.style.opacity = '0';
        overlay.style.visibility = 'hidden';
        setTimeout(() => overlay.remove(), 400);
      }
    } else {
      dot.textContent = 'Loading KB…';
      dot.className = 'status-dot loading';
      // Keep polling until ready
      setTimeout(checkStatus, 1500);
    }
  } catch {
    const dot = $status();
    dot.textContent = 'Offline';
    dot.className = 'status-dot offline';
    // Retry polling
    setTimeout(checkStatus, 2000);
  }
}

// ─────────────────────────────────────────────────────────────
// Chat Core
// ─────────────────────────────────────────────────────────────
async function handleSend() {
  const input = $input();
  const text = input.value.trim();
  if (!text || STATE.isTyping) return;

  // Hide welcome
  const welcome = document.getElementById('welcomeMsg');
  if (welcome) welcome.style.display = 'none';

  // Append user message
  appendMessage('user', text);
  input.value = '';
  input.style.height = 'auto';
  STATE.messageCount++;

  // Show typing indicator
  showTyping();

  try {
    const res = await fetch(API_BASE + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: STATE.sessionId }),
    });

    const data = await res.json();
    removeTyping();

    if (!res.ok) {
      appendMessage('bot', data.error || 'Something went wrong. Please try again.', [], true);
      return;
    }

    appendMessage('bot', data.response, data.sources || [], false, data.hallucination_flag);

    // Show booking CTA if intent detected
    if (data.booking_intent) {
      setTimeout(() => {
        appendBookingCTA();
      }, 400);
    }

  } catch (err) {
    removeTyping();
    appendMessage('bot', 'Connection error. Please make sure the server is running.', [], true);
  }
}

function sendSuggestedQuestion(question) {
  const welcome = document.getElementById('welcomeMsg');
  if (welcome) welcome.style.display = 'none';
  $input().value = question;
  handleSend();
}

// Global for onclick
window.sendSuggestedQuestion = sendSuggestedQuestion;

// ─────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────
// Message Rendering
// ─────────────────────────────────────────────────────────────
function getQuestionForSource(label, source) {
  const lbl = (label || '').toLowerCase();
  const src = (source || '').toLowerCase();
  
  if (lbl.includes('personal information') || lbl.includes('contact')) {
    return "Tell me about Shaan's background and how to contact him";
  }
  if (lbl.includes('role fit') || lbl.includes('why hire')) {
    return "Why is Shaan the right hire and what makes him stand out?";
  }
  if (lbl.includes('professional experience') || lbl.includes('internship') || lbl.includes('experience')) {
    return "Walk me through Shaan's work experience and internships";
  }
  if (lbl.includes('interview response') || lbl.includes('guidelines') || lbl.includes('values')) {
    return "How does Shaan typically approach interviews and what are his values?";
  }
  if (src === 'github' || lbl.startsWith('github')) {
    return "What projects has Shaan built on GitHub and what tech did he use?";
  }
  if (src === 'calendar' || lbl.includes('calendar') || lbl.includes('availability')) {
    return "What slots are available to book an interview with Shaan?";
  }
  
  return `Tell me more about ${label}`;
}

function appendMessage(role, text, sources = [], isError = false, hallFlag = false) {
  const area = $msg();
  const row = document.createElement('div');
  row.className = `message-row ${role}`;
  row.style.animationDelay = '0ms';

  const avatar = document.createElement('div');
  avatar.className = `msg-avatar ${role}`;
  avatar.textContent = role === 'bot' ? '🤖' : '👤';

  const content = document.createElement('div');
  content.className = 'message-content';

  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${role}${isError ? ' error' : ''}`;
  bubble.innerHTML = formatMessageText(text);

  content.appendChild(bubble);

  // Timestamp
  const time = document.createElement('div');
  time.className = 'message-time';
  time.textContent = formatTime(new Date());
  content.appendChild(time);

  // Hallucination warning
  if (hallFlag && role === 'bot') {
    const warn = document.createElement('div');
    warn.className = 'hallucination-warning';
    warn.innerHTML = '⚠️ Response flagged for potential inaccuracy — verify independently.';
    content.appendChild(warn);
  }

  // Sources
  if (sources && sources.length > 0 && role === 'bot') {
    const sourcesRow = document.createElement('div');
    sourcesRow.className = 'sources-row';

    sources.forEach(src => {
      const chip = document.createElement('button');
      chip.className = `source-chip ${src.source || ''}`;
      
      const question = getQuestionForSource(src.label || src.section, src.source);
      chip.title = `Click to ask: "${question}"`;
      chip.innerHTML = `${getSourceIcon(src.source)}${src.label || src.section}`;
      chip.setAttribute('aria-label', `Source: ${src.label}`);
      chip.style.cursor = 'pointer';
      
      chip.addEventListener('click', () => {
        sendSuggestedQuestion(question);
      });
      
      sourcesRow.appendChild(chip);
    });

    content.appendChild(sourcesRow);
  }

  // Latency disclaimer (bot only, skip on errors)
  if (role === 'bot' && !isError) {
    const disclaimer = document.createElement('div');
    disclaimer.className = 'response-disclaimer';
    disclaimer.innerHTML = '⏱ Running on free LLM models — sorry for the slow response!';
    content.appendChild(disclaimer);
  }

  row.appendChild(avatar);
  row.appendChild(content);
  area.appendChild(row);
  scrollToBottom();
}

function appendBookingCTA() {
  const area = $msg();
  const row = document.createElement('div');
  row.className = 'message-row bot';
  row.innerHTML = `
    <div class="msg-avatar bot">🤖</div>
    <div class="message-content">
      <div class="message-bubble bot" style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span>Ready to schedule? Click below to pick a slot 📅</span>
        <button onclick="openBookingModal()" style="background:var(--grad-gold);border:none;border-radius:8px;padding:8px 16px;color:#1a1000;font-weight:700;font-size:13px;cursor:pointer;font-family:var(--font-sans)">Book Interview →</button>
      </div>
    </div>
  `;
  area.appendChild(row);
  scrollToBottom();
}

function formatMessageText(text) {
  // Convert markdown-ish text to HTML
  let html = escapeHtml(text);

  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic: *text*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Code: `text`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bullet lists
  html = html.replace(/^[•\-\*]\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Numbered lists
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

  // Paragraphs
  html = html.split('\n\n').map(para => {
    const trimmed = para.trim();
    if (!trimmed) return '';
    if (trimmed.startsWith('<ul>') || trimmed.startsWith('<li>')) return trimmed;
    return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
  }).filter(Boolean).join('');

  return html || `<p>${escapeHtml(text)}</p>`;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function getSourceIcon(source) {
  const icons = {
    resume: '📄 ',
    github: '💻 ',
    calendar: '📅 ',
    project: '🔬 ',
  };
  return icons[source] || '📌 ';
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// ─────────────────────────────────────────────────────────────
// Typing Indicator
// ─────────────────────────────────────────────────────────────
function showTyping() {
  STATE.isTyping = true;
  $send().disabled = true;
  const area = $msg();
  const row = document.createElement('div');
  row.className = 'typing-row';
  row.id = 'typingIndicator';
  row.innerHTML = `
    <div class="msg-avatar bot">🤖</div>
    <div class="typing-bubble">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  area.appendChild(row);
  scrollToBottom();
}

function removeTyping() {
  STATE.isTyping = false;
  $send().disabled = false;
  const indicator = document.getElementById('typingIndicator');
  if (indicator) indicator.remove();
}

function scrollToBottom() {
  const area = $msg();
  requestAnimationFrame(() => {
    area.scrollTop = area.scrollHeight;
  });
}

// ─────────────────────────────────────────────────────────────
// Keyboard & Auto-Resize
// ─────────────────────────────────────────────────────────────
function initAutoResize() {
  const input = $input();
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });
}

function initKeyboardShortcuts() {
  $input().addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal-overlay.active').forEach(m => {
        closeModal(m.id);
      });
    }
  });
}

// ─────────────────────────────────────────────────────────────
// Clear Conversation
// ─────────────────────────────────────────────────────────────
async function clearConversation() {
  try {
    await fetch(API_BASE + '/api/session/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: STATE.sessionId }),
    });
  } catch {}

  STATE.sessionId = generateUUID();
  STATE.messageCount = 0;

  const area = $msg();
  area.innerHTML = `
    <div class="welcome-message" id="welcomeMsg">
      <div class="welcome-icon">🤖</div>
      <div class="welcome-title">Hi! I'm Shaan's AI Representative</div>
      <div class="welcome-subtitle">
        Ask me anything about Shaan's background, projects, skills, or experience — or schedule an interview. I'm grounded in his resume and GitHub data.
      </div>
      <div class="suggested-questions">
        <button class="suggest-btn" onclick="sendSuggestedQuestion('Tell me about Shaan Raza.')">Who is Shaan?</button>
        <button class="suggest-btn" onclick="sendSuggestedQuestion('Why should we hire Shaan?')">Why hire Shaan?</button>
        <button class="suggest-btn" onclick="sendSuggestedQuestion('What internships has Shaan completed?')">Internships</button>
        <button class="suggest-btn" onclick="sendSuggestedQuestion('Explain the FMCG Customer Churn Prediction project.')">ML Projects</button>
        <button class="suggest-btn" onclick="sendSuggestedQuestion('What technical skills does Shaan have?')">Skills</button>
        <button class="suggest-btn" onclick="sendSuggestedQuestion('When is Shaan available for an interview?')">Schedule Interview</button>
      </div>
    </div>
  `;
  showToast('Conversation cleared', 'success');
}

// ─────────────────────────────────────────────────────────────
// Booking Modal
// ─────────────────────────────────────────────────────────────
async function openBookingModal() {
  STATE.selectedSlot = { date: null, time: null };
  document.getElementById('bookingSuccess').style.display = 'none';
  document.getElementById('bookingError').style.display = 'none';
  document.getElementById('slotError').style.display = 'none';
  document.getElementById('bookingForm').reset();
  document.getElementById('selectedDate').value = '';
  document.getElementById('selectedTime').value = '';

  openModal('bookingModal');
  await loadAvailabilitySlots();
}

// Global for onclick
window.openBookingModal = openBookingModal;

async function loadAvailabilitySlots() {
  const picker = document.getElementById('slotPicker');
  picker.innerHTML = '<div class="no-slots-msg skeleton" style="height:60px"></div>';

  try {
    const res = await fetch(API_BASE + '/api/availability');
    const data = await res.json();

    if (!data.available || !data.dates || data.dates.length === 0) {
      picker.innerHTML = '<div class="no-slots-msg">No available slots at the moment. Please check back later.</div>';
      return;
    }

    STATE.availabilityCache = data.dates;
    renderSlotPicker(data.dates);
  } catch {
    picker.innerHTML = '<div class="no-slots-msg">Failed to load availability. Please refresh.</div>';
  }
}

function renderSlotPicker(dates) {
  const picker = document.getElementById('slotPicker');
  picker.innerHTML = '';

  dates.slice(0, 5).forEach(dateObj => {
    const group = document.createElement('div');
    group.className = 'slot-date-group';

    const dateLabel = document.createElement('div');
    dateLabel.className = 'slot-date-label';
    dateLabel.textContent = formatDateLabel(dateObj.date, dateObj.day);
    group.appendChild(dateLabel);

    const timesDiv = document.createElement('div');
    timesDiv.className = 'slot-times';

    dateObj.times.slice(0, 6).forEach(time => {
      const btn = document.createElement('button');
      btn.className = 'slot-time-btn';
      btn.type = 'button';
      btn.textContent = time;
      btn.dataset.date = dateObj.date;
      btn.dataset.time = time;
      btn.addEventListener('click', () => selectSlot(dateObj.date, time, btn));
      timesDiv.appendChild(btn);
    });

    group.appendChild(timesDiv);
    picker.appendChild(group);
  });
}

function selectSlot(date, time, btn) {
  // Deselect all
  document.querySelectorAll('.slot-time-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');

  STATE.selectedSlot = { date, time };
  document.getElementById('selectedDate').value = date;
  document.getElementById('selectedTime').value = time;
  document.getElementById('slotError').style.display = 'none';
}

function formatDateLabel(dateStr, dayName) {
  const d = new Date(dateStr + 'T00:00:00');
  return `${dayName}, ${d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}`;
}

async function confirmBooking() {
  const name  = document.getElementById('bookName').value.trim();
  const email = document.getElementById('bookEmail').value.trim();
  const phone = document.getElementById('bookPhone').value.trim();
  const date  = document.getElementById('selectedDate').value;
  const time  = document.getElementById('selectedTime').value;

  const errEl = document.getElementById('bookingError');
  const slotErr = document.getElementById('slotError');
  errEl.style.display = 'none';
  slotErr.style.display = 'none';

  // Validate
  const errors = [];
  if (!name) errors.push('Name is required.');
  if (!email || !email.includes('@')) errors.push('Valid email is required.');
  if (!date || !time) {
    slotErr.style.display = 'block';
    errors.push('');
  }

  if (errors.filter(Boolean).length) {
    errEl.textContent = errors.filter(Boolean).join(' ');
    errEl.style.display = 'block';
    return;
  }

  const btn = document.getElementById('confirmBookBtn');
  btn.disabled = true;
  btn.textContent = 'Booking…';

  try {
    const res = await fetch(API_BASE + '/api/book', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, phone, date, time }),
    });
    const data = await res.json();

    if (data.success) {
      const successEl = document.getElementById('bookingSuccess');
      if (successEl) {
        if (data.warning) {
          successEl.innerHTML = `✅ Interview booked in local database!<br><span style="color:#d97706; font-size: 0.9em; margin-top: 5px; display: block;">⚠️ Google Calendar Sync Warning: ${data.warning}</span>`;
        } else {
          successEl.innerHTML = `✅ Interview booked! You'll receive a confirmation shortly.`;
        }
      }
      if (successEl) successEl.style.display = 'block';
      document.getElementById('bookingForm').style.display = 'none';
      btn.style.display = 'none';

      // Add confirmation message to chat
      let msgText = `✅ You're all set! Your interview with Shaan is confirmed for ${date} at ${time} IST. A Google Meet link has been sent to your email. Looking forward to speaking with you!`;
      if (data.warning) {
        msgText += `\n\n⚠️ **Google Calendar Sync Warning:** ${data.warning}`;
      }
      appendMessage('bot',
        msgText,
        [{ label: 'Calendar/Availability', source: 'calendar', section: 'Booking Confirmed' }]
      );

      // Update session booking state and button
      STATE.hasBookedInCurrentSession = true;
      updateBookingButtonState(date, time);
      updateInterviewsCounter();

      showToast('Interview booked successfully! 🎉', 'success');
      setTimeout(() => closeModal('bookingModal'), 3000);
    } else {
      errEl.textContent = data.message || 'Booking failed. Please try again.';
      errEl.style.display = 'block';
    }
  } catch {
    errEl.textContent = 'Network error. Please try again.';
    errEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Confirm Booking';
  }
}

// Global for onclick
window.confirmBooking = confirmBooking;

// ─────────────────────────────────────────────────────────────
// Admin Modal
// ─────────────────────────────────────────────────────────────
async function openAdminModal() {
  openModal('adminModal');
  await loadAdminStatus();
  await loadBookings();
}

async function loadBookings() {
  const container = document.getElementById('adminBookingsRows');
  if (!container) return;
  container.innerHTML = '<div class="no-slots-msg">Loading bookings…</div>';
  try {
    const res = await fetch(API_BASE + '/api/contacts');
    const contacts = await res.json();
    if (!contacts || contacts.length === 0) {
      container.innerHTML = '<div class="no-slots-msg">No appointments booked yet.</div>';
      return;
    }
    
    // Sort logic: upcoming chronologically first, then past descending
    contacts.sort((a, b) => {
      const dateA = new Date(`${a.date}T${convertTo24h(a.time)}`);
      const dateB = new Date(`${b.date}T${convertTo24h(b.time)}`);
      const now = new Date();
      const isPastA = dateA < now;
      const isPastB = dateB < now;
      
      if (isPastA && !isPastB) return 1;  // Past goes down
      if (!isPastA && isPastB) return -1; // Upcoming goes up
      
      // Both upcoming: sort ascending (nearest first)
      if (!isPastA && !isPastB) return dateA - dateB;
      
      // Both past: sort descending (most recent past first)
      return dateB - dateA;
    });

    container.innerHTML = contacts.map(c => {
      const dateVal = new Date(`${c.date}T${convertTo24h(c.time)}`);
      const isPast = dateVal < new Date();
      const rowStyle = isPast ? 'opacity: 0.5; filter: grayscale(40%);' : '';
      const statusBadge = isPast 
        ? `<span style="font-size: 9px; padding: 2px 6px; background: rgba(255,255,255,0.08); color: var(--text-muted); border-radius: 4px; font-weight: 700; margin-left: 6px; text-transform: uppercase;">Past</span>`
        : `<span style="font-size: 9px; padding: 2px 6px; background: rgba(16, 185, 129, 0.12); color: var(--success); border-radius: 4px; font-weight: 700; margin-left: 6px; text-transform: uppercase;">Upcoming</span>`;

      const meetBtn = c.google_meet_link 
        ? `<a href="${c.google_meet_link}" target="_blank" class="quick-link" style="padding: 6px 10px; font-size: 11px; margin-top: 4px; display: inline-flex; align-items: center; justify-content: center; gap: 4px; border-color: rgba(6, 182, 212, 0.4); color: var(--accent);">
             <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" width="12" height="12">
               <path stroke-linecap="round" stroke-linejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9a2.25 2.25 0 0 0-2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
             </svg> Join Google Meet
           </a>` 
        : '';
        
      const eventBtn = c.google_event_link 
        ? `<a href="${c.google_event_link}" target="_blank" class="quick-link" style="padding: 6px 10px; font-size: 11px; margin-top: 4px; display: inline-flex; align-items: center; justify-content: center; gap: 4px; border-color: rgba(245, 158, 11, 0.4); color: var(--secondary);">
             <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" width="12" height="12">
               <path stroke-linecap="round" stroke-linejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5" />
             </svg> Event
           </a>` 
        : '';

      const createdTime = c.created_at ? new Date(c.created_at).toLocaleString() : 'N/A';

      return `
        <div class="config-row" style="flex-direction: column; align-items: stretch; gap: 6px; padding: 12px; margin-bottom: 4px; ${rowStyle}">
          <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
              <div style="font-weight: 700; color: var(--text-primary); font-size: 13px; display: flex; align-items: center;">
                ${escapeHtml(c.name)} ${statusBadge}
              </div>
              <div style="font-size: 11px; color: var(--text-secondary);">${escapeHtml(c.email)} ${c.phone ? '· ' + escapeHtml(c.phone) : ''}</div>
              <div style="font-size: 9px; color: var(--text-muted); margin-top: 2px;">Booked at: ${createdTime}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-weight: 700; color: var(--primary-light); font-size: 12px;">${c.time}</div>
              <div style="font-size: 10px; color: var(--text-muted);">${c.date}</div>
            </div>
          </div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;">
            ${meetBtn}
            ${eventBtn}
          </div>
        </div>
      `;
    }).join('');
  } catch (e) {
    container.innerHTML = '<div class="no-slots-msg" style="color:var(--danger)">Failed to load bookings.</div>';
  }
}

async function loadAdminStatus() {
  try {
    const [statusRes, configRes, ragRes] = await Promise.all([
      fetch(API_BASE + '/api/status'),
      fetch(API_BASE + '/api/config'),
      fetch(API_BASE + '/api/rag/stats'),
    ]);
    const status = await statusRes.json();
    const config = await configRes.json();
    const rag = await ragRes.json();

    const calInput = document.getElementById('calendarIdInput');
    if (calInput) {
      calInput.value = config.google_calendar_id || '';
    }

    const rows = document.getElementById('adminStatusRows');
    rows.innerHTML = [
      statusRow('RAG Knowledge Base', rag.is_loaded ? `✅ Loaded (${rag.total_chunks} chunks)` : '⏳ Loading…', rag.is_loaded),
      statusRow('NVIDIA NIM LLM', status.llm_ready ? '✅ Ready' : '❌ Not configured', status.llm_ready),
      statusRow('NVIDIA API Key', config.gemini_api_key ? `✅ ${config.gemini_api_key}` : '❌ Not set', !!config.gemini_api_key),
      statusRow('Active Sessions', String(status.active_sessions || 0), true),
    ].join('');

    const ragRows = document.getElementById('ragStatsRows');
    if (rag.by_source) {
      ragRows.innerHTML = Object.entries(rag.by_source).map(([src, count]) =>
        statusRow(src.charAt(0).toUpperCase() + src.slice(1), `${count} chunks`, true)
      ).join('');
    }
  } catch (e) {
    document.getElementById('adminStatusRows').innerHTML = '<div style="color:var(--danger);font-size:13px">Failed to load status.</div>';
  }
}

function statusRow(key, val, ok) {
  return `
    <div class="config-row">
      <span class="config-key">${key}</span>
      <span class="config-val ${ok ? '' : 'missing'}">${val}</span>
    </div>
  `;
}

async function saveConfig() {
  const key = document.getElementById('geminiKeyInput').value.trim();
  const calId = document.getElementById('calendarIdInput').value.trim();

  const payload = { google_calendar_id: calId };
  if (key) {
    payload.gemini_api_key = key;
  }

  try {
    const res = await fetch(API_BASE + '/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.success) {
      showToast('✅ Configuration saved successfully!', 'success');
      document.getElementById('geminiKeyInput').value = '';
      await loadAdminStatus();
      checkStatus();
    } else {
      showToast('⚠️ Configuration saved, but some services failed to initialize.', 'warning');
    }
  } catch {
    showToast('Failed to save configuration.', 'error');
  }
}

// Global for onclick
window.saveConfig = saveConfig;

async function reloadRAG() {
  try {
    const res = await fetch(API_BASE + '/api/rag/reload', { method: 'POST' });
    const data = await res.json();
    showToast('🔄 Knowledge base reload started…', 'success');
    setTimeout(() => loadAdminStatus(), 5000);
  } catch {
    showToast('Failed to trigger RAG reload.', 'error');
  }
}
window.reloadRAG = reloadRAG;

// ─────────────────────────────────────────────────────────────
// Evaluation Modal
// ─────────────────────────────────────────────────────────────
async function openEvalModal() {
  openModal('evalModal');
  const content = document.getElementById('evalContent');
  content.innerHTML = '<div class="no-slots-msg">Loading evaluation report…</div>';

  try {
    const res = await fetch(API_BASE + '/api/evaluation');
    if (!res.ok) {
      content.innerHTML = `
        <div style="text-align:center;padding:24px">
          <div style="font-size:40px;margin-bottom:12px">📋</div>
          <div style="font-size:14px;color:var(--text-secondary);margin-bottom:8px">No evaluation report yet.</div>
          <div style="font-size:13px;color:var(--text-muted)">Run this command to generate one:</div>
          <code style="display:block;margin-top:12px;padding:10px;background:var(--bg-input);border-radius:8px;font-size:12px;color:var(--accent)">python evaluator.py</code>
        </div>`;
      return;
    }
    const data = await res.json();
    renderEvalReport(data);
  } catch {
    content.innerHTML = '<div class="no-slots-msg">Failed to load evaluation data.</div>';
  }
}

function renderEvalReport(data) {
  const content = document.getElementById('evalContent');

  const overallColor = data.overall_score >= 8 ? 'var(--success)'
    : data.overall_score >= 6 ? 'var(--secondary)'
    : 'var(--danger)';

  const dimensionRows = Object.entries(data.dimensions || {}).map(([dim, score]) => {
    const barClass = score >= 8 ? 'high' : score >= 6 ? 'mid' : 'low';
    return `
      <div class="eval-dimension">
        <span class="eval-dim-name">${dim}</span>
        <div class="eval-bar-track">
          <div class="eval-bar-fill ${barClass}" style="width:${score * 10}%"></div>
        </div>
        <span class="eval-dim-score" style="color:${score >= 8 ? 'var(--success)' : score >= 6 ? 'var(--secondary)' : 'var(--danger)'}">${score}/10</span>
      </div>`;
  }).join('');

  const improvements = (data.improvements || []).map(imp => `
    <div style="padding:10px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:8px;margin-bottom:8px">
      <div style="font-size:12px;font-weight:700;color:var(--secondary);margin-bottom:4px">${imp.dimension} (${imp.score !== null ? imp.score + '/10' : 'Flagged'})</div>
      <div style="font-size:12px;color:var(--text-secondary)">${imp.recommendation}</div>
    </div>`).join('');

  const meta = data.metadata || {};

  content.innerHTML = `
    <div class="overall-score-display">
      <div class="score-number" style="color:${overallColor}">${data.overall_score}/10</div>
      <div class="score-label">Overall Production Score</div>
      <div style="font-size:12px;color:var(--text-muted);margin-top:4px">${meta.passed || 0}/${meta.total_tests || 0} tests passed · ${((meta.pass_rate || 0) * 100).toFixed(0)}% pass rate</div>
    </div>

    <div style="padding:0 4px">
      <div class="section-label" style="margin-bottom:12px">Dimension Scores</div>
      ${dimensionRows}
    </div>

    ${improvements ? `
    <div style="padding: 16px 4px 0">
      <div class="section-label" style="margin-bottom:12px">Top Improvements</div>
      ${improvements}
    </div>` : ''}

    <div style="text-align:center;padding-top:12px;font-size:11px;color:var(--text-muted)">
      Evaluated at: ${meta.evaluated_at ? new Date(meta.evaluated_at).toLocaleString() : 'N/A'}
    </div>
  `;
}

// ─────────────────────────────────────────────────────────────
// Modal Utilities
// ─────────────────────────────────────────────────────────────
function openModal(id) {
  document.getElementById(id).classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeModal(id) {
  document.getElementById(id).classList.remove('active');
  document.body.style.overflow = '';
}

// Globals for inline onclick
window.openModal = openModal;
window.closeModal = closeModal;

// ─────────────────────────────────────────────────────────────
// Toast Notifications
// ─────────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span>${message}</span>`;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function checkPersistedBooking() {
  const booking = localStorage.getItem('shaan_chatbot_booking');
  if (booking) {
    try {
      const data = JSON.parse(booking);
      updateBookingButtonState(data.date, data.time);
    } catch (e) {
      localStorage.removeItem('shaan_chatbot_booking');
    }
  }
}

function updateBookingButtonState(date, time) {
  const btn = document.getElementById('bookCTABtn');
  const info = document.getElementById('bookingTimeInfo');
  if (!btn) return;

  if (date && time) {
    btn.innerHTML = '✅ Appointment Already Booked';
    btn.disabled = true;
    btn.style.opacity = '0.7';
    btn.style.cursor = 'not-allowed';
    btn.style.background = 'rgba(16, 185, 129, 0.08)';
    btn.style.border = '1px solid rgba(16, 185, 129, 0.3)';
    btn.style.color = 'var(--success)';
    btn.style.boxShadow = 'none';
    
    if (info) {
      let formattedDate = date;
      try {
        const d = new Date(date + 'T00:00:00');
        formattedDate = d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
      } catch(e) {}
      info.textContent = `Scheduled for ${formattedDate} at ${time} IST`;
      info.style.display = 'block';
    }
  } else {
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" width="18" height="18"><path stroke-linecap="round" stroke-linejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5"/></svg> Book an Interview`;
    btn.disabled = false;
    btn.style.opacity = '';
    btn.style.cursor = '';
    btn.style.background = '';
    btn.style.border = '';
    btn.style.color = '';
    btn.style.boxShadow = '';
    if (info) {
      info.textContent = '';
      info.style.display = 'none';
    }
  }
}

async function resetAllBookings() {
  const confirmReset = confirm("Are you sure? This will permanently delete all booking records from the database.");
  if (!confirmReset) return;

  try {
    const res = await fetch(API_BASE + '/api/reset_bookings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await res.json();
    if (data.success) {
      showToast('All bookings have been successfully reset! 🧹', 'success');
      // Reload bookings list in admin panel
      await loadBookings();
      // Reload slots in availability UI
      if (typeof loadAvailability === 'function') {
        await loadAvailability();
      }
      // Reset memory state
      STATE.hasBookedInCurrentSession = false;
      
      // Update UI button and counter
      updateBookingButtonState();
      updateInterviewsCounter();
    } else {
      showToast(data.message || 'Failed to reset bookings.', 'error');
    }
  } catch (err) {
    showToast('Failed to reset bookings due to network error.', 'error');
  }
}

async function updateInterviewsCounter() {
  try {
    const res = await fetch(API_BASE + '/api/contacts');
    const contacts = await res.json();
    
    // Count only upcoming appointments
    const now = new Date();
    const upcoming = contacts.filter(c => {
      try {
        return new Date(`${c.date}T${convertTo24h(c.time)}`) >= now;
      } catch(e) {
        return true;
      }
    });
    
    const badge = document.getElementById('sidebarCounterBadge');
    if (badge) {
      if (upcoming.length > 0) {
        badge.innerHTML = `📅 ${upcoming.length} interview${upcoming.length > 1 ? 's' : ''} scheduled`;
        badge.style.display = 'inline-flex';
      } else {
        badge.style.display = 'none';
      }
    }
  } catch(e) {}
}

function convertTo24h(timeStr) {
  if (!timeStr) return "00:00:00";
  const parts = timeStr.split(' ');
  const time = parts[0];
  const modifier = parts[1];
  let [hours, minutes] = time.split(':');
  if (!minutes) minutes = "00";
  if (hours === '12') {
    hours = '00';
  }
  if (modifier === 'PM') {
    hours = parseInt(hours, 10) + 12;
  }
  return `${String(hours).padStart(2, '0')}:${minutes}:00`;
}
