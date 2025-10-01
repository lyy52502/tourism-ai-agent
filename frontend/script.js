const chatBox = document.getElementById('chat-box');
const messageInput = document.getElementById('message');
const sendBtn = document.getElementById('send-btn');

const userId = "user_001";
const sessionId = "session_001";
const location = { lat: 59.8586, lng: 17.6389 }; 

function appendMessage(text, sender="bot") {
    const msgDiv = document.createElement('div');
    msgDiv.className = sender === "user" ? "user-msg" : "bot-msg";
    msgDiv.innerHTML = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') sendMessage();
});

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;
    appendMessage(message, "user");
    messageInput.value = "";

    try {
        const res = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, user_id: userId, session_id: sessionId, location })
        });

        if (!res.ok) {
            appendMessage("Error: Failed to get response from server");
            return;
        }

        const data = await res.json();
        appendMessage(data.reply);

        if (data.recommendations && data.recommendations.length > 0) {
            let recText = "<ul>";
            data.recommendations.forEach(r => {
                recText += `<li><b>${r.name}</b> (${r.type}) - Rating: ${r.rating}, Distance: ${r.distance.toFixed(1)} km</li>`;
            });
            recText += "</ul>";
            appendMessage(recText);
        }

    } catch (err) {
        appendMessage("Error: " + err.message);
    }
}
