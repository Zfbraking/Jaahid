document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const messages = document.getElementById("chat-messages");
    const errorBox = document.getElementById("chat-error");

    if (!form || !input || !messages) return;

    const appendMessage = (role, content) => {
        const wrapper = document.createElement("div");
        wrapper.className = "d-flex flex-column mb-2 " + (role === "user" ? "align-items-end" : "align-items-start");

        const roleDiv = document.createElement("div");
        roleDiv.className = "chat-role";
        roleDiv.textContent = role === "user" ? "You" : "Assistant";

        const bubble = document.createElement("div");
        bubble.className = "chat-bubble " + (role === "user" ? "chat-user" : "chat-assistant");
        bubble.textContent = content;

        wrapper.appendChild(roleDiv);
        wrapper.appendChild(bubble);
        messages.appendChild(wrapper);

        messages.scrollTop = messages.scrollHeight;
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        errorBox.style.display = "none";
        const question = input.value.trim();
        if (!question) return;

        // Show user message immediately
        appendMessage("user", question);
        input.value = "";

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question })
            });

            const data = await res.json();
            if (!res.ok || data.error) {
                const msg = data.error || ("HTTP " + res.status);
                errorBox.textContent = msg;
                errorBox.style.display = "block";
                return;
            }

            appendMessage("assistant", data.answer || "");
        } catch (err) {
            errorBox.textContent = "Error: " + err.message;
            errorBox.style.display = "block";
        }
    });
});
