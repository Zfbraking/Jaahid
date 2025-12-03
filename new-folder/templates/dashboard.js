document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("summarizeForm");
    const results = document.getElementById("results");

    if (!form || !results) {
        // Page doesn't have the expected elements
        return;
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault(); // prevent full-page form submit

        const formData = new FormData(form);
        results.innerHTML = "<div class='alert alert-info'>Processing...</div>";

        try {
            const res = await fetch("/api/summarize", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (data.error) {
                results.innerHTML = "<div class='alert alert-danger'>" + data.error + "</div>";
                return;
            }

            const summary = data.summary || "";

            results.innerHTML = `
                <h4>Result</h4>
                <div class="card p-3 mb-3" style="white-space: pre-wrap;">
                    ${summary}
                </div>
            `;
        } catch (err) {
            results.innerHTML = "<div class='alert alert-danger'>Error: " + err.message + "</div>";
        }
    });
});
