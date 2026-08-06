const form = document.getElementById("predict-form");
const statusEl = document.getElementById("status");
const table = document.getElementById("results");
const tbody = table.querySelector("tbody");
const submitBtn = document.getElementById("submit-btn");

const MODEL_LABELS = { rfr: "Random Forest", svr: "SVR", mat: "MAT (graph + 3D)" };
const MODEL_ORDER = ["rfr", "svr", "mat"];

function fmt(x, digits = 3) {
  return typeof x === "number" ? x.toFixed(digits) : "\u2014";
}

function rowFor(key, res) {
  const tr = document.createElement("tr");
  tr.classList.add(`model-${key}`);

  if (!res || res.error) {
    tr.innerHTML = `
      <td>${MODEL_LABELS[key]}</td>
      <td colspan="4" class="error">${res ? res.error : "no result"}</td>
      <td></td>`;
    return tr;
  }

  const m = res.test_metrics || {};
  const notes = key === "mat" && res.conformer_ok === false
    ? "3D embed failed \u2014 used topological fallback"
    : "";

  tr.innerHTML = `
    <td>${MODEL_LABELS[key]}</td>
    <td class="pred">${fmt(res.prediction)}</td>
    <td>${fmt(m.r2)}</td>
    <td>${fmt(m.mae)}</td>
    <td>${fmt(m.mse)}</td>
    <td class="notes">${notes}</td>`;
  return tr;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const smiles = document.getElementById("smiles").value.trim();
  if (!smiles) return;

  submitBtn.disabled = true;
  statusEl.textContent = "Running predictions\u2026 MAT includes 3D conformer generation and may take a few seconds.";
  table.classList.add("hidden");
  tbody.innerHTML = "";

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "request failed");

    for (const key of MODEL_ORDER) {
      tbody.appendChild(rowFor(key, data[key]));
    }
    table.classList.remove("hidden");
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});
