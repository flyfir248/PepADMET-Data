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

function truncateSmiles(s, max = 28) {
  return s.length > max ? s.slice(0, max) + "\u2026" : s;
}

function parseSmilesInput(raw) {
  return raw
    .split(/\r?\n/)
    .flatMap((line) => line.split(","))
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function modelRow(key, res, isFirstInGroup, groupSize, smiles, groupIndex) {
  const tr = document.createElement("tr");
  tr.classList.add(`model-${key}`, groupIndex % 2 === 0 ? "group-even" : "group-odd");

  const smilesCell = isFirstInGroup
    ? `<td class="smiles-cell" rowspan="${groupSize}" title="${smiles}">${truncateSmiles(smiles)}</td>`
    : "";

  if (!res || res.error) {
    tr.innerHTML = `
      ${smilesCell}
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
    ${smilesCell}
    <td>${MODEL_LABELS[key]}</td>
    <td class="pred">${fmt(res.prediction)}</td>
    <td>${fmt(m.r2)}</td>
    <td>${fmt(m.mae)}</td>
    <td>${fmt(m.mse)}</td>
    <td class="notes">${notes}</td>`;
  return tr;
}

function errorGroupRow(smiles, message, groupIndex) {
  const tr = document.createElement("tr");
  tr.classList.add(groupIndex % 2 === 0 ? "group-even" : "group-odd");
  tr.innerHTML = `
    <td class="smiles-cell" title="${smiles}">${truncateSmiles(smiles)}</td>
    <td colspan="6" class="error">${message}</td>`;
  return tr;
}

function renderResults(results) {
  tbody.innerHTML = "";
  results.forEach((entry, groupIndex) => {
    if (entry.error && !entry.rfr && !entry.svr && !entry.mat) {
      tbody.appendChild(errorGroupRow(entry.smiles, entry.error, groupIndex));
      return;
    }
    MODEL_ORDER.forEach((key, i) => {
      tbody.appendChild(
        modelRow(key, entry[key], i === 0, MODEL_ORDER.length, entry.smiles, groupIndex)
      );
    });
  });
  table.classList.remove("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const smilesList = parseSmilesInput(document.getElementById("smiles").value);
  if (smilesList.length === 0) return;

  submitBtn.disabled = true;
  statusEl.textContent = smilesList.length > 1
    ? `Running predictions for ${smilesList.length} compounds\u2026 MAT includes 3D conformer generation per compound, so this may take a while for larger batches.`
    : "Running predictions\u2026 MAT includes 3D conformer generation and may take a few seconds.";
  table.classList.add("hidden");
  tbody.innerHTML = "";

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: smilesList }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "request failed");

    renderResults(data.results);
    statusEl.textContent = `${data.count} compound${data.count === 1 ? "" : "s"} predicted.`;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});
