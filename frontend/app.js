const backendUrlInput = document.getElementById("backendUrl");
const queryInput = document.getElementById("queryInput");
const kInput = document.getElementById("kInput");
const searchBtn = document.getElementById("searchBtn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

function setStatus(text) {
  statusEl.textContent = text;
}

function clearResults() {
  resultsEl.innerHTML = "";
}

function renderResults(payload) {
  const results = payload.results || [];
  if (results.length === 0) {
    setStatus("No results found.");
    return;
  }

  setStatus(`Found ${results.length} results in ${payload.search_ms} ms`);
  clearResults();

  for (const item of results) {
    const card = document.createElement("article");
    card.className = "card";

    const title = document.createElement("h3");
    title.textContent = item.title || "Untitled";
    card.appendChild(title);

    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = `Score: ${item.score.toFixed(3)} | Year: ${item.release_year ?? "NA"} | Language: ${item.original_language ?? "NA"}`;
    card.appendChild(meta);

    if (item.genre) {
      const genre = document.createElement("p");
      genre.className = "meta";
      genre.textContent = `Genre: ${item.genre}`;
      card.appendChild(genre);
    }

    if (item.description) {
      const desc = document.createElement("p");
      desc.className = "desc";
      desc.textContent = item.description;
      card.appendChild(desc);
    }

    if (item.url) {
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Open";
      card.appendChild(link);
    }

    resultsEl.appendChild(card);
  }
}

async function runSearch() {
  const baseUrl = backendUrlInput.value.trim().replace(/\/$/, "");
  const query = queryInput.value.trim();
  const k = Number(kInput.value);

  if (!baseUrl) {
    setStatus("Enter backend URL.");
    return;
  }

  if (!query) {
    setStatus("Enter a search query.");
    return;
  }

  if (!Number.isInteger(k) || k < 1 || k > 20) {
    setStatus("K must be between 1 and 20.");
    return;
  }

  searchBtn.disabled = true;
  setStatus("Searching...");
  clearResults();

  try {
    const url = `${baseUrl}/search?q=${encodeURIComponent(query)}&k=${k}`;
    const response = await fetch(url);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Request failed (${response.status}): ${errorText}`);
    }

    const payload = await response.json();
    renderResults(payload);
  } catch (error) {
    setStatus(error.message || "Search failed.");
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});
