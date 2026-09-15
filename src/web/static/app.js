/* Atlas Research — frontend logic.
   Vanilla JS on purpose: no build step, deploys as static files. */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("query-form");
  const input = $("query-input");
  const submitBtn = $("submit-btn");
  const formError = $("form-error");
  const pipelinePanel = $("pipeline-panel");
  const pipelineTitle = $("pipeline-title");
  const statusPill = $("status-pill");
  const stepper = $("stepper");
  const log = $("log");
  const reportPanel = $("report-panel");
  const reportQuery = $("report-query");
  const reportBody = $("report-body");
  const verifiedBadge = $("verified-badge");
  const dlMd = $("dl-md");
  const dlPdf = $("dl-pdf");

  let eventSource = null;

  /* ---------------------------------------------- example chips ---- */
  $("examples").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    input.value = chip.textContent.trim();
    input.focus();
  });

  /* ---------------------------------------------- run lifecycle ---- */
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (query.length < 3) return;

    setError(null);
    setRunning(true);
    resetPipeline(query);

    let runId;
    try {
      const res = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      runId = (await res.json()).run_id;
    } catch (err) {
      setRunning(false);
      pipelinePanel.hidden = true;
      setError(err.message);
      return;
    }

    // Stream live agent progress over SSE.
    eventSource = new EventSource(`/api/runs/${runId}/events`);
    eventSource.onmessage = (msg) => {
      const evt = JSON.parse(msg.data);
      if (evt.type === "log") {
        appendLog(evt.agent, evt.message);
        updateStepper(evt.agent);
      } else if (evt.type === "done") {
        eventSource.close();
        eventSource = null;
        finishRun(runId, evt);
      }
    };
    eventSource.onerror = () => {
      // EventSource retries transient drops on its own (readyState CONNECTING).
      // CLOSED means it gave up for good — e.g. the server restarted and the
      // run id no longer exists, so the stream 404s. Without handling that the
      // page sits on "Running" with the submit button disabled forever.
      if (eventSource && eventSource.readyState === EventSource.CLOSED) {
        eventSource = null;
        setRunning(false);
        statusPill.dataset.state = "failed";
        statusPill.textContent = "Disconnected";
        pipelineTitle.textContent = "Lost connection to the server";
        appendLog("System", "Progress stream closed. The run may still be going — reload to check.");
      }
    };
  });

  async function finishRun(runId, evt) {
    setRunning(false);
    if (evt.status !== "completed") {
      statusPill.dataset.state = "failed";
      statusPill.textContent = "Failed";
      pipelineTitle.textContent = "This run didn't make it";
      appendLog("System", evt.error || "Unknown error");
      return;
    }

    statusPill.dataset.state = "done";
    statusPill.textContent = "Complete";
    pipelineTitle.textContent = "Research complete";
    stepper.querySelectorAll(".step").forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });

    const run = await (await fetch(`/api/runs/${runId}`)).json();
    reportQuery.textContent = run.query;
    reportBody.innerHTML = run.report_html; // server-rendered, sanitized on the server
    wrapTables(reportBody);
    verifiedBadge.className = "badge " + (run.verified ? "ok" : "warn");
    verifiedBadge.textContent = run.verified ? "✓ Fact-checked" : "⚠ Needs review";
    dlMd.href = `/api/runs/${runId}/download/md`;
    dlPdf.href = `/api/runs/${runId}/download/pdf`;
    dlPdf.hidden = !run.has_pdf;
    reportPanel.hidden = false;
    reportPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    loadHistory();
  }

  /* ---------------------------------------------- UI helpers ---- */
  // Markdown emits bare <table> elements. Wrap each one so a wide table
  // scrolls inside its own box instead of stretching the page, while still
  // filling the column width when it fits.
  function wrapTables(root) {
    root.querySelectorAll("table").forEach((table) => {
      const wrap = document.createElement("div");
      wrap.className = "table-scroll";
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
  }

  function setRunning(running) {
    submitBtn.disabled = running;
    submitBtn.classList.toggle("is-loading", running);
    input.readOnly = running;
  }

  function setError(message) {
    formError.hidden = !message;
    formError.textContent = message || "";
  }

  function resetPipeline(query) {
    pipelineTitle.textContent = `Researching: ${query}`;
    statusPill.dataset.state = "running";
    statusPill.textContent = "Running";
    log.replaceChildren();
    stepper.querySelectorAll(".step").forEach((s) => s.classList.remove("active", "done"));
    reportPanel.hidden = true;
    pipelinePanel.hidden = false;
    pipelinePanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function appendLog(agent, message) {
    const line = document.createElement("div");
    line.className = "log-line";
    const tag = document.createElement("span");
    tag.className = "log-agent";
    tag.dataset.agent = agent;
    tag.textContent = agent;
    const msg = document.createElement("span");
    msg.className = "log-msg";
    msg.textContent = message; // textContent — log text is never parsed as HTML
    line.append(tag, msg);
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  const STEP_ORDER = ["Manager", "Researcher", "Writer", "Fact-Checker"];
  function updateStepper(agent) {
    const idx = STEP_ORDER.indexOf(agent);
    if (idx === -1) return; // System/Graph/Search messages don't move the stepper
    stepper.querySelectorAll(".step").forEach((step, i) => {
      // The Writer <-> Fact-Checker revision loop naturally re-activates
      // earlier steps; "done" = strictly before the active one.
      step.classList.toggle("active", i === idx);
      step.classList.toggle("done", i < idx);
    });
  }

  /* ---------------------------------------------- history ---- */
  async function loadHistory() {
    let reports = [];
    try { reports = await (await fetch("/api/reports")).json(); } catch { return; }
    const section = $("history-section");
    const list = $("history-list");
    section.hidden = reports.length === 0;
    list.replaceChildren(
      ...reports.slice(0, 12).map((r) => {
        const li = document.createElement("li");
        li.className = "history-item";
        const name = document.createElement("span");
        name.className = "history-name";
        // "20260827T194306Z_tell-me-about..." -> "tell me about..."
        name.textContent = r.name.replace(/^[^_]*_/, "").replace(/-/g, " ");
        const links = document.createElement("span");
        links.className = "history-links";
        const md = document.createElement("a");
        md.href = `/api/reports/${encodeURIComponent(r.md)}/file`;
        md.textContent = "MD";
        links.append(md);
        if (r.pdf) {
          const pdf = document.createElement("a");
          pdf.href = `/api/reports/${encodeURIComponent(r.pdf)}/file`;
          pdf.textContent = "PDF";
          links.append(pdf);
        }
        li.append(name, links);
        return li;
      })
    );
  }

  loadHistory();
})();
