(() => {
  const form = document.querySelector("#prediction-form");
  const sampleButton = document.querySelector("#sample-case");
  const modelState = document.querySelector("#model-load-state");
  const errorBox = document.querySelector("#form-error");
  const submitButton = document.querySelector(".predict-button");
  const placeholder = document.querySelector("#result-placeholder");
  const resultContent = document.querySelector("#result-content");
  const toast = document.querySelector("#toast");
  let modelPromise;
  let manifestCache;
  let forestView;

  const $ = (selector) => document.querySelector(selector);
  const localDate = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  };
  const todayAtNoon = () => {
    const now = new Date();
    return new Date(now.getFullYear(), now.getMonth(), now.getDate(), 12);
  };
  const prettyNumber = (value, digits = 0) => new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
  const prettyDate = (date, options = { month: "short", day: "numeric", year: "numeric" }) =>
    new Intl.DateTimeFormat("en-US", options).format(date);

  function setSampleDates() {
    const today = todayAtNoon();
    const lastService = new Date(today);
    lastService.setDate(lastService.getDate() - 180);
    $("#last-service-date").value = localDate(lastService);
    $("#last-service-date").max = localDate(today);
    $("#model-year").max = String(today.getFullYear());
    $("#model-year").value = String(today.getFullYear() - 5);
  }

  function readFields() {
    const values = Object.fromEntries(new FormData(form).entries());
    for (const key of [
      "current_odometer_km", "last_service_odometer_km", "manufacturing_year",
      "average_daily_km", "recommended_interval_km", "recommended_interval_months",
    ]) values[key] = Number(values[key]);
    return values;
  }

  function validate(values) {
    const issues = [];
    if (!Number.isFinite(values.current_odometer_km) || values.current_odometer_km < 0) issues.push("Enter a valid current odometer reading.");
    if (!Number.isFinite(values.last_service_odometer_km) || values.last_service_odometer_km < 0) issues.push("Enter a valid last-service odometer reading.");
    if (values.current_odometer_km < values.last_service_odometer_km) issues.push("The current odometer must be at least the last-service reading.");
    if (!(values.average_daily_km > 0)) issues.push("Average daily use must be greater than zero.");
    if (values.recommended_interval_km < 1000 || values.recommended_interval_km > 50000) issues.push("Enter a manual distance interval between 1,000 and 50,000 km.");
    if (values.recommended_interval_months < 1 || values.recommended_interval_months > 60) issues.push("Enter a manual time interval between 1 and 60 months.");
    const serviceDate = new Date(`${values.last_service_date}T12:00:00`);
    const today = todayAtNoon();
    const ageMonths = (today - serviceDate) / (86400000 * 30.4375);
    if (!values.last_service_date || Number.isNaN(serviceDate.getTime())) issues.push("Choose a valid last-service date.");
    else if (serviceDate > today) issues.push("The last-service date cannot be in the future.");
    else if (ageMonths > 60) issues.push("For this demo, choose a service date from the past 60 months.");
    if (values.manufacturing_year < 1980 || values.manufacturing_year > today.getFullYear()) issues.push("Choose a valid model year.");
    return issues;
  }

  async function loadModel() {
    if (manifestCache && forestView) return { manifest: manifestCache, forest: forestView };
    if (!modelPromise) {
      modelState.textContent = "LOADING MODEL";
      modelPromise = (async () => {
        const manifestUrl = new URL("./assets/model-manifest.json", window.location.href);
        const forestUrl = new URL("./assets/forest.bin.deflate", window.location.href);
        const [manifestResponse, forestResponse] = await Promise.all([fetch(manifestUrl), fetch(forestUrl)]);
        if (!manifestResponse.ok || !forestResponse.ok) throw new Error("The model files could not be loaded. Please refresh the demo and try again.");
        const manifest = await manifestResponse.json();
        if (!("DecompressionStream" in window)) throw new Error("This browser does not support the compressed model format. Please use a recent version of Chrome, Edge, Firefox, or Safari.");
        const compressed = await forestResponse.arrayBuffer();
        const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate"));
        const bytes = await new Response(stream).arrayBuffer();
        if (bytes.byteLength !== manifest.node_count * manifest.node_bytes) throw new Error("The downloaded model file did not pass its integrity check.");
        manifestCache = manifest;
        forestView = new DataView(bytes);
        modelState.textContent = `MODEL READY · ${manifest.tree_count} TREES`;
        return { manifest, forest: forestView };
      })().catch((error) => {
        modelPromise = null;
        modelState.textContent = "MODEL UNAVAILABLE";
        throw error;
      });
    }
    return modelPromise;
  }

  function modelFeatures(values, manifest) {
    const row = manifest.numeric_features.map((key) => {
      if (key === "km_since_last_service") return Math.max(0, values.current_odometer_km - values.last_service_odometer_km);
      if (key === "vehicle_age_years") return Math.max(0, todayAtNoon().getFullYear() - values.manufacturing_year);
      if (key === "months_since_last_service") {
        const serviceDate = new Date(`${values.last_service_date}T12:00:00`);
        return Math.max(0, (todayAtNoon() - serviceDate) / (86400000 * 30.4375));
      }
      return values[key];
    });
    for (const feature of manifest.categorical_features) {
      for (const category of manifest.categorical_categories[feature]) row.push(values[feature] === category ? 1 : 0);
    }
    return row;
  }

  function predictForest(row, manifest, forest) {
    let total = 0;
    for (let tree = 0; tree < manifest.tree_count; tree += 1) {
      let node = manifest.tree_offsets[tree];
      const end = manifest.tree_offsets[tree + 1];
      let hops = 0;
      while (true) {
        if (node < manifest.tree_offsets[tree] || node >= end || hops > 100) throw new Error("The browser encountered an invalid path in the model.");
        const offset = node * manifest.node_bytes;
        const left = forest.getInt32(offset, true);
        const right = forest.getInt32(offset + 4, true);
        const feature = forest.getInt32(offset + 8, true);
        if (feature < 0) {
          total += forest.getFloat64(offset + 20, true);
          break;
        }
        node = row[feature] <= forest.getFloat64(offset + 12, true) ? left : right;
        hops += 1;
      }
    }
    return Math.max(0, total / manifest.tree_count);
  }

  function makeEstimate(modelKm, values) {
    const today = todayAtNoon();
    const serviceDate = new Date(`${values.last_service_date}T12:00:00`);
    const monthsSince = Math.max(0, (today - serviceDate) / (86400000 * 30.4375));
    const kmSince = Math.max(0, values.current_odometer_km - values.last_service_odometer_km);
    const distanceCap = Math.max(0, values.recommended_interval_km - kmSince);
    const timeCapDays = Math.max(0, (values.recommended_interval_months - monthsSince) * 30.4375);
    const cappedKm = Math.min(modelKm, distanceCap, timeCapDays * values.average_daily_km);
    const remainingDays = Math.max(0, Math.min(Math.floor(cappedKm / values.average_daily_km), Math.floor(timeCapDays)));
    const dueDate = new Date(today);
    dueDate.setDate(dueDate.getDate() + remainingDays);
    let status = "Service Not Due";
    let tone = "safe";
    if (cappedKm <= 0 || remainingDays <= 0 || cappedKm <= .05 * values.recommended_interval_km || remainingDays <= 30) {
      status = "Service Due"; tone = "due";
    } else if (cappedKm <= .15 * values.recommended_interval_km || remainingDays <= 90) {
      status = "Service Approaching"; tone = "approaching";
    }
    const remainingDistance = Math.max(0, Math.round(cappedKm));
    return {
      modelKm, distanceCap, timeCapDays, remainingDistance, remainingDays, dueDate, status, tone,
      usedPercent: Math.min(100, Math.max(0, 100 * (values.recommended_interval_km - remainingDistance) / values.recommended_interval_km)),
    };
  }

  function renderEstimate(estimate) {
    placeholder.hidden = true;
    resultContent.hidden = false;
    const number = $("#remaining-km");
    number.textContent = prettyNumber(estimate.remainingDistance);
    number.animate?.([{ opacity: .2, transform: "translateY(7px)" }, { opacity: 1, transform: "translateY(0)" }], { duration: 480, easing: "cubic-bezier(.2,.8,.2,1)" });
    $("#used-percent").textContent = `${Math.round(estimate.usedPercent)}%`;
    $("#progress-fill").style.width = `${estimate.usedPercent}%`;
    $("#service-date").textContent = prettyDate(estimate.dueDate);
    $("#date-day").textContent = String(estimate.dueDate.getDate()).padStart(2, "0");
    $("#remaining-days").textContent = estimate.remainingDays === 0 ? "within the current service window" : `about ${prettyNumber(estimate.remainingDays)} days from today`;
    $("#raw-km").textContent = `${prettyNumber(Math.round(estimate.modelKm))} km`;
    $("#distance-cap").textContent = `${prettyNumber(Math.round(estimate.distanceCap))} km`;
    $("#time-cap").textContent = `${prettyNumber(Math.round(estimate.timeCapDays * readFields().average_daily_km))} km equivalent`;
    const status = $("#result-status");
    status.className = `result-status status-${estimate.tone}`;
    status.querySelector("span").textContent = estimate.status.toUpperCase();
    const reason = estimate.modelKm > estimate.distanceCap || estimate.modelKm > estimate.timeCapDays * readFields().average_daily_km
      ? "The final estimate was capped by the earlier owner-manual distance or time limit you entered."
      : "The model estimate falls within the owner-manual limits you entered.";
    $("#result-explanation").textContent = `${reason} This is a synthetic-data classroom demo, not a real-car service recommendation.`;
    $("#hero-km").textContent = prettyNumber(estimate.remainingDistance);
    $("#hero-status").textContent = estimate.status;
    $("#hero-date").textContent = prettyDate(estimate.dueDate, { month: "short", day: "numeric" });
    const heroDot = document.querySelector(".status-dot");
    heroDot.style.background = estimate.tone === "due" ? "var(--danger)" : estimate.tone === "approaching" ? "var(--orange)" : "var(--lime)";
    document.querySelector(".ring-value").style.strokeDashoffset = String(239 - Math.min(239, 239 * estimate.usedPercent / 100));
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    const values = readFields();
    const problems = validate(values);
    if (problems.length) return showError(problems[0]);
    submitButton.disabled = true;
    submitButton.querySelector(".button-label").textContent = "Running model…";
    try {
      const { manifest, forest } = await loadModel();
      const row = modelFeatures(values, manifest);
      const rawPrediction = predictForest(row, manifest, forest);
      renderEstimate(makeEstimate(rawPrediction, values));
    } catch (error) {
      modelState.textContent = "MODEL UNAVAILABLE";
      showError(error instanceof Error ? error.message : "Prediction failed. Please refresh and try again.");
    } finally {
      submitButton.disabled = false;
      submitButton.querySelector(".button-label").textContent = "Run prediction";
    }
  });

  sampleButton.addEventListener("click", () => {
    form.reset();
    setSampleDates();
    errorBox.hidden = true;
    form.requestSubmit();
  });

  const menuButton = $("#menu-toggle");
  const nav = $("#nav-links");
  menuButton.addEventListener("click", () => {
    const open = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    nav.classList.toggle("is-open", open);
  });
  nav.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      nav.classList.remove("is-open");
      menuButton.setAttribute("aria-expanded", "false");
      menuButton.setAttribute("aria-label", "Open navigation");
    }
  });

  setSampleDates();
  modelState.textContent = "MODEL READY · LOAD ON RUN";
})();
