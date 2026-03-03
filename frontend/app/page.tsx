"use client";

import { useEffect, useMemo, useState } from "react";

type Tab = "setup" | "daily" | "insights";

interface DailyLog {
  id: string;
  date: string;
  mood: number;
  cramps: number;
  fatigue: number;
  headache: number;
  acne: number;
  sleep_hours: number;
  caffeine_level: number;
  stress: number;
  workout: boolean;
  basal_temp: number | null;
  resting_hr: number | null;
}

interface TrainResponse {
  metrics: {
    mae: number;
    rmse: number;
    baseline_mae: number;
    baseline_rmse: number;
    model_name: string;
  };
  confidence: {
    value: number;
    label: string;
  };
  global_importance: Array<{ feature: string; label?: string; importance: number; importance_pct?: number }>;
  predictions: Array<{ date: string; actual: number; predicted: number; actual_pct?: number; predicted_pct?: number }>;
  data_quality?: {
    rows_used: number;
    used_reference_boost: boolean;
    low_variance_features: string[];
  };
  score_scale?: string;
  training_summary?: string;
  model_choice_reason?: string;
  next_step_hint?: string;
  notes: string;
}

interface PredictResponse {
  prediction: number;
  actual_logged_score?: number;
  actual_status_label?: string;
  actual_status_emoji?: string;
  predicted_status_label?: string;
  predicted_status_emoji?: string;
  expected_gap?: number;
  symptom_burden_pct?: number;
  wellness_score?: number;
  status_label?: string;
  status_emoji?: string;
  score_scale?: string;
  confidence: {
    value: number;
    label: string;
  };
  drivers: Array<{
    feature: string;
    label?: string;
    contribution: number;
    direction: string;
    current_value?: number;
    baseline_value?: number;
  }>;
  driver_summaries?: string[];
  interpretation?: string;
  notes: string;
}

interface ForecastResponse {
  score_scale?: string;
  confidence: {
    value: number;
    label: string;
  };
  forecast: Array<{
    date: string;
    predicted_bad_day_score: number;
    symptom_burden_pct?: number;
    wellness_score?: number;
    status_label?: string;
    status_emoji?: string;
    risk_label: string;
  }>;
  notes: string;
}

const STORAGE_KEY = "pms_mood_compass_session_v1";
const LEGACY_STORAGE_KEY = "pms_trigger_finder_data_v1";
const FEATURE_LABELS: Record<string, string> = {
  sleep_hours: "Sleep hours",
  caffeine_level: "Caffeine level",
  stress: "Stress",
  workout: "Workout",
  cycle_day: "Cycle day",
  cycle_day_normalized: "Cycle day (normalized)",
  late_luteal: "Late luteal phase",
  basal_temp: "Basal temperature",
  resting_hr: "Resting heart rate",
  basal_temp_missing: "Missing basal temp",
  resting_hr_missing: "Missing resting HR",
  sleep_prev: "Previous sleep",
  stress_prev: "Previous stress",
  score_prev: "Previous score",
  stress_x_late_luteal: "Stress x late luteal",
  sleep_x_late_luteal: "Sleep x late luteal",
  caffeine_x_sleep: "Caffeine x sleep",
};

interface GuideCard {
  category: "Cycle fact" | "Care tip" | "Nutrition tip";
  title: string;
  body: string;
  footer?: string;
}

const GUIDE_CARD_LIBRARY: GuideCard[] = [
  {
    category: "Cycle fact",
    title: "Cycle timing naturally shifts",
    body: "A cycle does not need to be identical every month. Small timing changes are common, which is why trends across multiple cycles matter more than one isolated month.",
  },
  {
    category: "Cycle fact",
    title: "Low-symptom days matter too",
    body: "Logging easier days is useful because the model needs both calm days and harder days to learn what is normal for you.",
  },
  {
    category: "Cycle fact",
    title: "Phase matters",
    body: "Symptoms often change by cycle phase. The days around bleeding and the late luteal window can feel different even when lifestyle habits stay similar.",
  },
  {
    category: "Cycle fact",
    title: "Patterns are usually multi-factor",
    body: "Mood, cramps, fatigue, sleep, and stress often move together. The model looks for combinations of signals, not a single universal trigger.",
  },
  {
    category: "Care tip",
    title: "Protect sleep before expected symptom days",
    body: "A steadier wind-down routine and a consistent sleep window can make your logged pattern easier to interpret and may reduce symptom spikes for some people.",
  },
  {
    category: "Care tip",
    title: "Keep movement gentle when symptoms rise",
    body: "Light walks, stretching, or yoga are often easier to sustain than intense workouts during tougher days and still give you a useful signal in the tracker.",
  },
  {
    category: "Care tip",
    title: "Support recovery with simple basics",
    body: "Hydration, regular meals, heat packs, and reduced caffeine can be practical first adjustments when cramps, fatigue, or irritability rise.",
  },
  {
    category: "Care tip",
    title: "Give the model enough history",
    body: "Try to log at least 2 to 3 cycles before treating the predictions as stable. Short histories can still help, but they are less reliable.",
  },
];

const NUTRITION_CARD_LIBRARY: GuideCard[] = [
  {
    category: "Nutrition tip",
    title: "Rebuild around bleeding days",
    body: "During and after bleeding days, prioritize iron-rich foods such as lentils, beans, spinach, tofu, eggs, or lean meats, and pair them with vitamin C sources to support absorption.",
  },
  {
    category: "Nutrition tip",
    title: "Seed cycling is optional",
    body: "Some people use flax and pumpkin seeds earlier in the cycle, then sesame and sunflower later. It can be a food routine if you enjoy it, but it is not a medical requirement.",
  },
  {
    category: "Nutrition tip",
    title: "Keep blood sugar steadier",
    body: "Meals that combine protein, fiber, and healthy fats can reduce sharp energy swings and may feel more sustainable during lower-mood or higher-fatigue days.",
  },
  {
    category: "Nutrition tip",
    title: "Hydration still matters",
    body: "Water, soups, fruit, and electrolyte drinks can help when cramps, headaches, or fatigue rise. Low hydration can make harder days feel heavier.",
  },
  {
    category: "Nutrition tip",
    title: "Track magnesium-rich foods",
    body: "Pumpkin seeds, leafy greens, almonds, and beans are worth tracking around cramps and headaches to see whether they line up with easier days for you.",
  },
];

const defaultLog = (): Omit<DailyLog, "id"> => ({
  date: new Date().toISOString().slice(0, 10),
  mood: 2,
  cramps: 2,
  fatigue: 2,
  headache: 1,
  acne: 1,
  sleep_hours: 7,
  caffeine_level: 1,
  stress: 2,
  workout: false,
  basal_temp: null,
  resting_hr: null,
});

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function stdDev(values: number[]): number {
  if (!values.length) return 0;
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function normalizeDateList(dates: string[]): string[] {
  return Array.from(new Set(dates.filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function cycleStats(periodStarts: string[]) {
  const sorted = normalizeDateList(periodStarts);
  const lengths: number[] = [];
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = new Date(sorted[i - 1]).getTime();
    const curr = new Date(sorted[i]).getTime();
    const diff = Math.round((curr - prev) / (1000 * 60 * 60 * 24));
    if (diff > 0) lengths.push(diff);
  }

  return {
    cycleLengths: lengths,
    estimatedCycleLength: lengths.length ? median(lengths) : 28,
    regularityScore: lengths.length ? stdDev(lengths) : 0,
  };
}

function inferPeriodStartsFromLogs(logs: DailyLog[]): string[] {
  if (!logs.length) return [];
  const sortedDates = [...logs].sort((a, b) => a.date.localeCompare(b.date));
  const starts: string[] = [];
  const first = new Date(sortedDates[0].date);
  const last = new Date(sortedDates[sortedDates.length - 1].date);
  const cursor = new Date(first);

  while (cursor <= new Date(last.getTime() + 28 * 24 * 60 * 60 * 1000)) {
    starts.push(cursor.toISOString().slice(0, 10));
    cursor.setDate(cursor.getDate() + 28);
  }

  return starts;
}

function parseJsonImport(raw: string): { periodStarts: string[]; logs: DailyLog[] } {
  const parsed = JSON.parse(raw);
  const incomingLogs: any[] = Array.isArray(parsed) ? parsed : parsed.logs || [];
  const incomingPeriodStarts: string[] = Array.isArray(parsed.period_starts)
    ? parsed.period_starts
    : Array.isArray(parsed.periodStarts)
      ? parsed.periodStarts
      : [];

  const logs: DailyLog[] = incomingLogs
    .map((item, index) => ({
      id: String(item.id || `${item.date || "log"}-${index}`),
      date: String(item.date),
      mood: Number(item.mood ?? 0),
      cramps: Number(item.cramps ?? 0),
      fatigue: Number(item.fatigue ?? 0),
      headache: Number(item.headache ?? 0),
      acne: Number(item.acne ?? 0),
      sleep_hours: Number(item.sleep_hours ?? 0),
      caffeine_level: Number(item.caffeine_level ?? 0),
      stress: Number(item.stress ?? 0),
      workout: Boolean(item.workout),
      basal_temp: item.basal_temp === null || item.basal_temp === undefined ? null : Number(item.basal_temp),
      resting_hr: item.resting_hr === null || item.resting_hr === undefined ? null : Number(item.resting_hr),
    }))
    .filter((log) => !!log.date)
    .sort((a, b) => a.date.localeCompare(b.date));

  return {
    periodStarts: normalizeDateList(incomingPeriodStarts),
    logs,
  };
}

function parseCsvLogs(raw: string): DailyLog[] {
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((header) => header.trim());

  const find = (row: Record<string, string>, key: string) => row[key] ?? "";

  return lines
    .slice(1)
    .map((line, index) => {
      const cells = line.split(",").map((cell) => cell.trim());
      const row: Record<string, string> = {};
      headers.forEach((header, columnIndex) => {
        row[header] = cells[columnIndex] ?? "";
      });

      const workoutValue = find(row, "workout").toLowerCase();
      return {
        id: `${find(row, "date")}-${index}`,
        date: find(row, "date"),
        mood: Number(find(row, "mood") || 0),
        cramps: Number(find(row, "cramps") || 0),
        fatigue: Number(find(row, "fatigue") || 0),
        headache: Number(find(row, "headache") || 0),
        acne: Number(find(row, "acne") || 0),
        sleep_hours: Number(find(row, "sleep_hours") || 0),
        caffeine_level: Number(find(row, "caffeine_level") || 0),
        stress: Number(find(row, "stress") || 0),
        workout: workoutValue === "1" || workoutValue === "true" || workoutValue === "yes",
        basal_temp: find(row, "basal_temp") ? Number(find(row, "basal_temp")) : null,
        resting_hr: find(row, "resting_hr") ? Number(find(row, "resting_hr")) : null,
      } as DailyLog;
    })
    .filter((log) => !!log.date)
    .sort((a, b) => a.date.localeCompare(b.date));
}

function exportLogsToCsv(logs: DailyLog[]): string {
  const headers = [
    "date",
    "mood",
    "cramps",
    "fatigue",
    "headache",
    "acne",
    "sleep_hours",
    "caffeine_level",
    "stress",
    "workout",
    "basal_temp",
    "resting_hr",
  ];

  const rows = logs.map((log) =>
    [
      log.date,
      log.mood,
      log.cramps,
      log.fatigue,
      log.headache,
      log.acne,
      log.sleep_hours,
      log.caffeine_level,
      log.stress,
      log.workout ? 1 : 0,
      log.basal_temp ?? "",
      log.resting_hr ?? "",
    ].join(","),
  );

  return [headers.join(","), ...rows].join("\n");
}

function calculateBadDayScore(log: DailyLog): number {
  const moodBurden = 5 - log.mood;
  return (
    0.35 * moodBurden +
    0.25 * log.fatigue +
    0.25 * log.cramps +
    0.1 * log.headache +
    0.05 * log.acne
  );
}

function confidenceMeter(daysLogged: number, regularityScore: number): { value: number; label: string } {
  if (daysLogged < 14) return { value: 0.28, label: "Low" };
  if (daysLogged <= 30) return { value: 0.62, label: "Medium" };
  if (regularityScore <= 3) return { value: 0.87, label: "High" };
  return { value: 0.68, label: "Medium" };
}

function riskClass(label: string): string {
  if (label.toLowerCase() === "high") return "risk-high";
  if (label.toLowerCase() === "medium") return "risk-medium";
  return "risk-low";
}

function severityLabel(score: number): string {
  if (score < 2.0) return "Good";
  if (score < 3.5) return "Moderate";
  return "Poor";
}

function featureLabel(name: string): string {
  return FEATURE_LABELS[name] || name.replaceAll("_", " ");
}

function drawGuideCard(logs: DailyLog[], estimatedCycleLength: number, regularityScore: number): GuideCard {
  const contextualCards: GuideCard[] = [];

  if (estimatedCycleLength > 0) {
    contextualCards.push({
      category: "Cycle fact",
      title: "Your current cycle estimate",
      body: `Based on your saved period starts, your estimated cycle length is about ${estimatedCycleLength.toFixed(1)} days.`,
      footer: regularityScore > 0 ? `Current regularity spread: ${regularityScore.toFixed(2)} days.` : undefined,
    });
  }

  if (logs.length) {
    const avgSleep = logs.reduce((sum, row) => sum + row.sleep_hours, 0) / logs.length;
    const avgStress = logs.reduce((sum, row) => sum + row.stress, 0) / logs.length;
    const workoutRate = logs.reduce((sum, row) => sum + (row.workout ? 1 : 0), 0) / logs.length;

    if (avgSleep < 6.8) {
      contextualCards.push({
        category: "Care tip",
        title: "Sleep looks like a live lever",
        body: `Your average logged sleep is ${avgSleep.toFixed(1)} hours. Try stabilizing bedtime for the next week and see whether the harder days soften.`,
      });
    }

    if (avgStress >= 3) {
      contextualCards.push({
        category: "Care tip",
        title: "Stress is running high",
        body: `Your average stress is ${avgStress.toFixed(1)} out of 5. Short breaks, lighter caffeine use, and lower-friction workouts are worth testing around tougher days.`,
      });
    }

    if (workoutRate < 0.35) {
      contextualCards.push({
        category: "Care tip",
        title: "Movement signal is limited",
        body: "You have relatively few workout entries. Even short walks can help you learn whether movement is linked to easier or tougher days for you.",
      });
    }
  }

  const pool = contextualCards.length ? [...contextualCards, ...GUIDE_CARD_LIBRARY] : GUIDE_CARD_LIBRARY;
  return pool[Math.floor(Math.random() * pool.length)];
}

function drawNutritionCard(logs: DailyLog[]): GuideCard {
  const contextualCards: GuideCard[] = [];

  if (logs.length) {
    const avgFatigue = logs.reduce((sum, row) => sum + row.fatigue, 0) / logs.length;
    const avgHeadache = logs.reduce((sum, row) => sum + row.headache, 0) / logs.length;

    if (avgFatigue >= 2.5) {
      contextualCards.push({
        category: "Nutrition tip",
        title: "Fatigue is showing up often",
        body: "Your recent logs show higher fatigue. Regular meals with iron, protein, and hydration are worth prioritizing so you can see whether energy improves across the next cycle.",
      });
    }

    if (avgHeadache >= 1.5) {
      contextualCards.push({
        category: "Nutrition tip",
        title: "Headache days may need a hydration check",
        body: "Frequent headaches can be easier to interpret when you keep hydration, caffeine timing, and meal timing more consistent for a week or two.",
      });
    }
  }

  const pool = contextualCards.length ? [...contextualCards, ...NUTRITION_CARD_LIBRARY] : NUTRITION_CARD_LIBRARY;
  return pool[Math.floor(Math.random() * pool.length)];
}

function SimpleLineChart({
  points,
}: {
  points: Array<{ date: string; actual: number; predicted: number }>;
}) {
  if (!points.length) return null;

  const width = 860;
  const height = 280;
  const padding = 36;
  const minY = 0;
  const maxY = 5;

  const xFor = (index: number) =>
    padding + (index / Math.max(1, points.length - 1)) * (width - padding * 2);
  const yFor = (value: number) =>
    height - padding - ((value - minY) / (maxY - minY)) * (height - padding * 2);

  const actualPath = points.map((point, index) => `${xFor(index)},${yFor(point.actual)}`).join(" ");
  const predictedPath = points
    .map((point, index) => `${xFor(index)},${yFor(point.predicted)}`)
    .join(" ");

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        {[0, 1, 2, 3, 4, 5].map((tick) => (
          <g key={tick}>
            <line x1={padding} x2={width - padding} y1={yFor(tick)} y2={yFor(tick)} className="grid-line" />
            <text x={8} y={yFor(tick) + 4} className="axis-text">
              {tick}
            </text>
          </g>
        ))}
        <polyline points={actualPath} className="line-actual" />
        <polyline points={predictedPath} className="line-predicted" />
      </svg>
      <div className="chart-legend">
        <span className="legend-item">
          <i className="legend-dot actual" /> Actual score
        </span>
        <span className="legend-item">
          <i className="legend-dot predicted" /> Predicted score
        </span>
      </div>
    </div>
  );
}

async function postToApi<T>(path: string, payload: unknown): Promise<T> {
  const baseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = body.detail || response.statusText;
      throw new Error(`API error (${response.status}): ${detail}`);
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(
        "Cannot reach backend service. Ensure FastAPI is running on port 8000 and NEXT_PUBLIC_API_BASE_URL is correct.",
      );
    }
    throw error;
  }
}

export default function HomePage() {
  const [tab, setTab] = useState<Tab>("setup");
  const [showGuideModal, setShowGuideModal] = useState(false);
  const [guideCard, setGuideCard] = useState<GuideCard | null>(null);
  const [periodStarts, setPeriodStarts] = useState<string[]>([]);
  const [nextPeriodStart, setNextPeriodStart] = useState("");
  const [logs, setLogs] = useState<DailyLog[]>([]);
  const [logForm, setLogForm] = useState<Omit<DailyLog, "id">>(defaultLog());
  const [editingLogId, setEditingLogId] = useState<string | null>(null);

  const [statusMessage, setStatusMessage] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isForecasting, setIsForecasting] = useState(false);

  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [forecastResult, setForecastResult] = useState<ForecastResponse | null>(null);
  const [selectedPredictionDate, setSelectedPredictionDate] = useState<string>("");

  useEffect(() => {
    // Remove old persistent data from previous app versions.
    localStorage.removeItem(LEGACY_STORAGE_KEY);

    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.period_starts)) {
        setPeriodStarts(normalizeDateList(parsed.period_starts.map(String)));
      }
      if (Array.isArray(parsed.logs)) {
        const cleanLogs = parsed.logs
          .map((log: any, index: number) => ({
            id: String(log.id || `${log.date || "log"}-${index}`),
            date: String(log.date),
            mood: Number(log.mood),
            cramps: Number(log.cramps),
            fatigue: Number(log.fatigue),
            headache: Number(log.headache),
            acne: Number(log.acne),
            sleep_hours: Number(log.sleep_hours),
            caffeine_level: Number(log.caffeine_level),
            stress: Number(log.stress),
            workout: Boolean(log.workout),
            basal_temp: log.basal_temp === null || log.basal_temp === undefined ? null : Number(log.basal_temp),
            resting_hr: log.resting_hr === null || log.resting_hr === undefined ? null : Number(log.resting_hr),
          }))
          .filter((log: DailyLog) => !!log.date)
          .sort((a: DailyLog, b: DailyLog) => a.date.localeCompare(b.date));
        setLogs(cleanLogs);
      }
    } catch {
      setErrorMessage("Stored data was invalid and could not be loaded.");
    }
  }, []);

  useEffect(() => {
    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        period_starts: periodStarts,
        logs,
      }),
    );
  }, [periodStarts, logs]);

  const stats = useMemo(() => cycleStats(periodStarts), [periodStarts]);

  const daysLogged = logs.length;
  const loggingSpan = useMemo(() => {
    if (logs.length < 2) return { expectedDays: logs.length, missingDays: 0, missingness: 0 };
    const sorted = [...logs].sort((a, b) => a.date.localeCompare(b.date));
    const first = new Date(sorted[0].date).getTime();
    const last = new Date(sorted[sorted.length - 1].date).getTime();
    const expectedDays = Math.round((last - first) / (1000 * 60 * 60 * 24)) + 1;
    const missingDays = Math.max(0, expectedDays - logs.length);
    const missingness = expectedDays ? missingDays / expectedDays : 0;
    return { expectedDays, missingDays, missingness };
  }, [logs]);

  const frontendConfidence = confidenceMeter(daysLogged, stats.regularityScore);
  const openCycleGuideCard = () => {
    setGuideCard(drawGuideCard(logs, stats.estimatedCycleLength, stats.regularityScore));
    setShowGuideModal(true);
  };
  const openNutritionGuideCard = () => {
    setGuideCard(drawNutritionCard(logs));
    setShowGuideModal(true);
  };
  const selectedDayLog = useMemo(
    () => logs.find((item) => item.date === selectedPredictionDate) || null,
    [logs, selectedPredictionDate],
  );

  useEffect(() => {
    if (tab !== "setup") {
      setShowGuideModal(false);
    }
  }, [tab]);

  const addPeriodStart = () => {
    if (!nextPeriodStart) return;
    setPeriodStarts((prev) => normalizeDateList([...prev, nextPeriodStart]));
    setNextPeriodStart("");
  };

  const updatePeriodStart = (index: number, value: string) => {
    setPeriodStarts((prev) => {
      const next = [...prev];
      next[index] = value;
      return normalizeDateList(next);
    });
  };

  const removePeriodStart = (index: number) => {
    setPeriodStarts((prev) => prev.filter((_, idx) => idx !== index));
  };

  const onLogFieldChange = (key: keyof Omit<DailyLog, "id">, value: string | number | boolean | null) => {
    setLogForm((prev) => ({ ...prev, [key]: value }));
  };

  const saveLog = () => {
    if (!logForm.date) {
      setErrorMessage("Date is required.");
      return;
    }

    setErrorMessage("");
    setStatusMessage("");

    setLogs((prev) => {
      const nextEntry: DailyLog = {
        id: editingLogId || (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`),
        ...logForm,
      };

      const updated = editingLogId
        ? prev.map((item) => (item.id === editingLogId ? nextEntry : item))
        : [...prev, nextEntry];

      return updated.sort((a, b) => a.date.localeCompare(b.date));
    });

    setEditingLogId(null);
    setLogForm(defaultLog());
    setStatusMessage("Daily log saved.");
  };

  const editLog = (log: DailyLog) => {
    setEditingLogId(log.id);
    const { id, ...rest } = log;
    setLogForm(rest);
    setTab("daily");
  };

  const deleteLog = (id: string) => {
    setLogs((prev) => prev.filter((item) => item.id !== id));
    if (editingLogId === id) {
      setEditingLogId(null);
      setLogForm(defaultLog());
    }
  };

  const onImportJson = async (file: File) => {
    setErrorMessage("");
    try {
      const raw = await file.text();
      const imported = parseJsonImport(raw);
      setPeriodStarts(imported.periodStarts.length ? imported.periodStarts : inferPeriodStartsFromLogs(imported.logs));
      setLogs(imported.logs);
      setStatusMessage("JSON import completed.");
    } catch {
      setErrorMessage("Invalid JSON file format.");
    }
  };

  const onImportCsv = async (file: File) => {
    setErrorMessage("");
    try {
      const raw = await file.text();
      const importedLogs = parseCsvLogs(raw);
      if (!importedLogs.length) {
        setErrorMessage("CSV had no valid log rows.");
        return;
      }
      setLogs(importedLogs);
      setPeriodStarts((prev) => (prev.length ? prev : inferPeriodStartsFromLogs(importedLogs)));
      setStatusMessage("CSV import completed.");
    } catch {
      setErrorMessage("CSV import failed.");
    }
  };

  const loadSampleCsv = async () => {
    setErrorMessage("");
    try {
      const response = await fetch("/sample_data.csv");
      if (!response.ok) throw new Error("Sample CSV not found");
      const text = await response.text();
      const importedLogs = parseCsvLogs(text);
      setLogs(importedLogs);
      setPeriodStarts(inferPeriodStartsFromLogs(importedLogs));
      setStatusMessage("Loaded sample data.");
    } catch {
      setErrorMessage("Could not load sample data from /sample_data.csv.");
    }
  };

  const exportJson = () => {
    const payload = {
      period_starts: periodStarts,
      logs,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "pms-trigger-finder-data.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const exportCsv = () => {
    const csv = exportLogsToCsv(logs);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "pms-trigger-finder-logs.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const toBackendLogs = (items: DailyLog[]) =>
    items.map((item) => ({
      date: item.date,
      mood: item.mood,
      cramps: item.cramps,
      fatigue: item.fatigue,
      headache: item.headache,
      acne: item.acne,
      sleep_hours: item.sleep_hours,
      caffeine_level: item.caffeine_level,
      stress: item.stress,
      workout: item.workout,
      basal_temp: item.basal_temp,
      resting_hr: item.resting_hr,
    }));

  const trainModel = async () => {
    setErrorMessage("");
    setStatusMessage("");
    setIsTraining(true);
    setPredictResult(null);
    setForecastResult(null);

    try {
      const result = await postToApi<TrainResponse>("/train", {
        period_starts: periodStarts,
        logs: toBackendLogs(logs),
        use_reference_boost: true,
      });
      setTrainResult(result);
      const newestDate = logs.length
        ? [...logs].sort((a, b) => b.date.localeCompare(a.date))[0].date
        : result.predictions[result.predictions.length - 1]?.date || "";
      setSelectedPredictionDate(newestDate);
      setStatusMessage("Model trained successfully.");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Model training failed.");
    } finally {
      setIsTraining(false);
    }
  };

  const predictSelectedDay = async () => {
    setErrorMessage("");
    if (!selectedPredictionDate) {
      setErrorMessage("Pick a date to analyze drivers.");
      return;
    }

    const target = logs.find((item) => item.date === selectedPredictionDate);
    if (!target) {
      setErrorMessage("The selected date is not in your daily logs.");
      return;
    }

    setIsPredicting(true);
    try {
      const historyLogs = logs.filter((item) => item.id !== target.id);
      const result = await postToApi<PredictResponse>("/predict", {
        period_starts: periodStarts,
        history_logs: toBackendLogs(historyLogs),
        target_log: toBackendLogs([target])[0],
      });
      setPredictResult(result);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  };

  const forecastNextWeek = async () => {
    setErrorMessage("");
    setIsForecasting(true);

    const averages = {
      sleep_hours: logs.length ? logs.reduce((sum, item) => sum + item.sleep_hours, 0) / logs.length : 7,
      caffeine_level: logs.length
        ? Math.round(logs.reduce((sum, item) => sum + item.caffeine_level, 0) / logs.length)
        : 1,
      stress: logs.length ? logs.reduce((sum, item) => sum + item.stress, 0) / logs.length : 2,
      workout: logs.length ? logs.reduce((sum, item) => sum + (item.workout ? 1 : 0), 0) / logs.length : 0,
      basal_temp: (() => {
        const values = logs.map((item) => item.basal_temp).filter((value): value is number => value !== null);
        return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 36.5;
      })(),
      resting_hr: (() => {
        const values = logs.map((item) => item.resting_hr).filter((value): value is number => value !== null);
        return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 70;
      })(),
      sleep_prev: logs.length ? logs[logs.length - 1].sleep_hours : 7,
      stress_prev: logs.length ? logs[logs.length - 1].stress : 2,
      score_prev: logs.length ? calculateBadDayScore(logs[logs.length - 1]) : 2.5,
    };

    try {
      const result = await postToApi<ForecastResponse>("/forecast", {
        period_starts: periodStarts,
        last_period_start: periodStarts.length ? periodStarts[periodStarts.length - 1] : undefined,
        estimated_cycle_length: stats.estimatedCycleLength,
        typical_features: averages,
        start_date: new Date().toISOString().slice(0, 10),
      });
      setForecastResult(result);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Forecast failed.");
    } finally {
      setIsForecasting(false);
    }
  };

  return (
    <main className="page-shell">
      <section className="page-header">
        <div>
          <h1>PMS Mood Compass</h1>
          <p>Spot tougher cycle days before they sneak up on you.</p>
          <p className="tiny muted">
            Log mood, symptoms, sleep, stress, and cycle timing to learn what patterns are shaping each month.
          </p>
        </div>
        <img
          src="/womens-health-cartoon.svg"
          alt="Women’s mood and cycle themed illustration"
          className="hero-illustration"
        />
      </section>

      {showGuideModal && guideCard && (
        <div className="modal-backdrop" onClick={() => setShowGuideModal(false)}>
          <section className="guide-modal" onClick={(event) => event.stopPropagation()}>
            <p className="guide-kicker">{guideCard.category}</p>
            <h3>{guideCard.title}</h3>
            <p>{guideCard.body}</p>
            {guideCard.footer && <p className="tiny muted">{guideCard.footer}</p>}
            <div className="inline-row wrap">
              <button
                className="primary"
                onClick={guideCard.category === "Nutrition tip" ? openNutritionGuideCard : openCycleGuideCard}
              >
                Another one
              </button>
              <button className="subtle" onClick={() => setShowGuideModal(false)}>
                Close
              </button>
            </div>
          </section>
        </div>
      )}

      <nav className="tabs">
        <button className={tab === "setup" ? "tab active" : "tab"} onClick={() => setTab("setup")}>
          Setup
        </button>
        <button className={tab === "daily" ? "tab active" : "tab"} onClick={() => setTab("daily")}>
          Daily Log
        </button>
        <button className={tab === "insights" ? "tab active" : "tab"} onClick={() => setTab("insights")}>
          Insights
        </button>
      </nav>

      {(statusMessage || errorMessage) && (
        <section className="status-box">
          {statusMessage && <p className="ok-msg">{statusMessage}</p>}
          {errorMessage && <p className="error-msg">{errorMessage}</p>}
        </section>
      )}

      {tab === "setup" && (
        <section className="card">
          <h2>Cycle Setup</h2>
          <p className="muted">Enter at least two recent period start dates to improve cycle-aware features.</p>
          <div className="inline-row">
            <input
              type="date"
              value={nextPeriodStart}
              onChange={(event) => setNextPeriodStart(event.target.value)}
            />
            <button className="primary" onClick={addPeriodStart}>
              Add Date
            </button>
          </div>

          <div className="period-list">
            {periodStarts.length === 0 && <p className="muted">No period dates yet.</p>}
            {periodStarts.map((periodStart, index) => (
              <div className="period-item" key={`${periodStart}-${index}`}>
                <input
                  type="date"
                  value={periodStart}
                  onChange={(event) => updatePeriodStart(index, event.target.value)}
                />
                <button className="danger" onClick={() => removePeriodStart(index)}>
                  Delete
                </button>
              </div>
            ))}
          </div>

          <div className="stat-grid">
            <article>
              <h3>Estimated Cycle Length</h3>
              <p>{stats.estimatedCycleLength.toFixed(1)} days</p>
            </article>
            <article>
              <h3>Cycle Regularity (STD)</h3>
              <p>{stats.regularityScore.toFixed(2)} days</p>
            </article>
            <article>
              <h3>Recent Cycle Lengths</h3>
              <p>{stats.cycleLengths.length ? stats.cycleLengths.join(", ") : "Not enough data"}</p>
            </article>
          </div>

          <div className="inline-row wrap">
            <button className="subtle" onClick={openCycleGuideCard}>
              Get a Cycle Tip
            </button>
            <button className="subtle" onClick={openNutritionGuideCard}>
              Get a Nutrition Tip
            </button>
          </div>
          <p className="tiny muted">Each button opens one fresh card at a time.</p>
        </section>
      )}

      {tab === "daily" && (
        <section className="card">
          <h2>Daily Log</h2>
          <p className="tiny muted">
            Mood scale: <strong>0 = very low mood</strong>, <strong>5 = very good mood</strong>. Symptom scales use
            <strong> 0 = none</strong> and <strong>5 = severe</strong>.
          </p>
          <div className="form-grid">
            <label>
              Date
              <input
                type="date"
                value={logForm.date}
                onChange={(event) => onLogFieldChange("date", event.target.value)}
              />
            </label>

            {(
              [
                ["Mood (0 low, 5 good)", "mood"],
                ["Cramps", "cramps"],
                ["Fatigue", "fatigue"],
                ["Headache", "headache"],
                ["Acne", "acne"],
                ["Stress", "stress"],
              ] as Array<[string, keyof Omit<DailyLog, "id">]>
            ).map(([label, field]) => (
              <label key={field}>
                {label}: {Number(logForm[field]).toFixed(0)}
                <input
                  type="range"
                  min={0}
                  max={5}
                  step={1}
                  value={Number(logForm[field])}
                  onChange={(event) => onLogFieldChange(field, Number(event.target.value))}
                />
              </label>
            ))}

            <label>
              Sleep Hours
              <input
                type="number"
                min={0}
                max={24}
                step={0.5}
                value={logForm.sleep_hours}
                onChange={(event) => onLogFieldChange("sleep_hours", Number(event.target.value))}
              />
            </label>

            <label>
              Caffeine
              <select
                value={logForm.caffeine_level}
                onChange={(event) => onLogFieldChange("caffeine_level", Number(event.target.value))}
              >
                <option value={0}>0</option>
                <option value={1}>1</option>
                <option value={2}>2+</option>
              </select>
            </label>

            <label className="toggle-row">
              Workout
              <input
                type="checkbox"
                checked={logForm.workout}
                onChange={(event) => onLogFieldChange("workout", event.target.checked)}
              />
            </label>

            <label>
              Basal Temp (optional)
              <input
                type="number"
                step={0.01}
                value={logForm.basal_temp ?? ""}
                onChange={(event) =>
                  onLogFieldChange("basal_temp", event.target.value === "" ? null : Number(event.target.value))
                }
              />
            </label>

            <label>
              Resting HR (optional)
              <input
                type="number"
                step={1}
                value={logForm.resting_hr ?? ""}
                onChange={(event) =>
                  onLogFieldChange("resting_hr", event.target.value === "" ? null : Number(event.target.value))
                }
              />
            </label>
          </div>

          <div className="inline-row">
            <button className="primary" onClick={saveLog}>
              {editingLogId ? "Update Entry" : "Save Entry"}
            </button>
            {editingLogId && (
              <button
                className="subtle"
                onClick={() => {
                  setEditingLogId(null);
                  setLogForm(defaultLog());
                }}
              >
                Cancel Edit
              </button>
            )}
          </div>

          <div className="inline-row wrap">
            <label className="file-btn">
              Import JSON
              <input
                type="file"
                accept="application/json,.json"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void onImportJson(file);
                  event.currentTarget.value = "";
                }}
              />
            </label>

            <label className="file-btn">
              Import CSV
              <input
                type="file"
                accept="text/csv,.csv"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void onImportCsv(file);
                  event.currentTarget.value = "";
                }}
              />
            </label>

            <button className="subtle" onClick={exportJson}>
              Export JSON
            </button>
            <button className="subtle" onClick={exportCsv}>
              Export CSV
            </button>
            <button className="subtle" onClick={() => void loadSampleCsv()}>
              Load Demo CSV
            </button>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Mood</th>
                  <th>Cramps</th>
                  <th>Fatigue</th>
                  <th>Stress</th>
                  <th>Sleep</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {logs.length === 0 && (
                  <tr>
                    <td colSpan={7} className="muted center">
                      No daily logs yet.
                    </td>
                  </tr>
                )}
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td>{log.date}</td>
                    <td>{log.mood}</td>
                    <td>{log.cramps}</td>
                    <td>{log.fatigue}</td>
                    <td>{log.stress}</td>
                    <td>{log.sleep_hours}</td>
                    <td className="inline-row">
                      <button className="subtle" onClick={() => editLog(log)}>
                        Edit
                      </button>
                      <button className="danger" onClick={() => deleteLog(log.id)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {tab === "insights" && (
        <section className="card">
          <h2>Insights</h2>
          <p className="muted">
            This view shows what the model learned from the logs currently loaded in this session. Not medical advice.
          </p>

          <div className="stat-grid">
            <article>
              <h3>Days Logged</h3>
              <p>{daysLogged}</p>
            </article>
            <article>
              <h3>Missingness</h3>
              <p>
                {(loggingSpan.missingness * 100).toFixed(1)}% ({loggingSpan.missingDays} of {loggingSpan.expectedDays} days)
              </p>
            </article>
            <article>
              <h3>Frontend Confidence Meter</h3>
              <p>{frontendConfidence.label}</p>
              <div className="meter">
                <div className="meter-fill" style={{ width: `${frontendConfidence.value * 100}%` }} />
              </div>
            </article>
          </div>

          {daysLogged < 14 && (
            <p className="muted warning-note">
              You have limited history. Add at least 2-4 full cycle weeks for more stable predictor importance and
              forecasts.
            </p>
          )}

          <div className="inline-row wrap">
            <button className="primary" disabled={isTraining} onClick={() => void trainModel()}>
              {isTraining ? "Training..." : "Train / Refresh Model"}
            </button>
            <button className="subtle" disabled={isForecasting || !trainResult} onClick={() => void forecastNextWeek()}>
              {isForecasting ? "Forecasting..." : "Forecast Next 7 Days"}
            </button>
          </div>
          <p className="tiny muted">
            Training rebuilds the model using the logs currently loaded in this browser session.
          </p>

          {trainResult && (
            <>
              <div className="stat-grid">
                <article>
                  <h3>Chosen Model</h3>
                  <p>{trainResult.metrics.model_name}</p>
                </article>
                <article>
                  <h3>Prediction Error</h3>
                  <p>
                    {trainResult.metrics.mae.toFixed(2)} avg / {trainResult.metrics.rmse.toFixed(2)} spread
                  </p>
                </article>
                <article>
                  <h3>Simple Average Error</h3>
                  <p>
                    {trainResult.metrics.baseline_mae.toFixed(2)} avg / {trainResult.metrics.baseline_rmse.toFixed(2)} spread
                  </p>
                </article>
                <article>
                  <h3>Backend Confidence</h3>
                  <p>
                    {trainResult.confidence.label} ({Math.round(trainResult.confidence.value * 100)}%)
                  </p>
                </article>
              </div>

              {trainResult.data_quality && (
                <p className="muted tiny">
                  Rows used: {trainResult.data_quality.rows_used}.{" "}
                  {trainResult.data_quality.low_variance_features.length > 0 && (
                    <>
                      Low-variation fields:{" "}
                      {trainResult.data_quality.low_variance_features.slice(0, 4).map(featureLabel).join(", ")}.
                    </>
                  )}
                </p>
              )}
              {trainResult.training_summary && <p className="tiny muted">{trainResult.training_summary}</p>}
              {trainResult.model_choice_reason && <p className="tiny muted">{trainResult.model_choice_reason}</p>}
              {trainResult.next_step_hint && <p className="tiny muted">{trainResult.next_step_hint}</p>}

              <article className="top-predictors">
                <h3>Main Pattern Signals</h3>
                <ul>
                  {trainResult.global_importance.slice(0, 3).map((item) => (
                    <li key={item.feature}>
                      <strong>{item.label || featureLabel(item.feature)}</strong>:{" "}
                      {((item.importance_pct ?? 0) as number).toFixed(2)}%
                    </li>
                  ))}
                </ul>
              </article>

              <SimpleLineChart points={trainResult.predictions} />

              <div className="driver-box">
                <h3>Top Drivers Today</h3>
                <p className="muted">
                  Pick a logged day to compare the recorded symptom load with the model's estimate from non-symptom inputs.
                </p>
                <div className="inline-row wrap">
                  <select
                    value={selectedPredictionDate}
                    onChange={(event) => setSelectedPredictionDate(event.target.value)}
                    disabled={!logs.length}
                  >
                    <option value="">Select day</option>
                    {[...logs]
                      .sort((a, b) => b.date.localeCompare(a.date))
                      .map((log) => (
                        <option key={log.id} value={log.date}>
                          {log.date}
                        </option>
                      ))}
                  </select>
                  <button className="subtle" disabled={isPredicting || !selectedPredictionDate} onClick={() => void predictSelectedDay()}>
                    {isPredicting ? "Explaining..." : "Explain Selected Day"}
                  </button>
                </div>

                {predictResult && (
                  <>
                    <p className="score-summary">
                      <strong>Logged symptom load</strong>
                    </p>
                    {predictResult.actual_logged_score !== undefined && (
                      <p>
                        <strong>{predictResult.actual_logged_score.toFixed(2)}/5</strong>{" "}
                        {predictResult.actual_status_emoji || ""}
                        {" "}
                        {predictResult.actual_status_label ||
                          severityLabel(predictResult.actual_logged_score)}
                      </p>
                    )}
                    <p>
                      Model expectation from cycle and lifestyle signals: <strong>{predictResult.prediction.toFixed(2)}/5</strong>{" "}
                      ({predictResult.predicted_status_label || predictResult.status_label || severityLabel(predictResult.prediction)}).
                      Confidence: {predictResult.confidence.label} ({Math.round(predictResult.confidence.value * 100)}%).
                    </p>
                    {predictResult.expected_gap !== undefined && (
                      <p className="tiny muted">
                        Difference between logged load and model expectation: {predictResult.expected_gap > 0 ? "+" : ""}
                        {predictResult.expected_gap.toFixed(2)} points.
                      </p>
                    )}
                    <p className="tiny muted">
                      Logged score uses only mood, cramps, fatigue, headache, and acne. Sleep, stress, cycle timing,
                      caffeine, workout, prior-day history, and optional sensors help the model estimate that score.
                    </p>
                    {predictResult.interpretation && <p className="tiny muted">{predictResult.interpretation}</p>}
                    <p className="tiny muted">These are the main factors the model used when it formed that expectation:</p>
                    <ul>
                      {(predictResult.driver_summaries && predictResult.driver_summaries.length > 0
                        ? predictResult.driver_summaries
                        : predictResult.drivers.slice(0, 4).map((driver) =>
                            `${driver.label || featureLabel(driver.feature)}: ${driver.contribution.toFixed(2)} (${driver.direction})`,
                          )
                      ).map((line) => (
                        <li key={line}>{line}</li>
                      ))}
                    </ul>
                    <details>
                      <summary>Show technical contribution values</summary>
                      <ul>
                        {predictResult.drivers.slice(0, 6).map((driver) => (
                        <li key={driver.feature}>
                          <strong>{driver.label || featureLabel(driver.feature)}</strong>:{" "}
                          {driver.contribution > 0 ? "+" : ""}
                          {driver.contribution.toFixed(2)} ({driver.direction})
                        </li>
                      ))}
                      </ul>
                    </details>
                    {selectedDayLog && (
                      <p className="tiny muted">
                        Day snapshot: mood {selectedDayLog.mood}, cramps {selectedDayLog.cramps}, fatigue{" "}
                        {selectedDayLog.fatigue}, stress {selectedDayLog.stress}, sleep {selectedDayLog.sleep_hours}h.
                      </p>
                    )}
                  </>
                )}
              </div>
            </>
          )}

          {forecastResult && (
            <article className="forecast-box">
              <h3>Next 7 Days Forecast</h3>
              <p className="muted">Lower score means an easier day. These are pattern estimates, not medical advice.</p>
              <div className="forecast-grid">
                {forecastResult.forecast.map((item) => (
                  <div className={`forecast-item ${riskClass(item.risk_label)}`} key={item.date}>
                    <p>{item.date}</p>
                    <strong>{item.predicted_bad_day_score.toFixed(2)}/5</strong>
                    <span>
                      {item.status_emoji || ""} {item.status_label || item.risk_label}
                    </span>
                  </div>
                ))}
              </div>
            </article>
          )}
        </section>
      )}

      <footer className="footer-note">Not medical advice. Use this app for informational pattern tracking only.</footer>
    </main>
  );
}
