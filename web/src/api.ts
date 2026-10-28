// Thin client. The UI only displays numbers returned by the backend; it never re-scores anything.
async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}
export const api = {
  meta: () => fetch("/api/meta").then(j<any>),
  case: (id: string) => fetch(`/api/case/${id}`).then(j<any>),
  score: (body: { case_id: string; model_id: string; removed: number[]; A_index?: number; B_index?: number }) =>
    fetch("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j<any>),
  explain: (body: { case_id: string; A_index: number; B_index: number; budget: number }) =>
    fetch("/api/explain", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j<any>),
  motifs: (id: string, A: number, B: number) => fetch(`/api/motifs/${id}?A=${A}&B=${B}`).then(j<any>),
  investigation: (id: string) => fetch(`/api/investigation/${id}`).then(j<any>),
  runInvestigation: () => fetch("/api/investigation/run", { method: "POST" }).then(j<any>),
  evalSummary: () => fetch("/api/eval").then(j<any>),
  negation: () => fetch("/api/diagnostics/negation").then(j<any>),
};

export const f = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : (x > 0 ? "+" : "") + x.toFixed(d);
export const n = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? "–" : x.toFixed(d);
export const dt = (ts: number) => new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 16) + " UTC";
