// Plain-English names for the ranker variants. Internal ids stay in code and data only.
export const NAME: Record<string, string> = {
  F0: "Current ranker",
  F1: "Rely less on viewing history for specific searches",
  Fctrl: "Rely less on viewing history for every search",
  F2: "Rely less on viewing history for new members",
  F3: "Understand what the request rules out",
  F4: "Rely less on popularity for specific searches",
  F5: "Ignore each title's latest ratings from strangers",
  "F4+F5": "Less popularity + ignore strangers' latest ratings",
};
export const SHORT: Record<string, string> = {
  F0: "current", F1: "less history (specific searches)", Fctrl: "less history (all searches)", F2: "less history for new members",
  F3: "understand the exclusion", F4: "less popularity", F5: "ignore strangers' ratings", "F4+F5": "less popularity + ignore strangers",
};
export const nm = (id: string) => NAME[id] ?? id;
export const sn = (id: string) => SHORT[id] ?? id;
