import { f, n } from "./api";

// One CATGX motif, drawn on its own: the concrete instance (real people / movies / dates) laid out left-to-right in
// time order, plus a plain-English account of what the structure is and why a temporal graph model is affected by it.
// The numbers (lift, S, count) come from /api/catgx motif_cards.

const PLAIN: Record<string, { title: string; why: string }> = {
  BRIDGE: {
    title: "A taste path that leads to A",
    why: "A graph model learns what a movie is from the people who watched it. Here someone who shares a movie with another viewer later watches A, so A starts to look like “what people with this taste go on to watch”. Anyone near that crowd gets A pulled up, whatever they typed.",
  },
  BRIDGEB: {
    title: "A taste path that leads to B",
    why: "The same mechanism, but ending at B, the better match. This is the evidence that should have helped B; when it matters less than the paths to A, A wins.",
  },
  CO: {
    title: "Two people rated the same movie",
    why: "Rating the same movie makes two people look alike to the model, so each person's other movies start to count as evidence for the other.",
  },
  SEQ: {
    title: "One person rated two movies in a row",
    why: "The model reads a person's recent run of ratings as their current taste, so movies that look like that run get a boost.",
  },
  POP: {
    title: "Several people rated A",
    why: "Several recent ratings make A look in demand around this person, which raises its score regardless of the search.",
  },
};

const day = (ts: number) => new Date(ts * 1000).toISOString().slice(0, 10);
const short = (s: string, k = 16) => (s.length > k ? s.slice(0, k - 1) + "…" : s);

export default function MotifCard({ card, meta, sg, color, compact = false }: any) {
  const P = PLAIN[card.motif] ?? { title: card.name, why: "" };
  const ex = card.example;
  const name = (v: number) => {
    if (sg && v === sg.u) return "this person (u)";
    if (sg && v === sg.A) return `A · ${short(sg.A_title, 14)}`;
    if (sg && v === sg.B) return `B · ${short(sg.B_title, 14)}`;
    const m = meta?.[String(v)];
    return m ? (m.type === "user" ? `user ${v}` : short(m.title)) : `node ${v}`;
  };
  const isUser = (v: number) => (meta?.[String(v)]?.type ?? (sg && v === sg.u ? "user" : "movie")) === "user";
  // path layout: nodes in their sigma order, edges labelled with the rating date
  const W = 400, H = 92, nN = ex.nodes.length, gap = (W - 110) / Math.max(nN - 1, 1);
  const xs = ex.nodes.map((_: number, i: number) => 55 + i * gap);
  // edges connect consecutive nodes for SEQ/POP differently: SEQ = (x, m1, m2) both edges from x; POP = (w1, w2, A) both edges into A
  const pairs: [number, number][] = card.motif === "SEQ" ? [[0, 1], [0, 2]] : card.motif === "POP" ? [[0, 2], [1, 2]] : card.motif === "CO" ? [[0, 1], [2, 1]] : [[0, 1], [2, 1], [2, 3]];
  const lift = card.lift;
  return (
    <div className={`motif ${compact ? "compact" : ""}`}>
      <div className="mh"><span className="mdot" style={{ background: color }} /><b>{P.title}</b>
        {lift != null && <em className={lift >= 1 ? "up" : "down"}>{lift >= 1 ? `${n(lift, 1)}× more weight` : `${n(lift, 1)}× (less weight)`}</em>}</div>
      {!compact && <svg viewBox={`0 0 ${W} ${H}`} className="msvg">
        {pairs.map(([i, j], k) => {
          const y = card.motif === "SEQ" || card.motif === "POP" ? (k === 0 ? 36 : 36) : 36;
          const curve = (card.motif === "SEQ" && k === 1) || (card.motif === "POP" && k === 0);
          const x1 = xs[i], x2 = xs[j];
          const d = curve ? `M${x1},${y} Q${(x1 + x2) / 2},${y - 34} ${x2},${y}` : `M${x1},${y} L${x2},${y}`;
          return <g key={k}><path d={d} stroke={color} strokeWidth={3} fill="none" />
            <text x={(x1 + x2) / 2} y={curve ? y - 20 : y - 6} textAnchor="middle" className="mdate">{day(ex.ts[k])}</text></g>;
        })}
        {ex.nodes.map((v: number, i: number) => <g key={i}>
          {isUser(v) ? <circle cx={xs[i]} cy={36} r={9} fill={sg && v === sg.u ? "#dce9f7" : "#eef2f7"} stroke="#7194b9" strokeWidth={2} />
            : <rect x={xs[i] - 9} y={27} width={18} height={18} rx={4} fill={sg && v === sg.A ? "#c82434" : sg && v === sg.B ? "#4f8f6e" : "#c5a353"} />}
          <text x={xs[i]} y={66} textAnchor="middle" className="mnode">{name(v)}</text>
        </g>)}
      </svg>}
      <p className="mstory">{story(card.motif, ex, name)}</p>
      {!compact && <p className="mwhy"><b>Why it moves the model:</b> {P.why}</p>}
      {lift != null && <p className="cap">Compared with other structures in this subgraph that are just as recent and involve equally active people and movies, the model puts {n(lift, 1)}× as much weight on this kind of structure (S = {f(card.S, 3)}; about {Math.round(card.est_count)} instances){ex.involves_u ? "" : ". Strongest instance shown"}.</p>}
    </div>
  );
}

function story(m: string, ex: any, name: (v: number) => string) {
  const [a, b, c, d] = ex.nodes;
  const t = (i: number) => day(ex.ts[i]);
  if (m === "BRIDGE" || m === "BRIDGEB")
    return <>{name(a)} rated <b>{name(b)}</b> ({t(0)}); {name(c)} also rated it ({t(1)}) and later rated <b>{name(d)}</b> ({t(2)}).</>;
  if (m === "CO") return <>{name(a)} ({t(0)}) and {name(c)} ({t(1)}) both rated <b>{name(b)}</b>.</>;
  if (m === "SEQ") return <>{name(a)} rated <b>{name(b)}</b> ({t(0)}) and then <b>{name(c)}</b> ({t(1)}).</>;
  if (m === "POP") return <>{name(a)} ({t(0)}) and {name(b)} ({t(1)}) both rated <b>{name(c)}</b>.</>;
  return null;
}
