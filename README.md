# Search Ranking Investigation Lab

**Find out why a search ranker got a result wrong, prove it, and pick a fix that holds on searches it never saw.**

This is a working prototype built on public data (MovieLens). It pairs a real search ranker, whose history signal comes from a temporal graph model, with two graph explainers from our research:

* **TemGX** (ICLR 2026) finds the few past interactions that drive a prediction.
* **CATGX** (KDD 2026) names the pattern those interactions form, after adjusting for recency and activity.

Around them are a deletion test that checks every explanation, a store of verified diagnoses that retrieval and an LLM draw on, and checks generated from the explanations that any model change can be scored against.

Everything shown is computed by the models in this repository. Nothing uses Netflix data, code or metrics, and nothing here measures production impact or A/B lift.

---

## Contents

1. [Start the demo](#1-start-the-demo)
2. [The 5-minute demo: one case, four steps](#2-the-5-minute-demo-one-case-four-steps)
3. [Why this matters for a search team](#3-why-this-matters-for-a-search-team)
4. [What is inside](#4-what-is-inside)
5. [Key results](#5-key-results-all-measured-held-out-where-stated)
6. [What we do not claim](#6-what-we-do-not-claim)
7. [The other tabs](#7-the-other-tabs)
8. [Rebuild from scratch](#8-rebuild-from-scratch)
9. [Repository layout](#9-repository-layout)
10. [HTTP API](#10-http-api)
11. [Tests](#11-tests)
12. [Further documents](#12-further-documents)
13. [Licences and sources](#13-licences-and-sources)

---

## 1. Start the demo

All checkpoints, cached results and the built web page are already in the repository, so the demo runs offline.

```bash
cd ~/Desktop/netflix_demo/search-ranking-investigation-lab
.venv/bin/python -m srlab.server
```

Open **http://127.0.0.1:8000**. The page opens on **▶ Case study**, member 290. Click steps 1 → 4.

| situation | what to do |
|---|---|
| "address already in use" or the page still looks old | stop the old server (Ctrl+C in its terminal, or `lsof -ti:8000 \| xargs kill`), then start again |
| want another port | `PORT=8001 .venv/bin/python -m srlab.server`, open http://127.0.0.1:8001 |
| page didn't update after a change | hard refresh (Cmd+Shift+R) |
| a banner says the live ranking did not load | the running server is older than the page: restart it |
| first load is slow | it loads the graph model on the first request (a few seconds) |

Tip: before presenting, click through all four steps once so the results are cached.

---

## 2. The 5-minute demo: one case, four steps

**The case.** A member asks *"a drama, but please nothing depressing"*. The ranker puts *Requiem for a Dream* first, and two of the top three are titles the member ruled out. The right answer, *Out of Sight (1998)*, is #2.

**Why this case.** It was picked by a rule fixed before looking at the results (`srlab/bestcase.py`). We kept only searches where the whole loop closes:

* the evidence holds on both sides: removing it beats removing random ratings, p ≤ 0.05;
* the fix recommended by retrieval works for this member live: the right title goes to #1 and the ruled-out title leaves the top 3.

29 of 32 searches pass the evidence checks; **1 of 32 passes everything: member 290**. Every step also shows the numbers across all searches, so the case is never the only evidence.

Left side of every step: the member's real neighbourhood in the rating graph. Click any rating to remove it, and the ranking updates live.

### Step 1 · Problem (*what went wrong*)

* **What you see:** the request, the live top 3 (ruled-out titles marked red), and how common this is.
* **Numbers:** 32 of 106 members asking this got Requiem first. 53% of chat-style requests start with a wrong title, against 18% of regular searches.
* **Takeaway:** not one unlucky member; the ranker has a pattern.

### Step 2 · Evidence (*why it happened*) · TemGX + CATGX

* **What the ranker added up:** three scores (matches the words / popular / fits their history). Requiem wins mainly by being popular.
* **What in the graph pushes each title up:** two cards side by side.
  * TemGX picks the ratings that move the graph model most. For Requiem: one recent fan of the film and two of the member's own ratings.
  * CATGX names the pattern, e.g. *"Recent fans piled on"*: the model leans on it 5.5× more than on similar patterns. It is drawn as its shape (triangle or quadrilateral).
* **In plain words:** one LLM sentence, written from this evidence only.
* **Controls:** how many ratings TemGX shows (1–8); "see every pattern CATGX found".

### Step 3 · Verification (*is that really why?*)

Remove the ratings and ask the model again. Each bar shows how sure the model is that Requiem beats Out of Sight.

| removed | Requiem over Out of Sight |
|---|---|
| nothing | 87% |
| the 3 ratings TemGX found | **78%** |
| the pattern CATGX found | 85% |
| what pushes Out of Sight up | 94% (should rise, does) |
| the 3 newest ratings of Requiem (what a standard GraphRAG would fetch) | 85% |
| the 3 the model pays most attention to | 90% |
| 3 random ratings from the same weeks | 87% |

Click a bar to see that removal on the graph and in the ranking.

Across all 32 searches:
* TemGX beats random ratings in 32/32.
* It moves the model more than the "newest ratings" GraphRAG pick in 22/32, and more than attention in 29/32.
* When we planted a known cause, TemGX recovered 97% of it; an LLM reading the history recovered 43%.

### Step 4 · Decision (*what to ship*)

* **Try a fix on this member, live:** current ranker, *Understand "not" in the request*, *Rely less on popularity for specific searches*, *Ignore each title's latest ratings from strangers*, or a combination.
* **Have we seen this before? (RAG over verified explanations)**
  1. **Find:** the 3 most similar of 343 past failures that were already explained and checked.
  2. **Collect:** what fixed each of them.
  3. **Write:** an LLM sums it up from that evidence only; every number and citation is checked automatically.
  * The lookup recommends *Understand "not"* (fixed 8 of the 10 most similar). Click **Try**: Requiem drops to #7 and Out of Sight is #1.
* **Before the A/B test: what the team would know**

  | | |
  |---|---|
  | 53% | where it fails (chat-style requests, vs 18% regular) |
  | 32/32 | why, checked (the reason really moves the model) |
  | 65% vs 46% | the fix holds on 576 new requests (ruled-out title kept out of the top 3); 1,726 such checks stay as a regression suite |

  Any model change, by a person or a coding agent, can be scored on these checks before an A/B test. **We have not run an agent on it.**
* **more:** every fix on every check, and the fix the graph explanation itself suggested (held-out table).

The spoken script with timings is at the top of `docs/DEMO_SCRIPT.md`.

---

## 3. Why this matters for a search team

| what a search team deals with | where the demo answers it |
|---|---|
| chat or voice search requests, including "not this" constraints | step 1: chat-style requests fail far more often (53% vs 18%) |
| evaluation mostly through A/B tests, with little analysis of *why* a failure happens | steps 2–3: the reason for each failure, checked by removing it |
| rankers that mix popularity, viewing history and text match | step 2: the score split, and which history is behind the history score |
| choosing which change is worth an A/B test | step 4: the fix is checked on held-out requests first |
| evolving the model, possibly with a coding agent | step 4: checks generated from explanations can score any proposed change (agent not run) |
| new members with little history | "more" in step 4: new-member results; *Rely less on viewing history for new members* |

Public context: Netflix has tested natural-language search where members describe a mood (BGR / Bloomberg, April 2025). Netflix ranks search and recommendations with one model that takes a user model learned from viewing history (UniCoRn, QCon London 2025). Sources are in section 13.

---

## 4. What is inside

### 4.1 Data

MovieLens `ml-latest-small`: 100,836 ratings, 610 members, 9,742 titles. It is turned into a time-stamped member–title graph and split by time:

* train < 2013-08-18;
* development (DEV) < 2016-11-19;
* held-out (FINAL) after that.

Search requests are written from templates, with 30 candidates each. Labels come from a rubric plus the member's later ratings. Request types:

* regular searches;
* chat-style requests with exclusions ("…but nothing depressing");
* new members (1–4 ratings).

DEV and FINAL use different request templates.

### 4.2 Graph models (the ranker's history signal)

| name in code | what it is |
|---|---|
| `tgat1` | TGAT, 1 layer (inside the original evaluation ranker) |
| `tgat2` | TGAT, 2 layers (used by the case study and the live ranker) |
| `tgn` | TGN with memory |

The models are trained on train data only. Their output is calibrated to *P(A beats B)* with a temperature.

### 4.3 The ranker and the fixes

The ranker adds three standardised scores: **fits their history** (graph model), **matches the words** (text match), and **popular** (interactions before the search time). The weights are fit on DEV.

| plain-English name (shown in the UI) | id in code | what changes |
|---|---|---|
| Current ranker | `F0` | nothing |
| Rely less on viewing history for specific searches | `F1` | earlier attempt; rejected (hurts specific searches) |
| Rely less on viewing history for every search | `Fctrl` | control for F1 |
| Rely less on viewing history for new members | `F2` | history weight depends on history length |
| Understand "not" in the request | `F3` | text match reads negations (β = 0.568, fit on DEV) |
| Rely less on popularity for specific searches | `F4` | popularity weight depends on how specific the request is |
| Ignore each title's latest ratings from strangers | `F5` | drop each candidate's 10 most recent ratings by other people before the history score (n chosen on DEV); suggested by the explanations |
| Less popularity + ignore strangers' latest ratings | `F4+F5` | both |

Plain-English names live in one file: `web/src/names.ts`.

### 4.4 Explainers

* **TemGX** (`srlab/temgx.py`):
  * scores each interaction by replay, influence along time-respecting paths, effective-resistance relevance and time decay;
  * then selects, verifies and replaces greedily.
  * No training.
* **CATGX** (`srlab/catgx.py`):
  * motif library (recent fans piled on, shared movie, run of ratings, shared-taste path);
  * HNSW retrieval with stratified sampling of comparable patterns (same recency and activity);
  * an adjusted score and a knapsack edge budget.
  * The case study uses the replay signal (`g_mode="do"`) and keeps only positive patterns.
* **Joint use** (`srlab/joint.py`): each TemGX rating is labelled with the CATGX pattern that contains it.
* **Both sides** (`srlab/contrast.py`): explains "why the wrong title" and "why the right title", and compares them.

### 4.5 Verification

Remove the explanation's ratings, re-run the graph model, and compare with random ratings from the same recency quartiles (40 draws) to get a p-value.

`srlab/retrievers.py` runs the same test for GraphRAG-style picks at equal size (newest ratings around the title or member, attention).

### 4.6 LLM layer

* The LLM only reads evidence and writes plain sentences (`srlab/structure.py`, `srlab/rag.py`).
* Every number it writes must appear in the evidence, and every citation must exist. This is checked automatically.
* Answers were pre-generated with Claude Sonnet. If `ANTHROPIC_API_KEY` is set, they are generated live.

### 4.7 RAG over verified explanations

* **Store** (`srlab/xstore.py`): 343 DEV failures. Each is explained on both sides, checked by deletion, and records which fixes worked when re-ranked live.
* **Index:** HNSW over a mechanism key.
* **Recommender:** majority vote of the nearest past failures. Features and k are chosen by DEV leave-one-out, then scored once on 280 FINAL failures.
* **Writing** (`srlab/rag.py`): retrieved cases plus this search's evidence go on one sheet, and the LLM writes the diagnosis with citations.
* **Graph-walk retrieval** (`srlab/graphrag.py`): personalized PageRank over a graph of diagnoses, tested as an alternative.

### 4.8 Checks generated from explanations

`srlab/testgen.py` writes 1,726 checks on 576 held-out chat requests:

* **T1:** the ruled-out title is not in the top 3.
* **T2:** the right title is above the ruled-out one.
* **T3:** the order holds without the ruled-out title's 3 recent ratings by strangers.

It also writes 282 labelled pairs from DEV, which have not yet been used for training.

---

## 5. Key results (all measured; held-out where stated)

| claim | number |
|---|---|
| Chat-style requests start with a wrong title | 53% (334/636) vs 18% regular, 20% new members |
| Requiem first for "a drama, but please nothing depressing" | 32 of 106 searches |
| Removing TemGX's ratings beats random (case-study searches) | 32/32 |
| TemGX vs GraphRAG-style "newest ratings around the title", same size | moves the model more in 22/32 (mean drop 5.4 vs 2.9 pts) |
| TemGX vs attention top-3 | more in 29/32 (attention: −1.7 pts) |
| Planted cause recovered | TemGX 97% vs LLM reading the history 43% |
| *Understand "not"*, held-out chat requests: ruled-out title out of top 3 (T1) | 65% vs 46% current |
| *Rely less on popularity*, held-out chat: right title above ruled-out (T2) | 73% vs 63% current |
| Held-out chat NDCG@10 vs current | *understand "not"* −0.016; *less popularity* +0.038 |
| Fix lookup (retrieval), 280 held-out failures | right fix 37% vs 24% one fix for everything, +13 pts [+9, +17]; best possible 42% |
| *Ignore strangers' latest ratings*, held-out NDCG@10 vs current | regular +0.015 [+0.012, +0.018], new members +0.014 [+0.010, +0.019], chat +0.003 (not significant) |

### Honest negative results (kept on purpose)

* **Graph-walk retrieval (GraphRAG over diagnoses) does not pick better fixes:** 35% vs 37% for vector retrieval, −2 pts [−5, +1]. Whether a fix works depends on how the ranker scores titles, which the graph does not encode.
* **Retrieving by the graph explanation alone picks fixes no better (27%).**
* **CATGX driven by attention fails the deletion test on TGAT-2L;** the replay-driven version is used instead.
* **For the demo member, *rely less on popularity* does not fix the search; *understand "not"* does.**
* ***Ignore strangers' latest ratings* does not help chat requests.** Adding it to *less popularity* changes chat NDCG by −0.002 (not significant).

Full numbers: `docs/METHODS_LEDGER.md` (Parts I–XII) and `docs/EVALUATION_REPORT.md`.

---

## 6. What we do not claim

* No Netflix data, features, model or traffic. A 3-signal ranker is not a 700-feature production ranker; section 7 shows a 45-feature LambdaMART with SHAP as a step toward that.
* No A/B test and no online effect.
* No coding agent was run. The checks are something an agent's changes *could* be scored on.
* The 282 labelled pairs have not been used to train anything.
* Requests and labels are written for the prototype; DEV and FINAL use different templates.
* Explanations describe what the model relies on, not why a person chose a title.

---

## 7. The other tabs

* **① Eval report:**
  * one evaluation run, every failed search grouped by cause, with a targeted fix per cause, each scored once on held-out searches;
  * views for new members, chat requests and semantic-ID patterns;
  * baselines on the full catalogue (popularity, BPR-MF, LightGCN, SASRec, TGAT, TGN);
  * retrieval followed by a 45-feature LambdaMART with SHAP triage;
  * the planted-cause test.
* **② Investigate:**
  * pick a graph model (TGAT 1-layer, TGAT 2-layer, TGN) and how far to look (1–3 hops);
  * run TemGX and CATGX on exactly the subgraph on screen, remove ratings and re-rank;
  * includes the original agent investigation (member 470, "a light, cozy mystery, not too intense").

---

## 8. Rebuild from scratch

Use this only if you want to regenerate everything; the bundled `artifacts/` are enough for the demo.

```bash
# Python 3.11–3.13
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
# data: unzip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip into data/raw/ml-latest-small/
make all     # prepare -> train -> cases -> rankers -> select -> investigate -> bench -> report -> test
make web     # build the React UI (Node 18+)
make serve
```

Case-study pipeline, in order (times on 2 CPUs):

| step | command | writes |
|---|---|---|
| extra request types | `python -m srlab.extra` | `artifacts/cases/extra_cases.json` |
| fixes (DEV fit) | `python -m srlab.fixes --fit` then `--fit4` | `artifacts/checkpoints/ranker_F2/F3/F4.json` |
| fixes (held-out, once) | `python -m srlab.fixes --final` | `fixes_final.json` |
| graph models | `python -m srlab.backbones tgat2` / `tgn` | `backbone_*.pt`, `calib_*.json` |
| case study | `python -m srlab.casestudy` | `casestudy.json` |
| CATGX structures + LLM evidence | `python -m srlab.structure` | `structure_evidence.json`, `structure_prompts.json` (then answers → `structure_llm.json`) |
| both-sides explanations | `python -m srlab.contrast` | `contrast_summary.json`, `artifacts/cache/contrast_*.json` |
| generated checks | `python -m srlab.testgen` | `testgen.json`, `tests_final.jsonl`, `training_pairs_dev.jsonl` |
| verified-diagnosis store | `python -m srlab.xstore build dev`, `… build final`, `python -m srlab.xstore eval` | `xstore/*.json`, `xstore_eval.json` |
| ignore strangers' latest ratings | `python -m srlab.f5 dev`, then `python -m srlab.f5 final` | `f5_choice.json`, `f5_dev.json`, `f5_final.json` |
| GraphRAG retrievers | `python -m srlab.retrievers` | `retrievers.json` |
| graph-walk retrieval | `python -m srlab.graphrag eval` | `graphrag_eval.json` |
| demo case (fixed rule) | `python -m srlab.bestcase` | `bestcase.json` |
| RAG sheets + answers | `python -m srlab.rag build`, write answers to `rag_llm_part*.json`, `python -m srlab.rag merge` | `rag_context.json`, `rag_prompts.json`, `rag_llm.json` |

All outputs go to `artifacts/reports/` unless shown otherwise.

Reproducibility:
* ratings.csv SHA-256 `80da8b33…b70e8`;
* events fingerprint `378d0db74318ab1e`;
* backbone `1d01889bb959b06c`, seed 20261022.

`docs/MANIFEST.md5` lists checksums of the code and results.

---

## 9. Repository layout

```
srlab/                 Python package (FastAPI server + all analyses)
  data.py history.py train.py backbones.py tgraph.py      data, graph models
  queries.py extra.py text.py ranker.py fixes.py live.py  requests, ranker, fixes, live re-ranking
  temgx.py catgx.py joint.py explain.py motifs.py         explainers
  contrast.py structure.py casestudy.py bestcase.py       case study
  testgen.py xstore.py rag.py graphrag.py retrievers.py f5.py
  triage.py semid.py baselines.py ltr.py plant*.py compare.py scale.py agent*.py report.py
  server.py                                               HTTP API, serves web/dist
web/src/               React + TypeScript (Vite)
  CaseStudy.tsx        the 4-step demo
  names.ts             plain-English method names
  StructurePanel.tsx SubGraph.tsx Report.tsx Details.tsx Production.tsx …
web/dist/              built page (served by the server)
artifacts/             checkpoints/, cases/, cache/, reports/ (all measured results)
data/                  MovieLens raw + processed
docs/                  DEMO_SCRIPT, METHODS_LEDGER, EVALUATION_REPORT, ARCHITECTURE, PROVENANCE, screenshots/
tests/                 pytest suite
```

---

## 10. HTTP API

Case-study endpoints (the full schema for the older endpoints is in `docs/ARCHITECTURE.md`):

| method | path | returns |
|---|---|---|
| GET | `/api/casestudy` | the scenario, numbers across searches, demo member (from `bestcase.json`) |
| GET | `/api/subgraph/{case}` | the member's neighbourhood for the graph view |
| POST | `/api/live` | `{case_id, backbone, model_id, removed[]}` → live ranking, P(A beats B) |
| POST | `/api/temgx` | TemGX explanation, `k` = how many ratings |
| POST | `/api/catgx_do` | CATGX with the replay signal, `kc` = edge budget |
| POST | `/api/joint` | TemGX + CATGX together, with deletion tests |
| GET | `/api/contrast/{case}` | both-sides explanation and deletion test |
| GET | `/api/structure/{case}` | CATGX structures with evidence and LLM text |
| GET | `/api/retrievers/{case}` | GraphRAG-style picks vs TemGX, and the summary |
| GET | `/api/similar/{case}` | nearest verified diagnoses and the recommended fix |
| GET | `/api/rag/{case}` | evidence sheet, LLM answer, number/citation check |
| GET | `/api/testgen` | generated checks and pass rates per fix |
| GET | `/api/f5` | held-out results of *ignore strangers' latest ratings* |

Other endpoints: `/api/meta`, `/api/case/{id}`, `/api/score`, `/api/explain`, `/api/motifs/{id}`, `/api/investigation/*`, `/api/eval`, `/api/evalreport`, `/api/compare`, `/api/scalability`, `/api/backbones`, `/api/neighborhood/{id}`, `/api/latency`, `/api/diagnostics/negation`, `/api/export/{id}`.

---

## 11. Tests

```bash
.venv/bin/python -m pytest -q      # 42 tests
```

The tests cover:
* API numbers equal direct model inference;
* explainers and deletion tests;
* both-sides explanations and generated checks;
* the verified-diagnosis store;
* RAG number and citation checks;
* graph-walk retrieval;
* *ignore strangers' latest ratings*.

---

## 12. Further documents

| file | what is in it |
|---|---|
| `docs/DEMO_SCRIPT.md` | the 5-minute script with timings (top), older scripts below |
| `docs/METHODS_LEDGER.md` | every method, what was adapted from the papers, every number, Parts I–XII |
| `docs/EVALUATION_REPORT.md` | the evaluation run in full |
| `docs/ARCHITECTURE.md` | data flow, schemas, older API |
| `docs/PROVENANCE.md` | where every number comes from |
| `docs/screenshots/best_*.png` | the four steps for member 290 |

---

## 13. Licences and sources

* **MovieLens:** research use. Cite Harper & Konstan (2015). Non-commercial without GroupLens permission. Redistribution only under the same terms.
* **spaCy model:** MIT.
* **This code:** interview and research demonstration.

Public Netflix context cited in the demo:
* [BGR: Netflix testing OpenAI-powered search (April 2025)](https://www.bgr.com/tech/openais-chatgpt-will-fix-the-worst-thing-about-netflix/)
* [InfoQ: Foundation Models for Ranking, Netflix, QCon London 2025](https://www.infoq.com/presentations/foundation-models-ranking)
