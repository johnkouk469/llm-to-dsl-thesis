# Experiment models

The DSL models generated for the evaluation chapter of the thesis (chapter 6), tagged by experiment, DSL and variant. They were produced in June 2025 with the code in this repository (GPT-4o through LangChain, validators from the SmAuto and dFlow projects running locally) and are the models the thesis scores.

The thesis evaluates 22 models per DSL. All 22 SmAuto models are here. For dFlow, 19 of the 22 survive; see "Missing models and other gaps" at the end.

## Layout and naming

```
experiments/
  smauto/
    exp1_complexity/            one description and one model per complexity level
    exp2_incremental/           one conversation: a simple model extended in two further turns
    exp3_robustness/            the same description submitted ten times
    exp4_description_quality/   good, medium and poor descriptions, each with and without the Q&A stage
  dflow/
    (same four experiments)
```

Inside a case folder:

| File | Meaning |
|---|---|
| `description.txt` | The natural-language description given to the system (for the incremental experiment, the message sent in that turn). |
| `model.auto` | The final model of the case, when only the final model was kept. |
| `initial.auto` / `initial.dflow` | The first model the LLM produced. |
| `regenerated_N.*` | The model after the N-th pass of the correction loop (validator error fed back to the LLM). The highest N is the final model of the run. |
| `after_qa.dflow` | The model generated after the question-and-answer stage. |
| `plain.*` / `qna.*` | Experiment 4: the final model without and with the Q&A stage. |
| `run_log.txt` | The pipeline's log for that run: validator messages, regeneration attempts, stop reason. |
| `session_log.txt` | The full console transcript of a Q&A session: analysis, questions, answers, generation, validation. |

The files are the pipeline's output files exactly as they were generated, with two cosmetic changes: Windows line endings were converted to LF, and for `smauto/exp4_description_quality/poor/qna.auto` the chat wrapper line (`SmAuto Assistant: ```smauto`) around the model was removed. `SOURCE_MAP.tsv` records the original file name of every file.

The descriptions use fictional hosts and credentials (`mqtt.smarthome.local`, `casa_nova`, `redis-4a9f...`) and, for dFlow, the demo endpoints of the lab's services with the same placeholder `Authorization` header value that the thesis prints.

## Evaluation criteria

Each final model was scored by the author on five criteria, 1 to 5 (thesis, section 6.2): Syntactic Validity, Semantic Fidelity, Operational Logic Soundness, Unjustified Additions (5 means none), Model Economy and Simplicity. The tables below list the scores in that order.

## Experiment 1: complexity of the description

Three descriptions of the same scenario at increasing complexity. "Special" means the description needs an optional feature of the DSL for which the system prompt has a dedicated section: value generators (simulated sensor readings with noise) in SmAuto, role-based access control in dFlow.

| DSL | Level | Folder | Final model | Scores |
|---|---|---|---|---|
| SmAuto | simple | `smauto/exp1_complexity/simple/` | `model.auto` | 5 / 5 / 3 / 5 / 5 |
| SmAuto | complex | `smauto/exp1_complexity/complex/` | `model.auto` | 5 / 5 / 3 / 5 / 5 |
| SmAuto | special | `smauto/exp1_complexity/special/` | `model.auto` | 5 / 5 / 4 / 5 / 5 |
| dFlow | simple | `dflow/exp1_complexity/simple/` | `regenerated_1.dflow` (loop stopped on a repeated error) | 3 / 4 / 3 / 3 / 5 |
| dFlow | complex | `dflow/exp1_complexity/complex/` | `regenerated_3.dflow` (loop stopped on a repeated error) | 4 / 3 / 2 / 5 / 5 |
| dFlow | special | missing (see the end of this page) | | 3 / 3 / 2 / 4 / 5 |

The dFlow simple model is the one the thesis walks through as its scoring example (section 6.2, listing "Simple model of the first experiment for the dFlow implementation"). The description in its `run_log.txt` is a longer variant of the one quoted in the thesis; the model file is the one the thesis lists.

## Experiment 2: incremental development

One conversation per DSL. The first turn asks for the simple model; the second turn extends it into the complex one; the third adds the special feature. The system keeps the conversation history, so each turn works on the previous model.

| DSL | Step | Folder | Final model | Scores |
|---|---|---|---|---|
| SmAuto | 1 simple | `smauto/exp2_incremental/step1_simple/` | `model.auto` | 5 / 4 / 4 / 5 / 5 |
| SmAuto | 2 complex | `smauto/exp2_incremental/step2_complex/` | `model.auto` | 5 / 5 / 4 / 5 / 5 |
| SmAuto | 3 special | `smauto/exp2_incremental/step3_special/` | `model.auto` | 5 / 5 / 4 / 5 / 5 |
| dFlow | 1 simple | `dflow/exp2_incremental/step1_simple/` | `regenerated_3.dflow` | 3 / 5 / 3 / 5 / 5 |
| dFlow | 2 complex | missing (see the end of this page) | | 3 / 4 / 3 / 5 / 5 |
| dFlow | 3 special | missing (see the end of this page) | | 3 / 4 / 3 / 5 / 5 |

## Experiment 3: robustness

The same refined description submitted ten times in fresh sessions, to see how stable the output is. The thesis reports mean and variance over the ten runs, not per-run scores.

SmAuto (`smauto/exp3_robustness/`, description in `description.txt`): 10 runs, `run01` to `run10`, named after the session timestamp. Mean scores 4.3 / 4.7 / 4 / 5 / 5, variances 0.23 / 0.23 / 0 / 0 / 0. The recurring error the thesis discusses (missing parentheses around terms joined with `AND`/`OR` in a compound condition, 7 of 10 runs) is visible in the `run_log.txt` files as `Expected ')'` and `Expected 'AND' or 'OR' ...`.

| Run | Session | Files | Outcome (from `run_log.txt`) |
|---|---|---|---|
| run01 | 20250609-233556 | initial, regenerated 1 to 2 | no log kept |
| run02 | 20250609-234100 | initial, regenerated 1 to 2, log | stopped: same error twice |
| run03 | 20250609-234318 | initial, regenerated 1 to 5, log | stopped: maximum of 5 regenerations |
| run04 | 20250609-234617 | initial, regenerated 1, log | run crashed in the repeated-error check (`'NoneType' object has no attribute 'index'`); the log ends with a manual note "Valid" |
| run05 | 20250609-235310 | initial, regenerated 1, log | stopped: same error twice |
| run06 | 20250609-235453 | initial, regenerated 1 to 4, log | stopped: same error twice |
| run07 | 20250609-235836 | initial, regenerated 1, log | stopped: same error twice |
| run08 | 20250610-000046 | initial, regenerated 1, log | valid after one regeneration |
| run09 | 20250610-000220 | initial, regenerated 1, log | stopped: same error twice |
| run10 | 20250610-000324 | initial, regenerated 1, log | valid after one regeneration |

dFlow (`dflow/exp3_robustness/`): 10 models, `run00.dflow` to `run09.dflow`, from one session (20250621-151436), numbered as the pipeline numbered them. Mean scores 3.5 / 3.8 / 3 / 4.9 / 4.9, variances 0.28 / 0.18 / 0.22 / 0.1 / 0.1. The description of this session was not kept; the models correspond to the full scenario with access control, that is, the "good" description of experiment 4 (`dflow/exp4_description_quality/good/qna/session_log.txt`, first line).

## Experiment 4: quality of the description, with and without the Q&A stage

Three descriptions of the full scenario at decreasing quality (good, medium, poor). Each was run twice: "plain" (description straight to generation) and "qna" (the analysis and question-and-answer stage first, then generation from the enriched description).

| DSL | Description | Plain approach | Scores | Q&A approach | Scores |
|---|---|---|---|---|---|
| SmAuto | good | `smauto/exp4_description_quality/good/plain.auto` | 5 / 5 / 4 / 5 / 5 | `.../good/qna.auto` | 5 / 5 / 5 / 5 / 5 |
| SmAuto | medium | `.../medium/plain.auto` | 5 / 3 / 4 / 5 / 5 | `.../medium/qna.auto` | 4 / 5 / 4 / 5 / 5 |
| SmAuto | poor | `.../poor/plain.auto` | 4 / 2 / 3 / 5 / 5 | `.../poor/qna.auto` | 4 / 5 / 4 / 5 / 5 |
| dFlow | good | `dflow/exp4_description_quality/good/plain/regenerated_5.dflow` (initial + 5 regenerations) | 4 / 3 / 2 / 4 / 4 | `.../good/qna/regenerated_2.dflow` (after_qa + 2 regenerations, loop stopped on a repeated error; full transcript in `session_log.txt`) | 4 / 2 / 2 / 4 / 4 |
| dFlow | medium | `.../medium/plain/regenerated_2.dflow` | 4 / 2 / 2 / 5 / 5 | `.../medium/qna/after_qa.dflow` (valid on the first attempt; transcript in `session_log.txt`) | 5 / 4 / 2 / 4 / 4 |
| dFlow | poor | `.../poor/plain/initial.dflow` | 5 / 1 / 3 / 4 / 4 | `.../poor/qna/after_qa.dflow` | 5 / 3 / 2 / 5 / 5 |

The SmAuto "good" description is the experiment 3 description. The SmAuto medium and poor descriptions were not kept; the thesis characterises them (medium: no monitoring broker and no threshold values; poor: a very general request). The dFlow good and medium descriptions are the first line of the respective `session_log.txt`; the dFlow poor description was not kept.

## Missing models and other gaps

- Three dFlow models are missing: the special model of experiment 1 and the complex and special steps of experiment 2. The dFlow script names its output files after the time the session started, so when a session was repeated the later generations overwrote the earlier files. The models were scored when they were generated; only the scores in the thesis survive.
- The description of the dFlow robustness session was not kept (see experiment 3 above), nor were the SmAuto medium and poor descriptions and the dFlow poor description of experiment 4.
- Run 1 of the SmAuto robustness experiment has no log, and the pipeline itself did not record the outcome of the last regeneration for runs that stopped on a repeated error; the highest-numbered `regenerated_N.auto` is the final model in every case.
- Where a `run_log.txt` is truncated or contains a manual note, it is reproduced as it was written at the time.
