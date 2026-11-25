#!/usr/bin/env node
/**
 * Zero-dependency Node.js experiment telemetry aggregator & report generator.
 *
 * Reads JSON result files produced by `qqn_llm.train` in a run directory,
 * prints a Markdown comparison table to stdout, and writes `summary.md` and
 * `summary.csv` (plus quantization leaderboard when quantization telemetry is
 * present) into that directory.
 *
 * Usage:
 *   node scripts/report.mjs runs/smoke_local
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const targetDir = path.resolve(process.cwd(), process.argv[2] || "runs/smoke_local");

if (!fs.existsSync(targetDir)) {
  console.error(`Run directory not found: ${targetDir}`);
  process.exit(1);
}

const files = fs
  .readdirSync(targetDir)
  .filter((f) => f.endsWith(".json") && f !== "summary.json")
  .sort();

if (files.length === 0) {
  console.error(`No optimizer result JSON files found in ${targetDir}`);
  process.exit(1);
}

const rows = files.map((file) => {
  const data = JSON.parse(fs.readFileSync(path.join(targetDir, file), "utf-8"));
  const hist = data.history || [];
  const last = hist.at(-1) || {};
  const first = hist[0] || {};
  const valSteps = hist.filter((h) => typeof h.val_loss === "number");
  const lastVal = valSteps.at(-1);
  const bestVal =
    typeof data.best_val_loss === "number"
      ? data.best_val_loss
      : valSteps.length > 0
        ? Math.min(...valSteps.map((h) => h.val_loss))
        : NaN;
  const bestRec = valSteps.reduce(
    (best, h) => (!best || h.val_loss < best.val_loss ? h : best),
    null
  );
  const bestStep = data.best_val_step ?? bestRec?.step ?? NaN;
  const bestEvals = data.best_val_evals ?? bestRec?.num_evals ?? NaN;
  const steadySteps = hist.slice(1);
  const steadySec =
    typeof data.steady_train_seconds === "number"
      ? data.steady_train_seconds
      : steadySteps.reduce((s, h) => s + (h.step_seconds ?? 0), 0);
  const evals = last.num_evals ?? hist.length;
  const steadyEvals = Math.max(1, evals - (first.num_evals ?? 1));
  const avgStepSize =
    hist.length > 0
      ? hist.reduce((s, h) => s + (h.step_size ?? 0), 0) / hist.length
      : 0;
  const acceptRate =
    hist.length > 0
      ? hist.reduce((s, h) => s + (h.ls_success ?? 1), 0) / hist.length
      : 1;

  return {
    name: data.optimizer_name || path.basename(file, ".json"),
    kind: data.optimizer_kind || "unknown",
    steps: hist.length,
    evals,
    initTrainLoss: data.init_train_loss ?? first.train_loss ?? NaN,
    finalTrainLoss: last.train_loss ?? NaN,
    bestValLoss: bestVal,
    bestStep,
    bestEvals,
    finalValLoss: lastVal ? lastVal.val_loss : NaN,
    bestValPpl: Number.isFinite(bestVal) ? Math.exp(Math.min(bestVal, 20)) : NaN,
    bestValL1: data.best_val_weight_l1_dist ?? NaN,
    bestValL2: data.best_val_weight_l2_dist ?? NaN,
    finalL1: data.final_weight_l1_dist ?? lastVal?.weight_l1_dist ?? NaN,
    finalL2: data.final_weight_l2_dist ?? lastVal?.weight_l2_dist ?? NaN,
    steadySec,
    secPerEval: steadySec / steadyEvals,
    wallSec: data.total_train_seconds ?? NaN,
    avgStepSize,
    acceptRate,
    paramsCount: data.num_params ?? 0,
    // Optional OWL / Quantization telemetry
    quantQType: data.quant_eval_qtype ?? null,
    quantBits: data.quant_eval_bits ?? null,
    baseTrainLoss: data.quant_baseline_train_loss ?? NaN,
    pureTrainLoss: data.final_pure_train_loss ?? lastVal?.pure_train_loss ?? NaN,
    quantTrainLoss: data.final_quant_train_loss ?? lastVal?.quant_train_loss ?? NaN,
    quantValLoss: data.final_quant_val_loss ?? lastVal?.quant_val_loss ?? NaN,
    quantValPpl: data.final_quant_val_ppl ?? lastVal?.quant_val_ppl ?? NaN,
    paramL1Norm: data.final_param_l1_norm ?? lastVal?.param_l1_norm ?? NaN,
    paramL1Delta: data.final_param_l1_delta ?? lastVal?.param_l1_delta ?? NaN,
    boundaryFrac: data.final_quant_boundary_frac ?? lastVal?.quant_boundary_frac ?? NaN,
    quantL1Err: data.final_quant_l1_error ?? lastVal?.quant_l1_error ?? NaN,
    noWorseViol: data.final_no_worse_violation ?? lastVal?.no_worse_violation ?? NaN,
  };
});

const hasQuant = rows.some((r) => r.quantQType !== null);

if (hasQuant) {
  // Sort quantization experiments primarily by quantized validation loss, then L1 norm
  rows.sort((a, b) => {
    const qa = Number.isFinite(a.quantValLoss) ? a.quantValLoss : Infinity;
    const qb = Number.isFinite(b.quantValLoss) ? b.quantValLoss : Infinity;
    if (Math.abs(qa - qb) > 1e-6) return qa - qb;
    return (a.paramL1Norm ?? Infinity) - (b.paramL1Norm ?? Infinity);
  });
} else {
  rows.sort((a, b) => (a.bestValLoss ?? Infinity) - (b.bestValLoss ?? Infinity));
}

const fmt = (v, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const fmtSigned = (v, d = 2) =>
  Number.isFinite(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(d)}` : "—";
const fmtCompact = (v) => {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(2)}k`;
  return v.toFixed(3);
};

const mdLines = [
  `# Experiment Summary: \`${path.basename(targetDir)}\``,
  "",
  `Parameters: **${(rows[0]?.paramsCount || 0).toLocaleString()}** | Optimizers compared: **${rows.length}** (${hasQuant ? "sorted by Quantized Val Loss" : "sorted by Best Val Loss"})`,
  "",
  "| Optimizer | Steps | Evals | Best @ (Step/Eval) | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |",
  "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ...rows.map(
    (r) =>
      `| \`${r.name}\` | ${r.steps} | ${r.evals} | ${Number.isFinite(r.bestStep) ? `${r.bestStep}/${r.bestEvals}` : "—"} | ${fmt(r.initTrainLoss)} | ${fmt(r.finalTrainLoss)} | **${fmt(r.bestValLoss)}** | ${fmt(r.bestValPpl, 2)} | ${fmtCompact(r.bestValL1)} | ${fmtCompact(r.bestValL2)} | ${fmtCompact(r.finalL1)} | ${fmtCompact(r.finalL2)} | ${fmt(r.steadySec, 1)} | ${fmt(r.secPerEval, 2)} | ${fmt(r.avgStepSize, 3)} | ${(r.acceptRate * 100).toFixed(0)}% |`
  ),
  "",
];

if (hasQuant) {
  mdLines.push(
    "## OWL Half-Quantization & No-Worse Constraint Telemetry",
    "",
    "| Optimizer | Target Grid | Ref Train | Pure Train | No-Worse Viol | Quant Train | Quant Val (PPL) | `‖θ‖₁` | `Δ‖θ‖₁` | Exact Boundary % | Quant L1 Err |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ...rows.map((r) => {
      const gridLabel =
        r.quantQType === "uniform"
          ? `uniform-${r.quantBits}b`
          : r.quantQType || "—";
      const bndPct = Number.isFinite(r.boundaryFrac)
        ? `${(r.boundaryFrac * 100).toFixed(2)}%`
        : "—";
      return `| \`${r.name}\` | \`${gridLabel}\` | ${fmt(r.baseTrainLoss)} | ${fmt(r.pureTrainLoss)} | ${fmt(r.noWorseViol, 5)} | ${fmt(r.quantTrainLoss)} | **${fmt(r.quantValLoss)}** (${fmt(r.quantValPpl, 2)}) | ${fmt(r.paramL1Norm, 2)} | **${fmtSigned(r.paramL1Delta, 2)}** | **${bndPct}** | ${fmt(r.quantL1Err, 4)} |`;
    }),
    ""
  );
}

const md = mdLines.join("\n");
fs.writeFileSync(path.join(targetDir, "summary.md"), md, "utf-8");

const csvHeader = [
  "optimizer",
  "kind",
  "steps",
  "evals",
  "init_train_loss",
  "final_train_loss",
  "best_val_loss",
  "best_val_step",
  "best_val_evals",
  "best_val_ppl",
  "final_val_loss",
  "best_val_l1_dist",
  "best_val_l2_dist",
  "final_l1_dist",
  "final_l2_dist",
  "steady_seconds",
  "sec_per_eval",
  "wall_seconds",
  "mean_step_size",
  "accept_rate",
  ...(hasQuant
    ? [
        "quant_qtype",
        "ref_train_loss",
        "pure_train_loss",
        "no_worse_violation",
        "quant_train_loss",
        "quant_val_loss",
        "quant_val_ppl",
        "param_l1_norm",
        "param_l1_delta",
        "quant_boundary_frac",
        "quant_l1_error",
      ]
    : []),
].join(",");
const csvRows = rows.map((r) =>
  [
    r.name,
    r.kind,
    r.steps,
    r.evals,
    r.initTrainLoss,
    r.finalTrainLoss,
    r.bestValLoss,
    r.bestStep,
    r.bestEvals,
    r.bestValPpl,
    r.finalValLoss,
    r.bestValL1,
    r.bestValL2,
    r.finalL1,
    r.finalL2,
    r.steadySec,
    r.secPerEval,
    r.wallSec,
    r.avgStepSize,
    r.acceptRate,
    ...(hasQuant
      ? [
          r.quantQType,
          r.baseTrainLoss,
          r.pureTrainLoss,
          r.noWorseViol,
          r.quantTrainLoss,
          r.quantValLoss,
          r.quantValPpl,
          r.paramL1Norm,
          r.paramL1Delta,
          r.boundaryFrac,
          r.quantL1Err,
        ]
      : []),
  ].join(",")
);
fs.writeFileSync(
  path.join(targetDir, "summary.csv"),
  [csvHeader, ...csvRows].join("\n") + "\n",
  "utf-8"
);

console.log(md);
