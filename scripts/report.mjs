#!/usr/bin/env node
/**
 * Zero-dependency Node.js experiment telemetry aggregator & report generator.
 *
 * Reads JSON result files produced by `qqn_llm.train` in a run directory,
 * prints a Markdown comparison table to stdout, and writes `summary.md` and
 * `summary.csv` into that directory.
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
  };
});

rows.sort((a, b) => (a.bestValLoss ?? Infinity) - (b.bestValLoss ?? Infinity));

const fmt = (v, d = 4) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const fmtCompact = (v) => {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(2)}k`;
  return v.toFixed(3);
};

const mdLines = [
  `# Experiment Summary: \`${path.basename(targetDir)}\``,
  "",
  `Parameters: **${(rows[0]?.paramsCount || 0).toLocaleString()}** | Optimizers compared: **${rows.length}** (sorted by Best Val Loss)`,
  "",
  "| Optimizer | Steps | Evals | Init Train | Final Train | Best Val | Best PPL | Best ΔL1 | Best ΔL2 | Final ΔL1 | Final ΔL2 | Steady (s) | s/eval | Mean `t` | Accept % |",
  "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
  ...rows.map(
    (r) =>
      `| \`${r.name}\` | ${r.steps} | ${r.evals} | ${fmt(r.initTrainLoss)} | ${fmt(r.finalTrainLoss)} | **${fmt(r.bestValLoss)}** | ${fmt(r.bestValPpl, 2)} | ${fmtCompact(r.bestValL1)} | ${fmtCompact(r.bestValL2)} | ${fmtCompact(r.finalL1)} | ${fmtCompact(r.finalL2)} | ${fmt(r.steadySec, 1)} | ${fmt(r.secPerEval, 2)} | ${fmt(r.avgStepSize, 3)} | ${(r.acceptRate * 100).toFixed(0)}% |`
  ),
  "",
];

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
  ].join(",")
);
fs.writeFileSync(path.join(targetDir, "summary.csv"), [csvHeader, ...csvRows].join("\n") + "\n", "utf-8");

console.log(md);
