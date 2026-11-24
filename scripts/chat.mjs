#!/usr/bin/env node
/**
 * Zero-dependency interactive Node.js terminal chat client for qqn-llm's
 * OpenAI-compatible Gemma 4 inference server.
 *
 * Usage:
 *   npm run chat -- --url http://<ec2-public-ip>:8000
 *   npm run chat -- --url http://127.0.0.1:8000
 */

import readline from "node:readline";
import process from "node:process";

const args = process.argv.slice(2);
let baseUrl = "http://127.0.0.1:8000";
let maxTokens = 200;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--url" && args[i + 1]) baseUrl = args[++i].replace(/\/+$/, "");
  if (args[i] === "--max-tokens" && args[i + 1]) maxTokens = Number(args[++i]);
}

async function checkHealth() {
  try {
    const res = await fetch(`${baseUrl}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    console.log(`Connected to ${baseUrl} (model=${data.model || "gemma4_e2b"})`);
    if (data.metadata?.optimizer_name) {
      console.log(
        `Checkpoint: ${data.metadata.optimizer_name} | val_loss=${data.metadata.val_loss?.toFixed?.(4) ?? "—"} | ΔL1=${data.metadata.weight_l1_dist?.toFixed?.(0) ?? "—"}`
      );
    }
  } catch (err) {
    console.error(`Warning: could not reach ${baseUrl}/health (${err.message}).`);
  }
}

await checkHealth();
console.log("Type a message and press Enter (type /quit to exit).\n");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: "you> ",
});

rl.prompt();

rl.on("line", async (line) => {
  const prompt = line.trim();
  if (!prompt) {
    rl.prompt();
    return;
  }
  if (prompt === "/quit" || prompt === "/exit") {
    rl.close();
    return;
  }

  try {
    const res = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma4_e2b",
        messages: [{ role: "user", content: prompt }],
        max_tokens: maxTokens,
      }),
    });
    if (!res.ok) {
      console.error(`Server error ${res.status}: ${await res.text()}`);
    } else {
      const data = await res.json();
      const reply = data.choices?.[0]?.message?.content ?? "";
      const sec = data.latency_seconds != null ? ` (${data.latency_seconds}s)` : "";
      console.log(`\nmodel${sec}> ${reply}\n`);
    }
  } catch (err) {
    console.error(`Request failed: ${err.message}`);
  }
  rl.prompt();
});

rl.on("close", () => {
  console.log("\nGoodbye!");
  process.exit(0);
});
