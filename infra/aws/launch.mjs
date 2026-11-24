#!/usr/bin/env node
/**
 * Lightweight Node.js AWS EC2 GPU launcher, S3 code bundler, multi-region
 * capacity fallback, live S3 log tailer, result puller, cost estimator, and
 * teardown CLI.
 *
 * Uses zero external npm dependencies (invokes AWS CLI v2 + tar).
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const STATE_DIR = path.resolve(process.cwd(), ".aws-state");
const STATE_FILE = path.join(STATE_DIR, "instances.json");
const DEFAULT_BUCKET = "qqn-llm-470240306861-us-east-1";
const DEFAULT_IAM_PROFILE = "qqn-llm-ec2-profile";
const DEFAULT_REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-west-1"];

const MODE_CONFIGS = {
  sweep: "configs/gemma4_e2b_aws_sweep.yaml",
  "macro-batch": "configs/gemma4_e2b_aws_macro_batch.yaml",
  "linesearch-suite": "configs/gemma4_e2b_aws_linesearch_suite.yaml",
  "two-phase": "configs/gemma4_e2b_aws_two_phase.yaml",
  "pokemon-pirate": "configs/gemma4_e2b_pokemon_pirate.yaml",
  g6e: "configs/gemma4_e2b_aws_g6e.yaml",
};

const DEFAULT_DLAMI_SSM_PATHS = [
  "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id",
  "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
];

const INSTANCE_CATALOG = {
  "g6e.xlarge": {
    gpus: "1x L40S (48 GB)",
    hourlyUsd: 1.86,
    spotHourlyUsd: 0.42,
    fsdpAxisSize: 1,
    notes: "Ideal single-GPU Gemma 4 E2B inference server (~9.3 GB weights in bf16)",
  },
  "g6e.12xlarge": {
    gpus: "4x L40S (192 GB)",
    hourlyUsd: 10.49,
    spotHourlyUsd: 3.8,
    fsdpAxisSize: 4,
    notes: "Good for AWS bring-up, AdamW, Secant, AdamOracle, L-BFGS m<=5",
  },
  "g6e.48xlarge": {
    gpus: "8x L40S (384 GB)",
    hourlyUsd: 30.13,
    spotHourlyUsd: 11.0,
    fsdpAxisSize: 8,
    notes: "Fits E2B L-BFGS m=10-15 in bf16 (PCIe interconnect)",
  },
  "p4d.24xlarge": {
    gpus: "8x A100-40GB (320 GB, NVLink)",
    hourlyUsd: 22.0,
    spotHourlyUsd: 9.5,
    fsdpAxisSize: 8,
    notes: "NVLink FSDP throughput for E2B L-BFGS m=5-10 in bf16",
  },
  "p5.48xlarge": {
    gpus: "8x H100-80GB (640 GB, NVSwitch)",
    hourlyUsd: 55.04,
    spotHourlyUsd: 24.0,
    fsdpAxisSize: 8,
    notes: "Full E2B sweep (m=10-20 bf16 or m=10 fp32) & E4B",
  },
};

function parseArgs(argv) {
  const [command = "help", ...rest] = argv;
  const flags = {};
  for (let i = 0; i < rest.length; i++) {
    const token = rest[i];
    if (token.startsWith("--")) {
      const key = token.slice(2);
      const next = rest[i + 1];
      if (!next || next.startsWith("--")) {
        flags[key] = true;
      } else {
        flags[key] = next;
        i++;
      }
    }
  }
  return { command, flags };
}

function runAws(args, { dryRun = false, capture = true } = {}) {
  if (dryRun) {
    console.log("[dry-run] aws", args.join(" "));
    return "";
  }
  return execFileSync("aws", args, {
    encoding: "utf-8",
    stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit",
  });
}

function loadState() {
  if (!fs.existsSync(STATE_FILE)) return { runs: [] };
  return JSON.parse(fs.readFileSync(STATE_FILE, "utf-8"));
}

function saveState(state) {
  fs.mkdirSync(STATE_DIR, { recursive: true });
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2) + "\n", "utf-8");
}

function packageAndUploadBundle(bucket, runId, { dryRun = false } = {}) {
  const bundleS3Uri = `s3://${bucket}/qqn-llm/runs/${runId}/bundle.tar.gz`;
  const includePaths = [
    "pyproject.toml",
    "package.json",
    "src",
    "configs",
    "data",
    "scripts",
    "infra/aws/run_experiment.sh",
    "infra/aws/run_serve.sh",
    "qqn-jax/pyproject.toml",
    "qqn-jax/qqn_jax",
  ].filter((p) => fs.existsSync(path.resolve(process.cwd(), p)));

  if (fs.existsSync(path.resolve(process.cwd(), "qqn-jax/README.md"))) {
    includePaths.push("qqn-jax/README.md");
  }

  if (dryRun) {
    console.log(`[dry-run] tar -czf <tmp>/bundle.tar.gz ${includePaths.join(" ")}`);
    console.log(`[dry-run] aws s3 cp <tmp>/bundle.tar.gz ${bundleS3Uri}`);
    return bundleS3Uri;
  }

  const tmpFile = path.join(os.tmpdir(), `qqn-llm-bundle-${runId}.tar.gz`);
  try {
    execFileSync("tar", ["-czf", tmpFile, "--exclude=__pycache__", "--exclude=*.pyc", ...includePaths], {
      cwd: process.cwd(),
      stdio: ["ignore", "pipe", "inherit"],
    });
    const sizeKb = (fs.statSync(tmpFile).size / 1024).toFixed(1);
    console.log(`Uploading local code bundle (${sizeKb} KB) -> ${bundleS3Uri}`);
    runAws(["s3", "cp", tmpFile, bundleS3Uri, "--only-show-errors"], { capture: false });
  } finally {
    if (fs.existsSync(tmpFile)) fs.unlinkSync(tmpFile);
  }
  return bundleS3Uri;
}

function resolveLatestDlami(region, { dryRun = false } = {}) {
  if (dryRun) {
    console.log(
      `[dry-run] aws ssm get-parameter --region ${region} --name ${DEFAULT_DLAMI_SSM_PATHS[0]} --query Parameter.Value --output text`
    );
    return `ami-AUTO_${region}`;
  }
  for (const paramPath of DEFAULT_DLAMI_SSM_PATHS) {
    try {
      const ami = execFileSync(
        "aws",
        [
          "ssm",
          "get-parameter",
          "--region",
          region,
          "--name",
          paramPath,
          "--query",
          "Parameter.Value",
          "--output",
          "text",
        ],
        { encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"] }
      ).trim();
      if (ami && ami.startsWith("ami-")) {
        return ami;
      }
    } catch {
      // Try next fallback SSM path
    }
  }
  return null;
}

function buildUserData({
  bundleS3Uri,
  configPath,
  bucket,
  runId,
  maxWallMinutes,
  autoTerminate,
}) {
  const shutdownCmd = autoTerminate ? "sudo shutdown -h now" : "true";
  return `#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /var/log/qqn-llm-user-data.log) 2>&1

export RUN_ID="${runId}"
export S3_BUCKET="${bucket}"
export CONFIG_PATH="${configPath}"
export MAX_WALL_MINUTES="${maxWallMinutes}"

upload_boot_log_and_exit() {
  local rc=$?
  echo "[user-data] Exiting with status \${rc}; uploading /var/log/qqn-llm-user-data.log to S3..."
  aws s3 cp /var/log/qqn-llm-user-data.log "s3://${bucket}/qqn-llm/runs/${runId}/user-data.log" --only-show-errors || true
  ${shutdownCmd}
}
trap upload_boot_log_and_exit EXIT

mkdir -p /opt/qqn-llm/work
cd /opt/qqn-llm/work

echo "[user-data] Downloading code bundle from ${bundleS3Uri}..."
aws s3 cp "${bundleS3Uri}" bundle.tar.gz --only-show-errors
tar -xzf bundle.tar.gz

bash infra/aws/run_experiment.sh
`;
}

function cmdCost(flags) {
  const instance = flags.instance || "g6e.12xlarge";
  const hours = Number(flags.hours || 2);
  const spec = INSTANCE_CATALOG[instance];
  if (!spec) {
    console.error(`Unknown instance '${instance}'. Known: ${Object.keys(INSTANCE_CATALOG).join(", ")}`);
    process.exit(1);
  }
  console.log(`Instance:      ${instance} (${spec.gpus})`);
  console.log(`Duration:      ${hours} hr(s)`);
  console.log(`On-Demand est: $${(spec.hourlyUsd * hours).toFixed(2)} ($${spec.hourlyUsd.toFixed(2)}/hr)`);
  console.log(`Spot est:      $${(spec.spotHourlyUsd * hours).toFixed(2)} ($${spec.spotHourlyUsd.toFixed(2)}/hr)`);
  console.log(`Notes:         ${spec.notes}`);
}

function cmdLaunch(flags) {
  const instanceType = flags.instance || "g6e.12xlarge";
  const mode = flags.mode;
  if (mode && !MODE_CONFIGS[mode]) {
    console.error(`Unknown --mode '${mode}'. Available modes: ${Object.keys(MODE_CONFIGS).join(", ")}`);
    process.exit(1);
  }
  const configPath = flags.config || (mode ? MODE_CONFIGS[mode] : "configs/gemma4_e2b_aws_g6e.yaml");
  const bucket = flags.bucket || process.env.QQN_S3_BUCKET || DEFAULT_BUCKET;
  const regions = flags.regions
    ? flags.regions.split(",").map((r) => r.trim()).filter(Boolean)
    : flags.region
      ? [flags.region]
      : process.env.AWS_REGION
        ? [process.env.AWS_REGION, ...DEFAULT_REGIONS.filter((r) => r !== process.env.AWS_REGION)]
        : DEFAULT_REGIONS;

  const explicitAmi = flags.ami || process.env.QQN_AWS_AMI_ID;
  const keyName = flags["key-name"] || process.env.QQN_AWS_KEY_NAME;
  const subnetId = flags.subnet || process.env.QQN_AWS_SUBNET_ID;
  const sgId = flags["security-group"] || process.env.QQN_AWS_SG_ID;
  const iamProfile = flags["iam-profile"] || process.env.QQN_AWS_IAM_PROFILE || DEFAULT_IAM_PROFILE;
  const maxWallMinutes = Number(flags["max-wall-minutes"] || 120);
  const volumeGb = Number(flags["volume-gb"] || 250);
  const spot = Boolean(flags.spot);
  const dryRun = Boolean(flags["dry-run"]);
  const autoTerminate = flags["no-auto-terminate"] ? false : true;

  if (explicitAmi && regions.length > 1) {
    console.warn("Note: --ami is region-specific; only trying the first region:", regions[0]);
    regions.length = 1;
  }

  const runId = flags["run-id"] || `qqn-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const bundleS3Uri = packageAndUploadBundle(bucket, runId, { dryRun });
  const userData = buildUserData({
    bundleS3Uri,
    configPath,
    bucket,
    runId,
    maxWallMinutes,
    autoTerminate,
  });
  const userDataB64 = Buffer.from(userData, "utf-8").toString("base64");

  const spec = INSTANCE_CATALOG[instanceType];
  console.log(`Preparing run ${runId} on ${instanceType}${spot ? " (SPOT)" : " (ON-DEMAND)"}`);
  if (spec) {
    console.log(`  Hardware:    ${spec.gpus} | Est rate: ~$${(spot ? spec.spotHourlyUsd : spec.hourlyUsd).toFixed(2)}/hr`);
  }
  console.log(`  Config:      ${configPath}`);
  console.log(`  IAM Profile: ${iamProfile}`);
  console.log(`  S3 Sync:     s3://${bucket}/qqn-llm/runs/${runId}/`);
  console.log(`  Regions:     ${regions.join(" -> ")}`);

  let lastErr = null;
  for (const region of regions) {
    const amiId = explicitAmi || resolveLatestDlami(region, { dryRun });
    if (!amiId) {
      console.warn(`  [${region}] Could not resolve DLAMI via SSM; skipping.`);
      continue;
    }
    if (!dryRun) {
      console.log(`  [${region}] Trying launch with AMI ${amiId}...`);
    }

    const awsArgs = [
      "ec2",
      "run-instances",
      "--region",
      region,
      "--image-id",
      amiId,
      "--instance-type",
      instanceType,
      "--count",
      "1",
      "--instance-initiated-shutdown-behavior",
      "terminate",
      "--block-device-mappings",
      JSON.stringify([
        {
          DeviceName: "/dev/sda1",
          Ebs: { VolumeSize: volumeGb, VolumeType: "gp3", DeleteOnTermination: true },
        },
      ]),
      "--tag-specifications",
      JSON.stringify([
        {
          ResourceType: "instance",
          Tags: [
            { Key: "Name", Value: `qqn-llm-${runId}` },
            { Key: "Project", Value: "qqn-llm" },
            { Key: "RunId", Value: runId },
          ],
        },
      ]),
      "--user-data",
      userDataB64,
    ];

    if (keyName) awsArgs.push("--key-name", keyName);
    if (subnetId) awsArgs.push("--subnet-id", subnetId);
    if (sgId) awsArgs.push("--security-group-ids", sgId);
    if (iamProfile) awsArgs.push("--iam-instance-profile", `Name=${iamProfile}`);
    if (spot) {
      awsArgs.push(
        "--instance-market-options",
        JSON.stringify({ MarketType: "spot", SpotOptions: { SpotInstanceType: "one-time" } })
      );
    }

    try {
      const out = runAws(awsArgs, { dryRun });
      if (dryRun) {
        console.log("\n--- Generated User-Data Preview ---\n" + userData);
        return;
      }
      const parsed = JSON.parse(out);
      const instanceId = parsed.Instances?.[0]?.InstanceId;
      console.log(`\nLaunched instance: ${instanceId} in ${region}`);
      console.log(`Monitor status:    npm run aws:status -- --region ${region}`);
      console.log(`Tail live logs:    npm run aws:tail -- --run-id ${runId}`);
      console.log(`Tail boot log:     npm run aws:tail -- --run-id ${runId} --user-data`);
      console.log(`Pull & report:     npm run aws:pull -- --run-id ${runId}`);

      const state = loadState();
      state.runs.push({
        runId,
        instanceId,
        instanceType,
        region,
        bucket,
        configPath,
        spot,
        launchedAt: new Date().toISOString(),
      });
      saveState(state);
      return;
    } catch (err) {
      const stderr = String(err?.stderr || err?.message || err);
      lastErr = stderr;
      const retryable =
        stderr.includes("InsufficientInstanceCapacity") ||
        stderr.includes("Unsupported") ||
        stderr.includes("SpotMaxPriceTooLow") ||
        stderr.includes("InstanceLimitExceeded") ||
        stderr.includes("VcpuLimitExceeded") ||
        stderr.includes("MaxSpotInstanceCountExceeded");
      console.warn(`  [${region}] Launch failed: ${stderr.trim().split("\n")[0]}`);
      if (!retryable) {
        console.error(stderr);
        process.exit(1);
      }
    }
  }

  console.error(`\nFailed to provision ${instanceType} across regions [${regions.join(", ")}].`);
  if (lastErr) console.error(lastErr);
  process.exit(1);
}

function cmdStatus(flags) {
  const state = loadState();
  const regions = flags.region
    ? [flags.region]
    : [...new Set([process.env.AWS_REGION || "us-east-1", ...state.runs.map((r) => r.region)])];
  const dryRun = Boolean(flags["dry-run"]);

  for (const region of regions) {
    console.log(`\n=== Region: ${region} ===`);
    const out = runAws(
      [
        "ec2",
        "describe-instances",
        "--region",
        region,
        "--filters",
        "Name=tag:Project,Values=qqn-llm",
        "--query",
        "Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,IP:PublicIpAddress,Launch:LaunchTime,RunId:Tags[?Key=='RunId']|[0].Value}",
        "--output",
        "table",
      ],
      { dryRun }
    );
    if (!dryRun) console.log(out.trim() || "(no instances)");
  }
}

function cmdTerminate(flags) {
  const state = loadState();
  const lastRun = state.runs.at(-1);
  const instanceId = flags["instance-id"] || lastRun?.instanceId;
  const region = flags.region || lastRun?.region || process.env.AWS_REGION || "us-east-1";
  const dryRun = Boolean(flags["dry-run"]);
  if (!instanceId) {
    console.error("Error: pass --instance-id <id> (no tracked instance in .aws-state/instances.json).");
    process.exit(1);
  }
  console.log(`Terminating ${instanceId} in ${region}...`);
  runAws(["ec2", "terminate-instances", "--region", region, "--instance-ids", instanceId], {
    dryRun,
    capture: false,
  });
}

function cmdLogs(flags) {
  const lastRun = loadState().runs.at(-1);
  const bucket = flags.bucket || lastRun?.bucket || DEFAULT_BUCKET;
  const runId = flags["run-id"] || lastRun?.runId;
  const dryRun = Boolean(flags["dry-run"]);
  if (!runId) {
    console.error("Error: pass --run-id <run-id>.");
    process.exit(1);
  }
  runAws(["s3", "ls", `s3://${bucket}/qqn-llm/runs/${runId}/`, "--recursive"], {
    dryRun,
    capture: false,
  });
}

function cmdTail(flags) {
  const lastRun = loadState().runs.at(-1);
  const bucket = flags.bucket || lastRun?.bucket || DEFAULT_BUCKET;
  const runId = flags["run-id"] || lastRun?.runId;
  const lines = Number(flags.lines || 60);
  const logName = flags["user-data"] ? "user-data.log" : "train.log";
  const dryRun = Boolean(flags["dry-run"]);
  if (!runId) {
    console.error("Error: pass --run-id <run-id>.");
    process.exit(1);
  }
  const s3Uri = `s3://${bucket}/qqn-llm/runs/${runId}/${logName}`;
  if (dryRun) {
    console.log(`[dry-run] aws s3 cp ${s3Uri} -`);
    return;
  }
  try {
    const content = runAws(["s3", "cp", s3Uri, "-"]);
    const tailLines = content.split("\n").slice(-lines).join("\n");
    console.log(`=== ${s3Uri} (last ${lines} lines) ===\n${tailLines}`);
  } catch {
    console.log(`Log not yet uploaded to ${s3Uri} (instance may still be booting / installing deps; syncs every 30s).`);
  }
}

function cmdPull(flags) {
  const lastRun = loadState().runs.at(-1);
  const bucket = flags.bucket || lastRun?.bucket || DEFAULT_BUCKET;
  const runId = flags["run-id"] || lastRun?.runId;
  const dryRun = Boolean(flags["dry-run"]);
  if (!runId) {
    console.error("Error: pass --run-id <run-id>.");
    process.exit(1);
  }
  const localDir = path.resolve(process.cwd(), "runs", runId);
  fs.mkdirSync(localDir, { recursive: true });
  const s3Uri = `s3://${bucket}/qqn-llm/runs/${runId}/`;
  console.log(`Syncing ${s3Uri} -> ${localDir}`);
  runAws(["s3", "sync", s3Uri, localDir], { dryRun, capture: false });
  if (!dryRun) {
    try {
      execFileSync("node", ["scripts/report.mjs", localDir], { stdio: "inherit" });
    } catch {
      console.log("No completed optimizer JSON files found yet in", localDir);
    }
  }
}

function buildServeUserData({
  bundleS3Uri,
  checkpointUri,
  bucket,
  serveId,
  port,
  maxWallMinutes,
  evalPokemonPirate,
  autoTerminate,
}) {
  const shutdownCmd = autoTerminate ? "sudo shutdown -h now" : "true";
  return `#!/usr/bin/env bash
set -euo pipefail
exec > >(tee /var/log/qqn-llm-serve-user-data.log) 2>&1

export RUN_ID="${serveId}"
export S3_BUCKET="${bucket}"
export CHECKPOINT_URI="${checkpointUri}"
export SERVE_PORT="${port}"
export MAX_WALL_MINUTES="${maxWallMinutes}"
export EVAL_POKEMON_PIRATE="${evalPokemonPirate ? "1" : "0"}"

upload_boot_log_and_exit() {
  local rc=$?
  aws s3 cp /var/log/qqn-llm-serve-user-data.log "s3://${bucket}/qqn-llm/serve/${serveId}/user-data.log" --only-show-errors || true
  ${shutdownCmd}
}
trap upload_boot_log_and_exit EXIT

mkdir -p /opt/qqn-llm/work
cd /opt/qqn-llm/work

aws s3 cp "${bundleS3Uri}" bundle.tar.gz --only-show-errors
tar -xzf bundle.tar.gz

bash infra/aws/run_serve.sh
`;
}

function cmdServe(flags) {
  const lastRun = loadState().runs.at(-1);
  const instanceType = flags.instance || "g6e.xlarge";
  const bucket = flags.bucket || lastRun?.bucket || DEFAULT_BUCKET;
  const runId = flags["run-id"] || lastRun?.runId;
  const ckptName = flags.checkpoint || "lbfgs_adam_strong_wolfe_best";
  const checkpointUri = flags["checkpoint-uri"] || (runId ? `s3://${bucket}/qqn-llm/runs/${runId}/checkpoints/${ckptName}` : null);
  if (!checkpointUri) {
    console.error("Error: pass --run-id <train-run-id> or --checkpoint-uri <s3://...>");
    process.exit(1);
  }
  const port = Number(flags.port || 8000);
  const maxWallMinutes = Number(flags["max-wall-minutes"] || 120);
  const volumeGb = Number(flags["volume-gb"] || 200);
  const spot = Boolean(flags.spot);
  const dryRun = Boolean(flags["dry-run"]);
  const evalPokemonPirate = flags["no-eval"] ? false : true;
  const autoTerminate = flags["no-auto-terminate"] ? false : true;
  const iamProfile = flags["iam-profile"] || process.env.QQN_AWS_IAM_PROFILE || DEFAULT_IAM_PROFILE;
  const keyName = flags["key-name"] || process.env.QQN_AWS_KEY_NAME;
  const sgId = flags["security-group"] || process.env.QQN_AWS_SG_ID;
  const regions = flags.regions
    ? flags.regions.split(",").map((r) => r.trim()).filter(Boolean)
    : flags.region
      ? [flags.region]
      : DEFAULT_REGIONS;

  const serveId = `serve-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const bundleS3Uri = packageAndUploadBundle(bucket, serveId, { dryRun });
  const userData = buildServeUserData({
    bundleS3Uri,
    checkpointUri,
    bucket,
    serveId,
    port,
    maxWallMinutes,
    evalPokemonPirate,
    autoTerminate,
  });
  const userDataB64 = Buffer.from(userData, "utf-8").toString("base64");

  console.log(`Launching Gemma 4 inference server ${serveId} on ${instanceType}${spot ? " (SPOT)" : " (ON-DEMAND)"}`);
  console.log(`  Checkpoint:  ${checkpointUri}`);
  console.log(`  Port:        ${port} (auto-shutdown after ${maxWallMinutes}m)`);
  console.log(`  Regions:     ${regions.join(" -> ")}`);

  for (const region of regions) {
    const amiId = flags.ami || resolveLatestDlami(region, { dryRun });
    if (!amiId) continue;
    const awsArgs = [
      "ec2",
      "run-instances",
      "--region",
      region,
      "--image-id",
      amiId,
      "--instance-type",
      instanceType,
      "--count",
      "1",
      "--instance-initiated-shutdown-behavior",
      "terminate",
      "--block-device-mappings",
      JSON.stringify([
        { DeviceName: "/dev/sda1", Ebs: { VolumeSize: volumeGb, VolumeType: "gp3", DeleteOnTermination: true } },
      ]),
      "--tag-specifications",
      JSON.stringify([
        {
          ResourceType: "instance",
          Tags: [
            { Key: "Name", Value: `qqn-llm-${serveId}` },
            { Key: "Project", Value: "qqn-llm" },
            { Key: "RunId", Value: serveId },
          ],
        },
      ]),
      "--user-data",
      userDataB64,
      "--iam-instance-profile",
      `Name=${iamProfile}`,
    ];
    if (keyName) awsArgs.push("--key-name", keyName);
    if (sgId) awsArgs.push("--security-group-ids", sgId);
    if (spot) {
      awsArgs.push(
        "--instance-market-options",
        JSON.stringify({ MarketType: "spot", SpotOptions: { SpotInstanceType: "one-time" } })
      );
    }

    try {
      const out = runAws(awsArgs, { dryRun });
      if (dryRun) {
        console.log("\n--- Generated Serve User-Data Preview ---\n" + userData);
        return;
      }
      const parsed = JSON.parse(out);
      const instanceId = parsed.Instances?.[0]?.InstanceId;
      console.log(`\nInference server launched: ${instanceId} in ${region}`);
      console.log(`1. Check serve log & auto-eval in S3:`);
      console.log(`   aws s3 cp s3://${bucket}/qqn-llm/serve/${serveId}/serve.log -`);
      console.log(`2. Port-forward port ${port} securely via SSM (no open firewall needed!):`);
      console.log(`   aws ssm start-session --region ${region} --target ${instanceId} --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["${port}"],"localPortNumber":["${port}"]}'`);
      console.log(`3. Chat from another terminal:`);
      console.log(`   npm run chat -- --url http://127.0.0.1:${port}`);
      const state = loadState();
      state.runs.push({ runId: serveId, instanceId, instanceType, region, bucket, checkpointUri, spot, launchedAt: new Date().toISOString() });
      saveState(state);
      return;
    } catch (err) {
      const stderr = String(err?.stderr || err?.message || err);
      const retryable =
        stderr.includes("InsufficientInstanceCapacity") ||
        stderr.includes("Unsupported") ||
        stderr.includes("SpotMaxPriceTooLow") ||
        stderr.includes("InstanceLimitExceeded") ||
        stderr.includes("VcpuLimitExceeded") ||
        stderr.includes("MaxSpotInstanceCountExceeded");
      console.warn(`  [${region}] Serve launch failed: ${stderr.trim().split("\n")[0]}`);
      if (!retryable) {
        console.error(stderr);
        process.exit(1);
      }
    }
  }
  console.error(`Failed to provision ${instanceType} across regions [${regions.join(", ")}].`);
  process.exit(1);
}

function printHelp() {
  console.log(`qqn-llm AWS Orchestrator (Node.js)

Commands:
  cost       Estimate On-Demand & Spot cost (--instance, --hours)
  launch     Bundle local code & data to S3 and launch EC2 GPU training instance
             (--instance, [--mode sweep|macro-batch|linesearch-suite|two-phase|pokemon-pirate|g6e],
              [--config], [--bucket], [--regions ...], [--spot], [--max-wall-minutes], [--dry-run])
  serve      Launch single-GPU Gemma 4 inference server from an S3 checkpoint
             (--run-id <train-run-id>, [--checkpoint lbfgs_adam_strong_wolfe_best],
              [--instance g6e.xlarge], [--port 8000], [--spot], [--dry-run])
  status     List all EC2 instances tagged Project=qqn-llm across tracked regions
  tail       Print latest live train.log synced to S3 (--run-id, [--lines 60], [--user-data])
  logs       List all S3 artifacts for a run (--run-id, [--bucket])
  pull       Download S3 results to runs/<run-id>/ and generate summary.md/summary.csv (--run-id)
  terminate  Terminate a running instance (--instance-id, [--region])

Known GPU instances: ${Object.keys(INSTANCE_CATALOG).join(", ")}
Default bucket:      ${DEFAULT_BUCKET}
Default IAM profile: ${DEFAULT_IAM_PROFILE}
Default regions:     ${DEFAULT_REGIONS.join(", ")}
`);
}

const { command, flags } = parseArgs(process.argv.slice(2));
switch (command) {
  case "cost":
    cmdCost(flags);
    break;
  case "launch":
    cmdLaunch(flags);
    break;
  case "serve":
    cmdServe(flags);
    break;
  case "status":
    cmdStatus(flags);
    break;
  case "tail":
    cmdTail(flags);
    break;
  case "logs":
    cmdLogs(flags);
    break;
  case "pull":
    cmdPull(flags);
    break;
  case "terminate":
    cmdTerminate(flags);
    break;
  default:
    printHelp();
}
