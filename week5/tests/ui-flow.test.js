/**
 * week5/tests/ui-flow.test.js — headless end-to-end validation of the REAL
 * application flow in week5/index.html: the actual window.onload bootstrap,
 * the actual button handlers wired by init(), and the actual DOM renderers —
 * driven through a minimal DOM stub (the desktop browser tool was unavailable;
 * every JavaScript code path a user would click is exercised here).
 *
 * Run:  cd week5/tests && node ui-flow.test.js
 *
 * The page's own defaults are used: epochs=20, lr=0.01, reg=0.05, batch=64.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { loadHW5, WEEK5_DIR } = require("./load-hw5");

function makeElement(id) {
  const listeners = {};
  return {
    id,
    innerHTML: "",
    textContent: "",
    className: "",
    value: "",
    addEventListener(type, fn) {
      (listeners[type] = listeners[type] || []).push(fn);
    },
    click() {
      (listeners.click || []).forEach((fn) => fn());
    },
  };
}

const stripTags = (html) => html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();

let failures = 0;
function assert(cond, message) {
  if (!cond) {
    failures++;
    console.log(`FAIL  ${message}`);
  } else {
    console.log(`PASS  ${message}`);
  }
}

async function main() {
  const ids = [
    "tf-status", "status", "dataset-summary-body", "split-summary-body", "training-curve",
    "epochs", "lr", "reg", "batch-size", "user-select", "train", "recommend-btn", "recall-btn",
    "run-tests", "testLog", "recommendations", "recall-results", "result",
  ];
  const els = {};
  for (const id of ids) els[id] = makeElement(id);
  // Default hyper-parameter inputs exactly as hard-coded in index.html.
  els.epochs.value = "20";
  els.lr.value = "0.01";
  els.reg.value = "0.05";
  els["batch-size"].value = "64";

  const documentStub = {
    getElementById: (id) => els[id] || null,
    createElement: (tag) => makeElement(tag),
    body: { insertBefore() {} },
  };
  const windowStub = {};
  const fetchStub = async (p) => {
    const text = fs.readFileSync(path.join(WEEK5_DIR, p), "utf8");
    return { ok: true, status: 200, text: async () => text };
  };

  const { hw5, tf, tfModule } = await loadHW5({
    window: windowStub,
    document: documentStub,
    fetch: fetchStub,
  });

  console.log(`tf module: ${tfModule} | initial backend: ${tf.getBackend()}`);

  // --- the real page bootstrap: initTf -> loadData -> splitByTimestamp -> init
  await windowStub.onload();
  console.log(`backend after initTf: ${tf.getBackend()} | #tf-status: "${els["tf-status"].textContent}"`);
  assert(hw5.trainSet.length + hw5.valSet.length + hw5.testSet.length === 100000, "window.onload loaded and split 100,000 ratings");
  assert(els["dataset-summary-body"].innerHTML.includes("1,682"), "dataset summary renders 1,682 movies");
  assert(
    els["split-summary-body"].innerHTML.includes("79619") &&
      els["split-summary-body"].innerHTML.includes("9596") &&
      els["split-summary-body"].innerHTML.includes("10785"),
    "split summary renders 79619 / 9596 / 10785",
  );
  const optionCount = (els["user-select"].innerHTML.match(/<option/g) || []).length;
  assert(optionCount === new Set(hw5.trainSet.map((r) => r.userId)).size, `user dropdown lists all ${optionCount} train users`);

  // initTf() selected the pure-JS "cpu" backend in Node (there is no WebGL).
  // The page code is backend-agnostic, so switch to the native tfjs-node
  // backend for a fast training run; a real browser would use WebGL here.
  if (tf.getBackend() !== "tensorflow") {
    try {
      await tf.setBackend("tensorflow");
    } catch (e) {
      /* keep the current backend */
    }
  }
  console.log(`backend for training: ${tf.getBackend()}`);

  // --- Train button (page defaults) ------------------------------------------
  console.log("\n== Train click (epochs=20, lr=0.01, reg=0.05, batch=64) ==");
  const tensorsBefore = tf.memory().numTensors;
  const t0 = Date.now();
  els.train.click();
  const trainSeconds = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`#status: ${els.status.textContent}`);
  console.log(`training time: ${trainSeconds}s`);
  assert(/best val RMSE \d+\.\d+ at epoch \d+/.test(els.status.textContent), "status reports best val RMSE and epoch");
  const best = hw5.bestCheckpoint;
  const last = hw5.trainingHistory[hw5.trainingHistory.length - 1];
  console.log(`history: ${hw5.trainingHistory.length} epochs | best: epoch ${best.epoch} valRMSE ${best.valRMSE.toFixed(4)} (last epoch val ${last.valRMSE.toFixed(4)})`);
  assert(hw5.trainingHistory.length === 20, "training history has 20 rows");
  assert(
    (els["training-curve"].innerHTML.match(/<tr/g) || []).length === 21,
    "training-curve table renders 20 epoch rows (+header)",
  );
  assert(els["training-curve"].innerHTML.includes("best-epoch"), "best epoch row is highlighted");
  // Weight restoration on the real run: the live model must reproduce the BEST
  // epoch's val RMSE, which differs from the last epoch's when overfitting.
  const liveVal = hw5.computeRMSEFor(hw5.valSet);
  console.log(`restored-model val RMSE: ${liveVal.toFixed(4)} (best ${best.valRMSE.toFixed(4)}, last ${last.valRMSE.toFixed(4)})`);
  assert(Math.abs(liveVal - best.valRMSE) < 1e-4, "live model holds the best-epoch weights");
  const testRMSE = hw5.computeRMSEFor(hw5.testSet);
  const baselineTest = hw5.computeBaselineRMSE(hw5.testSet);
  console.log(`test RMSE: ${testRMSE.toFixed(4)} vs training-mean baseline ${baselineTest.toFixed(4)}`);
  assert(testRMSE < baselineTest, "MF test RMSE beats the training-mean baseline");
  assert(
    tf.memory().numTensors === tensorsBefore + 2,
    `Train click leaves exactly the 2 model variables (${tf.memory().numTensors - tensorsBefore} alive)`,
  );

  // --- Recommend Top-10 for two users ---------------------------------------
  const ratedTitles = (userId) =>
    new Set(
      [...hw5.trainSet, ...hw5.valSet]
        .filter((r) => r.userId === userId)
        .map((r) => hw5.movieTitle(r.movieId)),
    );
  const recBlocks = {};
  for (const userId of [196, 405]) {
    els["user-select"].value = String(userId);
    const before = tf.memory().numTensors;
    els["recommend-btn"].click();
    const recHtml = els.recommendations.innerHTML;
    recBlocks[userId] = recHtml;
    const items = (recHtml.match(/<li>/g) || []).length;
    const rated = ratedTitles(userId);
    const leaked = [...rated].filter((t) => recHtml.includes(`rec-title">${t}<`));
    assert(items === 10, `user ${userId}: 10 recommendations rendered`);
    assert(leaked.length === 0, `user ${userId}: no train/val-rated movie in the list`);
    assert(tf.memory().numTensors === before, `user ${userId}: recommend click leaks no tensors`);
    console.log(`\n== Top-10 for user ${userId} (as rendered) ==`);
    console.log(stripTags(recHtml));
  }

  // --- Recall@10 button ------------------------------------------------------
  const beforeRecall = tf.memory().numTensors;
  els["recall-btn"].click();
  const recallHtml = els["recall-results"].innerHTML;
  console.log("\n== Recall@10 panel ==");
  console.log(stripTags(recallHtml).slice(0, 300));
  const exclLine = stripTags(recallHtml).match(/Excluded:[^.]*\./);
  console.log(exclLine ? exclLine[0] : "(exclusion line not found)");
  assert(recallHtml.includes("Recall@10"), "recall panel renders the headline");
  assert(/Excluded: \d+ unseen relevant test item\(s\); \d+ user\(s\) without a relevant candidate\./.test(recallHtml), "recall panel reports exclusions");
  assert(tf.memory().numTensors === beforeRecall, "recall click leaks no tensors");

  // --- Run tests button (in-page harness) ------------------------------------
  els["run-tests"].click();
  const log = els.testLog.textContent;
  console.log("\n== In-page self-checks ==");
  console.log(log.split("\n")[0]);
  assert(log.includes("pass 11, fail 0, pending 0"), "in-page harness: 11 pass / 0 fail / 0 pending");

  // --- repeated Train click --------------------------------------------------
  // NB: the in-page runTests() above leaves its own fixtureModel tensors
  // behind (provided harness scaffolding, bounded per click), so measure the
  // delta around the click: training must replace the model 1:1 and add none.
  const tensorsBeforeRetrain = tf.memory().numTensors;
  els.train.click();
  assert(hw5.trainingHistory.length === 20, "second Train click replaces the history (no append)");
  assert(
    tf.memory().numTensors === tensorsBeforeRetrain,
    `repeated training adds no tensors (before ${tensorsBeforeRetrain}, after ${tf.memory().numTensors})`,
  );

  console.log(`\n=== ui-flow: ${failures === 0 ? "ALL CHECKS PASSED" : `${failures} FAILURES`} ===`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
