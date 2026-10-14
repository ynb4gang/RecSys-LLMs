/**
 * week5/tests/load-hw5.js — load the classic (non-module) week5 scripts into
 * Node so the six `TODO(hw5)` functions can be tested headlessly.
 *
 * data.js and script.js share one top-level scope (classic <script> semantics),
 * so both files are evaluated in a single `new Function` body together with a
 * footer that exports the interesting bindings onto `globalThis.__hw5`
 * (getters are used for mutable state so tests always see current values).
 *
 * The DOM (`window`, `document`) is stubbed: script.js only touches it inside
 * functions that the tests do not call (renderers, initTf, window.onload).
 *
 * TensorFlow.js is loaded from `@tensorflow/tfjs-node` when installed (native
 * CPU backend, much faster), else from the pure-JS `@tensorflow/tfjs`.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const WEEK5_DIR = path.join(__dirname, "..");

/** Footer evaluated together with data.js + script.js to export bindings. */
const FOOTER = `
globalThis.__hw5 = {
  NUM_FACTORS,
  get movies() { return movies; },
  get ratings() { return ratings; },
  get numUsers() { return numUsers; },
  get numMovies() { return numMovies; },
  get SPLIT_EXCLUSIONS() { return SPLIT_EXCLUSIONS; },
  get trainSet() { return trainSet; },
  get valSet() { return valSet; },
  get testSet() { return testSet; },
  get model() { return model; },
  set model(v) { model = v; },
  get trainingHistory() { return trainingHistory; },
  set trainingHistory(v) { trainingHistory = v; },
  get bestCheckpoint() { return bestCheckpoint; },
  set bestCheckpoint(v) { bestCheckpoint = v; },
  get tfBackend() { return tfBackend; },
  isTfModel,
  clamp,
  fmt,
  movieTitle,
  escapeHtml,
  computeRMSEFor,
  computeBaselineRMSE,
  predictRating,
  initializeFactors,
  trainMF,
  selectBestCheckpoint,
  recommendTopK,
  recallAtK,
  fixtureModel,
  withFixture,
  runTests,
  splitByTimestamp,
  TOY_RATINGS,
  TOY_MOVIES,
  REC_FIXTURE,
  recallFixture,
  // Mirror of data.js loadData() without the fetch() calls.
  loadRealData() {
    parseItemData(require("fs").readFileSync(require("path").join(__dirname, "u.item"), "utf8"));
    parseRatingData(require("fs").readFileSync(require("path").join(__dirname, "u.data"), "utf8"));
    numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
    numMovies = movies.length;
    buildRatingMatrix();
  },
  // Mirror of the window.onload split assignment in script.js.
  applySplit() {
    const s = splitByTimestamp();
    trainSet = s.train;
    valSet = s.val;
    testSet = s.test;
    return { train: s.train.length, val: s.val.length, test: s.test.length };
  },
};
`;

/**
 * Evaluate the week5 scripts and return the exported `__hw5` namespace plus the
 * tfjs module. Async because the tf backend must be initialized first.
 * Prefers the native `@tensorflow/tfjs-node` backend when installed (much
 * faster on CPU), and falls back to the pure-JS `@tensorflow/tfjs`.
 *
 * @param {{window?: Object, document?: Object, fetch?: Function}} [opts]
 *        optional DOM / window / fetch stubs (used by ui-flow.test.js to drive
 *        the real page code: window.onload, button handlers, renderers).
 * @returns {Promise<{hw5: Object, tf: Object, tfModule: string}>}
 */
async function loadHW5(opts = {}) {
  let tf;
  let tfModule;
  try {
    tf = require("@tensorflow/tfjs-node");
    tfModule = "@tensorflow/tfjs-node";
    await tf.ready();
  } catch (e) {
    tf = require("@tensorflow/tfjs"); // pure-JS CPU backend in Node
    tfModule = "@tensorflow/tfjs";
    await tf.setBackend("cpu");
    await tf.ready();
  }

  const src = [
    fs.readFileSync(path.join(WEEK5_DIR, "data.js"), "utf8"),
    fs.readFileSync(path.join(WEEK5_DIR, "script.js"), "utf8"),
    FOOTER,
  ].join("\n;\n");

  const docStub =
    opts.document ||
    {
      getElementById: () => null,
      createElement: () => ({ style: {}, setAttribute() {}, appendChild() {} }),
      body: null,
    };
  const winStub = opts.window || {};
  const fetchStub = opts.fetch || globalThis.fetch;
  const fn = new Function("tf", "window", "document", "require", "__dirname", "fetch", src);
  fn(tf, winStub, docStub, require, WEEK5_DIR, fetchStub);

  if (!globalThis.__hw5) throw new Error("failed to export __hw5 from week5 scripts");
  return { hw5: globalThis.__hw5, tf, tfModule };
}

module.exports = { loadHW5, WEEK5_DIR };
