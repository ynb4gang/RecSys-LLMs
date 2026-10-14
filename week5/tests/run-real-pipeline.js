/**
 * week5/tests/run-real-pipeline.js — run the complete HW5 pipeline on the real
 * MovieLens 100k data with the assignment's DEFAULT hyper-parameters, in Node,
 * as an independent cross-check of the browser run.
 *
 * Run:  cd week5/tests && node run-real-pipeline.js
 *
 * Prints: dataset shape / density / sparsity, split counts, training mean, an
 * independently computed (raw-parse) baseline, the full per-epoch history,
 * best checkpoint, test RMSE, Recall@10 with exclusions, and Top-10 lists for
 * two users with their rating history. Also cross-checks `predictRating`
 * against a hand-written dot product over the factor arrays.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { loadHW5, WEEK5_DIR } = require("./load-hw5");

const fmt4 = (x) => (Number.isFinite(x) ? x.toFixed(4) : String(x));

async function main() {
  const { hw5, tf, tfModule } = await loadHW5();
  hw5.loadRealData();
  const counts = hw5.applySplit();
  const { trainSet, valSet, testSet, ratings, movies, numUsers, numMovies } = hw5;

  console.log(`tf module: ${tfModule} | backend: ${tf.getBackend()} | NUM_FACTORS: ${hw5.NUM_FACTORS}`);

  // --- dataset shape, density, sparsity --------------------------------------
  const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
  // The conceptual user-movie matrix is numUsers x maxMovieId (ids are 1-based).
  // The factor tensors carry an extra sentinel row/column 0 so that tensor row
  // index == raw id; that sentinel is NOT a user or a movie and is excluded
  // from the density/sparsity computation below.
  const conceptualCells = numUsers * maxMovieId;
  const density = ratings.length / conceptualCells;
  console.log("\n== Dataset ==");
  console.log(`users=${numUsers} movies=${numMovies} ratings=${ratings.length}`);
  console.log(`factor tensors: [${numUsers + 1}, F] user rows x [${maxMovieId + 1}, F] item rows (sentinel row/col 0)`);
  console.log(`conceptual matrix: ${numUsers} x ${maxMovieId} = ${conceptualCells} cells`);
  console.log(`density  = ${ratings.length}/${conceptualCells} = ${(100 * density).toFixed(3)}%`);
  console.log(`sparsity = 1 - density = ${(100 * (1 - density)).toFixed(3)}%`);
  console.log(`split: train=${counts.train} val=${counts.val} test=${counts.test}`);
  console.log(`split exclusions (users with <5 ratings): ${hw5.SPLIT_EXCLUSIONS.length}`);

  // --- independent baseline (raw parse, does NOT use the week5 helpers) -----
  const raw = fs
    .readFileSync(path.join(WEEK5_DIR, "u.data"), "utf8")
    .trim()
    .split("\n")
    .map((line) => {
      const f = line.split("\t");
      return { userId: +f[0], movieId: +f[1], rating: +f[2], timestamp: +f[3] };
    });
  const key = (r) => `${r.userId}_${r.movieId}_${r.timestamp}`;
  const trainKeys = new Set(trainSet.map(key));
  const rawTrain = raw.filter((r) => trainKeys.has(key(r)));
  if (rawTrain.length !== trainSet.length) throw new Error("raw/train split mismatch");
  const muRaw = rawTrain.reduce((s, r) => s + r.rating, 0) / rawTrain.length;
  const baseline = (split) => Math.sqrt(split.reduce((s, r) => s + (muRaw - r.rating) ** 2, 0) / split.length);
  console.log("\n== Baseline (independent raw computation) ==");
  console.log(`train mean mu = ${muRaw.toFixed(6)} | week5 computeBaselineRMSE(val) = ${hw5.computeBaselineRMSE(valSet)} | raw = ${fmt4(baseline(valSet))}`);
  console.log(`baseline val RMSE = ${fmt4(baseline(valSet))} | baseline test RMSE = ${fmt4(baseline(testSet))}`);
  if (Math.abs(hw5.computeBaselineRMSE(testSet) - baseline(testSet)) > 1e-9) {
    throw new Error("baseline cross-check mismatch");
  }

  // --- training (assignment defaults) ---------------------------------------
  const opts = { epochs: 20, lr: 0.01, reg: 0.05, batchSize: 64 };
  console.log(`\n== Training (epochs=${opts.epochs}, lr=${opts.lr}, reg=${opts.reg}, batchSize=${opts.batchSize}) ==`);
  const t0 = Date.now();
  hw5.initializeFactors(trainSet);
  const history = hw5.trainMF(trainSet, valSet, opts);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  for (const row of history) {
    console.log(`epoch ${String(row.epoch).padStart(2)}  trainRMSE ${fmt4(row.trainRMSE)}  valRMSE ${fmt4(row.valRMSE)}`);
  }
  console.log(`training time: ${elapsed}s | tensors alive: ${tf.memory().numTensors} (expect 2)`);
  if (tf.memory().numTensors !== 2) throw new Error("tensor leak during training");

  const best = hw5.selectBestCheckpoint(history);
  const last = history[history.length - 1];
  console.log(`best checkpoint: epoch ${best.epoch} valRMSE ${fmt4(best.valRMSE)} (last epoch ${last.epoch} valRMSE ${fmt4(last.valRMSE)})`);
  // The restored model must reproduce the best epoch's train AND val RMSE —
  // if the last epoch differs, this proves the weights really were restored.
  const restoredVal = hw5.computeRMSEFor(valSet);
  console.log(`restored-model val RMSE = ${fmt4(restoredVal)} (matches best: ${Math.abs(restoredVal - best.valRMSE) < 1e-4})`);

  // --- test RMSE vs baseline -------------------------------------------------
  const testRMSE = hw5.computeRMSEFor(testSet);
  console.log("\n== Test ==");
  console.log(`MF test RMSE = ${fmt4(testRMSE)} | baseline test RMSE = ${fmt4(baseline(testSet))} | beats baseline: ${testRMSE < baseline(testSet)}`);

  // --- hand cross-check of predictRating ------------------------------------
  const uData = await hw5.model.userFactors.data();
  const iData = await hw5.model.itemFactors.data();
  const F = 16;
  for (const [u, m] of [[196, 50], [405, 100], [1, 1]]) {
    let dot = 0;
    for (let f = 0; f < F; f++) dot += uData[u * F + f] * iData[m * F + f];
    const manual = Math.min(5, Math.max(1, hw5.model.globalMean + dot));
    const viaFn = hw5.predictRating(u, m);
    console.log(`predictRating(${u}, ${m}): fn=${viaFn.toFixed(8)} manual=${manual.toFixed(8)} match=${Math.abs(viaFn - manual) < 1e-5}`);
  }

  // --- Recall@10 -------------------------------------------------------------
  console.log("\n== Recall@10 ==");
  const recall = hw5.recallAtK(10);
  console.log(`overall Recall@10 = ${fmt4(recall.overallRecall)} over ${recall.perUser.length} evaluable users`);
  console.log(`excluded unseen relevant test items: ${recall.excluded.unseenIds.length} -> ${JSON.stringify(recall.excluded.unseenIds)}`);
  console.log(`excluded users without a relevant candidate: ${recall.excluded.usersWithoutRelevant.length}`);
  const avgRelevant = recall.perUser.reduce((s, u) => s + u.relevant, 0) / recall.perUser.length;
  const avgHits = recall.perUser.reduce((s, u) => s + u.hit, 0) / recall.perUser.length;
  console.log(`avg relevant per user ${avgRelevant.toFixed(3)} | avg hits ${avgHits.toFixed(3)}`);

  // --- two users' Top-10 + history -------------------------------------------
  for (const userId of [196, 405]) {
    console.log(`\n== User ${userId} ==`);
    const hist = [...trainSet, ...valSet].filter((r) => r.userId === userId);
    const genreCount = new Map();
    for (const r of hist) {
      const movie = movies.find((m) => m.id === r.movieId);
      if (!movie) continue;
      for (const g of movie.genres) genreCount.set(g, (genreCount.get(g) || 0) + 1);
    }
    const topGenres = [...genreCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
    console.log(`history: ${hist.length} ratings (train+val), avg ${(hist.reduce((s, r) => s + r.rating, 0) / hist.length).toFixed(2)}`);
    console.log(`top genres: ${topGenres.map(([g, c]) => `${g}(${c})`).join(", ")}`);
    console.log("highest-rated history movies:");
    hist
      .slice()
      .sort((a, b) => b.rating - a.rating || a.movieId - b.movieId)
      .slice(0, 8)
      .forEach((r) => console.log(`  ${r.rating}  ${movies.find((m) => m.id === r.movieId).title}`));
    console.log("Top-10 recommendations:");
    hw5.recommendTopK(userId, 10).forEach((rec, i) => {
      const g = (movies.find((m) => m.id === rec.movieId) || { genres: [] }).genres.slice(0, 3).join("/");
      console.log(`  ${String(i + 1).padStart(2)}. ${fmt4(rec.score)}  ${rec.title}  [${g}]`);
    });
    const rated = new Set(hist.map((r) => r.movieId));
    const recIds = new Set(hw5.recommendTopK(userId, 10).map((r) => r.movieId));
    console.log(`leak check (recs ∩ train/val rated): ${[...recIds].filter((id) => rated.has(id)).length}`);
  }

  disposeAll(hw5);
  console.log(`\nfinal tensors alive: ${tf.memory().numTensors}`);
}

function disposeAll(hw5) {
  const m = hw5.model;
  if (m) {
    m.userFactors.dispose();
    m.itemFactors.dispose();
    hw5.model = null;
  }
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
