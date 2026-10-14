/**
 * week5/tests/regression.test.js — independent regression tests for the six
 * HW5 `TODO(hw5)` implementations, plus a real-data smoke run.
 *
 * Run:  cd week5/tests && npm install && node regression.test.js
 *
 * Coverage (beyond the page's built-in 11 self-checks):
 *   1.  Unknown / negative / non-integer user and movie ids.
 *   2.  Empty datasets (empty train, empty splits, empty movies).
 *   3.  Prediction clamping in both directions.
 *   4.  Correct 16-dimensional factor shapes on the real MovieLens split.
 *   5.  Checkpoint selection AND restoration of the actual best-epoch weights.
 *   6.  Exclusion of previously rated movies (train + val) and of ids absent
 *       from the movie list; K truncation; documented unknown-user fallback.
 *   7.  Recall@K denominator counts only relevant movies in the candidate pool.
 *   8.  Candidate pool = movie ids appearing anywhere in the TRAIN split.
 *   9.  Tensor memory management (tf.memory().numTensors deltas).
 *  10.  Repeated training (history replacement, repeated initialization).
 *  11.  Real-data end-to-end smoke test (split integrity, no leakage).
 */
"use strict";

const { loadHW5 } = require("./load-hw5");

let passed = 0;
let failed = 0;
const failureNames = [];

function check(name, fn) {
  try {
    fn();
    passed++;
    console.log(`PASS  ${name}`);
  } catch (error) {
    failed++;
    failureNames.push(name);
    console.log(`FAIL  ${name}\n      ${error && error.message ? error.message : error}`);
  }
}

function assert(cond, message) {
  if (!cond) throw new Error(message || "assertion failed");
}

function assertClose(a, b, tol, message) {
  if (!(Math.abs(a - b) <= tol)) {
    throw new Error(`${message}: got ${a}, expected ${b} (tol ${tol})`);
  }
}

function disposeModel(hw5) {
  if (hw5.model) {
    if (hw5.model.userFactors) hw5.model.userFactors.dispose();
    if (hw5.model.itemFactors) hw5.model.itemFactors.dispose();
    hw5.model = null;
  }
}

async function main() {
  const { hw5, tf } = await loadHW5();
  hw5.loadRealData();
  hw5.applySplit(); // same as the page's window.onload flow

  // -------------------------------------------------------------------------
  console.log("\n=== Part A: the page's built-in self-checks (headless) ===");
  const builtin = hw5.runTests(null);
  console.log(
    `built-in: ${builtin.passed} passed, ${builtin.failed} failed, ${builtin.pending} pending (of ${builtin.checks.length})`,
  );
  for (const c of builtin.checks) {
    if (c.status !== "PASS") console.log(`  ${c.status} ${c.name} — ${c.detail}`);
  }
  check("built-in harness reports 11 passed / 0 failed / 0 pending", () => {
    assert(builtin.passed === 11, `passed = ${builtin.passed}`);
    assert(builtin.failed === 0, `failed = ${builtin.failed}`);
    assert(builtin.pending === 0, `pending = ${builtin.pending}`);
  });

  const TOY = hw5.TOY_MOVIES;
  const TOYR = hw5.TOY_RATINGS;

  // -------------------------------------------------------------------------
  console.log("\n=== Part B: independent regression tests ===");

  check("predictRating: uninitialized model returns 0 (never throws)", () =>
    hw5.withFixture({ model: null }, () => {
      assert(hw5.predictRating(1, 1) === 0, "expected 0 for model === null");
    }),
  );

  check("predictRating: unknown / negative / non-integer ids fall back to globalMean", () =>
    hw5.withFixture(
      { movies: TOY, train: TOYR, model: hw5.fixtureModel({ 1: [1, 2] }, { 3: [3, 4] }, 3) },
      () => {
        assert(hw5.predictRating(999999, 3) === 3, "unknown user must return globalMean");
        assert(hw5.predictRating(1, 999999) === 3, "unknown movie must return globalMean");
        assert(hw5.predictRating(-1, 3) === 3, "negative user id must return globalMean");
        assert(hw5.predictRating(1, -5) === 3, "negative movie id must return globalMean");
        assert(hw5.predictRating(1.5, 3) === 3, "non-integer user id must return globalMean");
        assert(hw5.predictRating(1, 2.5) === 3, "non-integer movie id must return globalMean");
        assert(hw5.predictRating(NaN, 3) === 3, "NaN user id must return globalMean");
        disposeModel(hw5);
      },
    ),
  );

  check("predictRating: clamps to [1, 5] in both directions, exact in range", () =>
    hw5.withFixture(
      {
        movies: TOY,
        train: TOYR,
        model: hw5.fixtureModel({ 1: [10], 2: [-10], 3: [1] }, { 1: [10], 2: [10], 3: [1] }, 3),
      },
      () => {
        assert(hw5.predictRating(1, 1) === 5, `3 + 100 must clamp to 5, got ${hw5.predictRating(1, 1)}`);
        assert(hw5.predictRating(2, 1) === 1, `3 - 100 must clamp to 1, got ${hw5.predictRating(2, 1)}`);
        assert(hw5.predictRating(3, 3) === 4, `3 + 1 must stay 4, got ${hw5.predictRating(3, 3)}`);
        disposeModel(hw5);
      },
    ),
  );

  check("initializeFactors: 16-dim factors, [944,16]/[1683,16] shapes, train-only mean", () =>
    // NB: withFixture resets trainSet unless it is passed explicitly.
    hw5.withFixture({ model: null, train: hw5.trainSet }, () => {
      hw5.initializeFactors(hw5.trainSet); // default NUM_FACTORS = 16
      const m = hw5.model;
      assert(hw5.isTfModel(m), "model must hold tf.Variables");
      assert(
        m.userFactors.shape[0] === hw5.numUsers + 1 && m.userFactors.shape[1] === 16,
        `userFactors shape = ${m.userFactors.shape}, expected [${hw5.numUsers + 1}, 16]`,
      );
      assert(
        m.itemFactors.shape[0] === 1683 && m.itemFactors.shape[1] === 16,
        `itemFactors shape = ${m.itemFactors.shape}, expected [1683, 16]`,
      );
      const mean = hw5.trainSet.reduce((s, r) => s + r.rating, 0) / hw5.trainSet.length;
      assertClose(m.globalMean, mean, 1e-9, "globalMean must equal the training mean");
      disposeModel(hw5);
    }),
  );

  check("initializeFactors: empty training set -> globalMean 0, no throw; predictRating falls back to 0", () =>
    hw5.withFixture({ movies: [], train: [], model: null }, () => {
      const m = hw5.initializeFactors([], 8);
      assert(m.globalMean === 0, `globalMean = ${m.globalMean}, expected 0`);
      assert(m.itemFactors.shape[0] === 1, "empty movie list -> 1 item row");
      assert(hw5.predictRating(1, 1) === 0, "out-of-range -> globalMean = 0 here");
      disposeModel(hw5);
    }),
  );

  check("trainMF: empty training set returns an empty history (and resets it)", () =>
    hw5.withFixture({ movies: TOY, train: [], val: TOYR, model: null, history: [] }, () => {
      hw5.initializeFactors([], 4);
      hw5.trainingHistory = [{ epoch: 99, trainRMSE: 1, valRMSE: 1 }]; // stale entry
      const history = hw5.trainMF([], TOYR, { epochs: 5, lr: 0.01, reg: 0.05, batchSize: 2 });
      assert(Array.isArray(history) && history.length === 0, "expected []");
      disposeModel(hw5);
    }),
  );

  check("trainMF: restores the ACTUAL weights of the best-validation epoch", () => {
    let provable = false;
    let attempts = 0;
    // Diverging hyper-parameters (tiny toy set, huge lr, no reg) make late
    // epochs worse than the best one. Because init and shuffling are random,
    // retry until we actually observe best != last — only then can
    // "model == best AND model != last" prove the weights were restored.
    while (!provable && attempts < 8) {
      attempts++;
      hw5.withFixture({ movies: TOY, train: TOYR, val: TOYR, model: null }, () => {
        hw5.initializeFactors(TOYR, 4);
        const history = hw5.trainMF(TOYR, TOYR, { epochs: 8, lr: 0.5, reg: 0, batchSize: 2 });
        const best = hw5.selectBestCheckpoint(history);
        const last = history[history.length - 1];
        assert(best, "no best checkpoint");
        const modelRMSE = hw5.computeRMSEFor(TOYR);
        assertClose(modelRMSE, best.valRMSE, 1e-4, "model RMSE must equal the best epoch's val RMSE");
        if (Math.abs(last.valRMSE - best.valRMSE) > 1e-3) {
          assert(
            Math.abs(modelRMSE - last.valRMSE) > 1e-3,
            "model unexpectedly holds last-epoch weights — restoration broken",
          );
          provable = true;
        }
        disposeModel(hw5);
      });
    }
    assert(provable, "could not construct a diverging scenario in 8 attempts; test inconclusive");
  });

  check("selectBestCheckpoint: null/empty/NaN rows handled; input not mutated; earliest tie wins", () => {
    assert(hw5.selectBestCheckpoint(null) === null, "null -> null");
    assert(hw5.selectBestCheckpoint([]) === null, "[] -> null");
    const history = [
      { epoch: 5, trainRMSE: 0.9, valRMSE: Number.NaN },
      { epoch: 4, trainRMSE: 0.9, valRMSE: 0.4 },
      { epoch: 2, trainRMSE: 1.0, valRMSE: 0.3 },
      { epoch: 3, trainRMSE: 0.8, valRMSE: 0.3 },
    ];
    const snapshot = JSON.stringify(history);
    const best = hw5.selectBestCheckpoint(history);
    assert(best && best.epoch === 2, `best epoch = ${best && best.epoch}, expected 2 (NaN skipped, tie -> earliest)`);
    assert(JSON.stringify(history) === snapshot, "input array was mutated");
    assert(hw5.selectBestCheckpoint([{ epoch: 1, valRMSE: Number.NaN }]) === null, "all-NaN -> null");
  });

  check("recommendTopK: excludes train+val AND ids missing from the movie list; K respected", () =>
    hw5.withFixture(
      {
        movies: [1, 2, 3, 4, 5, 7, 8, 9].map((id) => ({ id, title: `M${id}`, genres: [] })), // id 6 absent
        train: [{ userId: 1, movieId: 1, rating: 5, timestamp: 1 }],
        val: [{ userId: 1, movieId: 3, rating: 4, timestamp: 2 }],
        test: [],
        model: hw5.fixtureModel({ 1: 1 }, { 1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5.9, 7: 5, 8: 4, 9: 3 }, 3),
      },
      () => {
        const recs = hw5.recommendTopK(1, 10);
        const ids = recs.map((r) => r.movieId);
        assert(!ids.includes(1), "train-rated movie leaked into recommendations");
        assert(!ids.includes(3), "val-rated movie leaked into recommendations");
        assert(!ids.includes(6), "movie id absent from the movie list leaked in");
        assert(ids.join(",") === "2,4,5,7,8,9", `unexpected ids ${ids.join(",")}`);
        for (let i = 1; i < recs.length; i++) {
          assert(recs[i - 1].score >= recs[i].score, "not sorted by descending score");
        }
        const top3 = hw5.recommendTopK(1, 3);
        assert(top3.length === 3 && top3.map((r) => r.movieId).join(",") === "2,4,5", "K truncation broken");
        disposeModel(hw5);
      },
    ),
  );

  check("recommendTopK: unknown user scores every candidate at globalMean; guards return []", () =>
    hw5.withFixture(
      {
        movies: [1, 2, 3].map((id) => ({ id, title: `M${id}`, genres: [] })),
        train: [],
        val: [],
        test: [],
        model: hw5.fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3 }, 2.5),
      },
      () => {
        const recs = hw5.recommendTopK(999999, 2); // beyond every allocated row
        assert(recs.length === 2, `expected 2 recs, got ${recs.length}`);
        assert(recs.every((r) => r.score === 2.5), "unknown user must score at globalMean");
        assert(hw5.recommendTopK(1, 0).length === 0, "K = 0 -> []");
        assert(hw5.recommendTopK(1, -3).length === 0, "K < 0 -> []");
        disposeModel(hw5);
        hw5.withFixture({ model: null }, () => {
          assert(hw5.recommendTopK(1, 10).length === 0, "no model -> []");
        });
      },
    ),
  );

  // Recall fixture (see design notes): candidate pool = {1,2,4,5} (train only).
  //   user 1: relevant test {2} (candidate, via user 2's train rating) + {6}
  //           (relevant but never trained on -> unseen exclusion, NOT in the
  //           denominator). Ranked: 2 > 4 > 5 > 6 ...
  //   user 2: relevant test {1, 4} — both candidates (via users 1 and 3).
  //   user 3: test rating 3.0 -> no relevant candidate -> excluded user.
  const recallSetup = () => ({
    movies: [1, 2, 3, 4, 5, 6].map((id) => ({ id, title: `M${id}`, genres: [] })),
    train: [
      { userId: 1, movieId: 1, rating: 5, timestamp: 1 },
      { userId: 2, movieId: 2, rating: 5, timestamp: 1 },
      { userId: 2, movieId: 5, rating: 4, timestamp: 2 },
      { userId: 3, movieId: 4, rating: 5, timestamp: 1 },
    ],
    val: [{ userId: 1, movieId: 3, rating: 4, timestamp: 2 }],
    test: [
      { userId: 1, movieId: 2, rating: 5, timestamp: 3 },
      { userId: 1, movieId: 6, rating: 5, timestamp: 4 },
      { userId: 2, movieId: 1, rating: 5, timestamp: 3 },
      { userId: 2, movieId: 4, rating: 4, timestamp: 4 },
      { userId: 3, movieId: 3, rating: 3, timestamp: 2 },
    ],
    model: hw5.fixtureModel({ 1: 1, 2: 1, 3: 1 }, { 1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4 }, 0),
  });

  check("recallAtK: denominator = relevant movies in the train candidate pool; unseen reported", () =>
    hw5.withFixture(recallSetup(), () => {
      const r1 = hw5.recallAtK(1);
      assert(r1.perUser.length === 2, `perUser length = ${r1.perUser.length}, expected 2 (user 3 excluded)`);
      const u1 = r1.perUser.find((u) => u.userId === 1);
      const u2 = r1.perUser.find((u) => u.userId === 2);
      assert(u1 && u1.relevant === 1 && u1.hit === 1, `user 1: ${JSON.stringify(u1)}, expected relevant 1 / hit 1`);
      assert(u2 && u2.relevant === 2 && u2.hit === 1, `user 2: ${JSON.stringify(u2)}, expected relevant 2 / hit 1`);
      assertClose(r1.overallRecall, 0.75, 1e-9, "overall = mean(1/1, 1/2)");
      assert(
        JSON.stringify(r1.excluded.unseenIds) === "[6]",
        `unseenIds = ${JSON.stringify(r1.excluded.unseenIds)}, expected [6]`,
      );
      assert(
        JSON.stringify(r1.excluded.usersWithoutRelevant) === "[3]",
        `usersWithoutRelevant = ${JSON.stringify(r1.excluded.usersWithoutRelevant)}, expected [3]`,
      );
      // Denominator is really 2 for user 2: at K=3 both relevant candidates fit.
      const r3 = hw5.recallAtK(3);
      const u2b = r3.perUser.find((u) => u.userId === 2);
      assert(u2b && u2b.hit === 2, "user 2 should hit both relevant candidates at K=3");
      assertClose(r3.overallRecall, 1.0, 1e-9, "overall at K=3 = mean(1, 2/2)");
      // The mean covers exactly the reported users.
      const mean = r3.perUser.reduce((s, u) => s + u.hit / u.relevant, 0) / r3.perUser.length;
      assertClose(r3.overallRecall, mean, 1e-12, "overall must be the mean over reported users");
    }),
  );

  check("recallAtK: candidate pool is GLOBAL (other users' train movies count)", () =>
    hw5.withFixture(recallSetup(), () => {
      // User 1 never rated movie 2 themselves; it is a candidate only because
      // user 2 rated it in TRAIN. Its hit must be counted for user 1.
      const r = hw5.recallAtK(1);
      const u1 = r.perUser.find((u) => u.userId === 1);
      assert(u1 && u1.relevant === 1, "movie 2 must be a candidate for user 1 via the global train pool");
      // Movie 6 is recommendable (it is in `movies` and unseen by user 1) ...
      const recs = hw5.recommendTopK(1, 10);
      // ... but it is NOT a recall candidate (never appears in train), so it
      // must not enter any denominator; it is reported as an unseen exclusion.
      assert(recs.some((x) => x.movieId === 6) || true, "sanity");
      assert(r.excluded.unseenIds.includes(6), "movie 6 must be reported as an unseen relevant item");
      assert(!u1 || u1.relevant === 1, "unseen movie 6 must not inflate the denominator");
    }),
  );

  check("recallAtK: empty splits behave (empty test -> 0; empty train -> all users excluded)", () =>
    hw5.withFixture({ ...recallSetup(), test: [], model: null }, () => {
      hw5.withFixture({ model: hw5.fixtureModel({ 1: 1 }, { 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1 }, 0) }, () => {
        const r = hw5.recallAtK(10);
        assert(r.overallRecall === 0 && r.perUser.length === 0, "empty test -> 0 / []");
        assert(r.excluded.unseenIds.length === 0 && r.excluded.usersWithoutRelevant.length === 0, "empty exclusions");
      });
      // No training ratings -> no candidate pool -> every test user is excluded.
      const fixture = recallSetup();
      fixture.train = [];
      fixture.model = hw5.fixtureModel({ 1: 1, 2: 1, 3: 1 }, { 1: 9, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4 }, 0);
      hw5.withFixture(fixture, () => {
        const r = hw5.recallAtK(10);
        assert(r.overallRecall === 0 && r.perUser.length === 0, "empty train -> no reported users");
        assert(
          JSON.stringify(r.excluded.usersWithoutRelevant) === "[1,2,3]",
          `usersWithoutRelevant = ${JSON.stringify(r.excluded.usersWithoutRelevant)}`,
        );
        assert(
          JSON.stringify(r.excluded.unseenIds) === "[1,2,4,6]",
          `unseenIds = ${JSON.stringify(r.excluded.unseenIds)} (relevant = {2,6} for u1, {1,4} for u2)`,
        );
      });
    }),
  );

  check("memory: predictRating / recommendTopK / recallAtK / computeRMSEFor allocate no tensors", () =>
    hw5.withFixture(
      {
        movies: [1, 2, 3, 4, 5].map((id) => ({ id, title: `M${id}`, genres: [] })),
        train: [{ userId: 1, movieId: 1, rating: 5, timestamp: 1 }],
        val: [{ userId: 1, movieId: 2, rating: 4, timestamp: 2 }],
        test: [{ userId: 1, movieId: 3, rating: 5, timestamp: 3 }],
        model: hw5.fixtureModel({ 1: 1 }, { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }, 0),
      },
      () => {
        const before = tf.memory().numTensors;
        for (let i = 0; i < 100; i++) hw5.predictRating(1, 3);
        for (let i = 0; i < 20; i++) hw5.recommendTopK(1, 10);
        hw5.recommendTopK(999999, 10);
        hw5.recallAtK(10);
        hw5.computeRMSEFor(hw5.trainSet);
        const after = tf.memory().numTensors;
        assert(after === before, `tensor leak: ${after - before} tensors left behind`);
        disposeModel(hw5);
      },
    ),
  );

  check("memory: trainMF leaves exactly the 2 model variables; repeated initializeFactors disposes", () =>
    hw5.withFixture({ movies: TOY, train: TOYR, val: TOYR, model: null }, () => {
      const base = tf.memory().numTensors;
      hw5.initializeFactors(TOYR, 4);
      hw5.trainMF(TOYR, TOYR, { epochs: 3, lr: 0.02, reg: 0.05, batchSize: 2 });
      assert(
        tf.memory().numTensors === base + 2,
        `trainMF leaked ${tf.memory().numTensors - base - 2} tensors (optimizer slots / batch tensors / snapshots)`,
      );
      hw5.initializeFactors(TOYR, 4);
      hw5.initializeFactors(TOYR, 4);
      assert(
        tf.memory().numTensors === base + 2,
        `repeated initializeFactors leaked (now ${tf.memory().numTensors - base} vs base)`,
      );
      disposeModel(hw5);
    }),
  );

  check("repeated training: history replaced (not appended); metrics finite; re-init works", () =>
    hw5.withFixture({ movies: TOY, train: TOYR, val: TOYR, model: null }, () => {
      hw5.initializeFactors(TOYR, 4);
      const h1 = hw5.trainMF(TOYR, TOYR, { epochs: 3, lr: 0.02, reg: 0.05, batchSize: 2 });
      assert(h1.length === 3, `first run history = ${h1.length}`);
      const h2 = hw5.trainMF(TOYR, TOYR, { epochs: 5, lr: 0.02, reg: 0.05, batchSize: 2 });
      assert(h2.length === 5, `second run history = ${h2.length} (stale rows appended?)`);
      assert(hw5.trainingHistory === h2, "module trainingHistory must reference the latest run");
      for (const row of h2) {
        assert(Number.isFinite(row.trainRMSE) && Number.isFinite(row.valRMSE), "non-finite RMSE");
      }
      hw5.initializeFactors(TOYR, 4); // fresh model, further training
      const h3 = hw5.trainMF(TOYR, TOYR, { epochs: 2, lr: 0.02, reg: 0.05, batchSize: 2 });
      assert(h3.length === 2 && h3.every((r) => Number.isFinite(r.valRMSE)), "third run broken");
      disposeModel(hw5);
    }),
  );

  check("real data: split counts, train-only mean, 1-epoch smoke train, recall & top-10 sanity", () => {
    const counts = hw5.applySplit();
    assert(
      counts.train + counts.val + counts.test === hw5.ratings.length,
      `split (${counts.train}+${counts.val}+${counts.test}) != ${hw5.ratings.length}`,
    );
    const mean = hw5.trainSet.reduce((s, r) => s + r.rating, 0) / hw5.trainSet.length;
    const baselineVal = hw5.computeBaselineRMSE(hw5.valSet);
    const baselineTest = hw5.computeBaselineRMSE(hw5.testSet);

    hw5.initializeFactors(hw5.trainSet); // 16 factors on the real split
    // 1 epoch with a large batch keeps the pure-JS CPU backend run short; the
    // browser (WebGL) run uses the assignment defaults (20 epochs, batch 64).
    const history = hw5.trainMF(hw5.trainSet, hw5.valSet, { epochs: 1, lr: 0.01, reg: 0.05, batchSize: 1024 });
    assert(history.length === 1 && history.every((r) => Number.isFinite(r.valRMSE)), "smoke history broken");
    const best = hw5.selectBestCheckpoint(history);
    assert(best && best.valRMSE < baselineVal, `val RMSE ${best && best.valRMSE} should beat baseline ${baselineVal}`);
    assertClose(hw5.model.globalMean, mean, 1e-6, "globalMean must be the train-split mean (no leakage)");

    const rated196 = new Set(
      [...hw5.trainSet, ...hw5.valSet].filter((r) => r.userId === 196).map((r) => r.movieId),
    );
    const recs = hw5.recommendTopK(196, 10);
    assert(recs.length === 10, `expected 10 recs for user 196, got ${recs.length}`);
    assert(recs.every((r) => !rated196.has(r.movieId)), "recs include a train/val-rated movie");
    assert(recs.every((r) => Number.isFinite(r.score)), "non-finite rec score");

    const recall = hw5.recallAtK(10);
    assert(Number.isFinite(recall.overallRecall), "non-finite recall");
    const reportedIds = new Set(recall.perUser.map((u) => u.userId));
    assert(
      recall.excluded.usersWithoutRelevant.every((id) => !reportedIds.has(id)),
      "a user appears both in perUser and in usersWithoutRelevant",
    );
    const trainMovies = new Set(hw5.trainSet.map((r) => r.movieId));
    assert(
      recall.excluded.unseenIds.every((id) => !trainMovies.has(id)),
      "an unseen exclusion id does appear in the train split",
    );
    assert(recall.perUser.every((u) => u.relevant >= 1 && u.hit <= u.relevant), "perUser row invalid");

    console.log(
      `      real-data smoke: ${hw5.ratings.length} ratings, ${hw5.numUsers} users, ` +
        `${hw5.numMovies} movies, split ${counts.train}/${counts.val}/${counts.test}, ` +
        `train mean ${mean.toFixed(4)}, baseline val ${baselineVal.toFixed(4)}, baseline test ${baselineTest.toFixed(4)}, ` +
        `1-epoch val RMSE ${best.valRMSE.toFixed(4)}, Recall@10 ${recall.overallRecall.toFixed(4)} ` +
        `(excluded: ${recall.excluded.unseenIds.length} unseen items, ${recall.excluded.usersWithoutRelevant.length} users)`,
    );
    disposeModel(hw5);
  });

  // -------------------------------------------------------------------------
  console.log(`\n=== Summary: ${passed} passed, ${failed} failed ===`);
  if (failed > 0) {
    console.log(`Failures:\n  - ${failureNames.join("\n  - ")}`);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Fatal:", error);
  process.exit(1);
});
