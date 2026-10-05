/**
 * week4/script.js — Association-rule mining starter (HW4).
 *
 * This is a plain (classic) script, NOT an ES module. `week4/transactions.js`
 * (loaded first, in a regular <script> tag) assigns the dictionary-encoded UCI
 * Online Retail baskets to `window.HW4`. This file reads that global, decodes
 * the integer-index baskets back into `{ stock, description }` items, renders a
 * dataset summary, and wires two slider controls ("minimum support" and
 * "minimum confidence") plus a "Run rules" button. Because nothing is fetched,
 * the page works from a `file://` URL with no server.
 *
 * HW4 FUNCTIONS (`TODO(hw4)` stubs in the starter — all seven are implemented
 * below; the marker strings remain only where the test harness looks for them):
 *   1. `dedupeBasket`         — unique stock codes in a basket, first-appearance order.
 *   2. `countItemset`         — baskets containing every requested stock (`0` if any is absent).
 *   3. `computeSupport`       — support = count(A union B) / N, guarded when N = 0.
 *   4. `computeConfidence`    — confidence = count(A union B) / count(A), guarded when count(A) = 0.
 *   5. `computeLift`          — lift = confidence / (count(B) / N), guarded.
 *   6. `findFrequentItemsets` — mine frequent itemsets (Apriori or equivalent).
 *   7. `generateRules`        — turn frequent itemsets into both-direction rules.
 *
 * PROVIDED for you (scaffolding): dataset loading/decoding from `window.HW4`, the
 * inverted-index builder (`buildIndex` / `asIndex` / `indexCache`), the thin
 * counting wrappers `countItem` / `countPair` (they call your `countItemset` and
 * therefore also throw until it is implemented), threshold validation and slider
 * readout, the DOM wiring, all formatting and rendering helpers, and the test
 * harness. Leave the provided code as-is and implement only the stubs above.
 *
 * Metrics (see week4/readme.md for definitions):
 *   support(A -> B)    = count(A union B) / N
 *   confidence(A -> B) = count(A union B) / count(A)
 *   lift(A -> B)       = confidence(A -> B) / (count(B) / N)
 *
 * @module week4/script
 */

// Defensive guard: `transactions.js` must run before this file. If `window.HW4`
// is missing (for example because `transactions.js` was not copied next to
// `index.html`), fail loudly and visibly instead of throwing an opaque error.
if (typeof window === "undefined" || typeof window.HW4 === "undefined") {
  if (typeof document !== "undefined") {
    document.body.innerHTML =
      '<p style="padding:2rem;font-family:sans-serif">' +
      "Failed to load `transactions.js`. Make sure it sits next to `index.html` " +
      "and is loaded before `script.js`.</p>";
  }
  throw new Error(
    "window.HW4 is not defined — load transactions.js before script.js.",
  );
}

/** The raw dataset global assigned by `week4/transactions.js`. */
const data = window.HW4;

/**
 * Baskets decoded from the dictionary-encoded arrays in `window.HW4`. Each
 * basket is an array of `{ stock, description }` items, matching the fixture
 * shape used by the tests.
 *
 * @type {Array<Array<Item>>}
 */
const TRANSACTIONS = data.baskets.map((basket) =>
  basket.map((stockIndex) => ({
    stock: data.stocks[stockIndex],
    description: data.descriptions[stockIndex],
  })),
);

/**
 * Dataset-wide inverted index, built once at startup. `null` until `init()` has
 * run, so the UI can tell "still loading" from "loaded".
 *
 * @type {BasketIndex|null}
 */
let DATASET_INDEX = null;

/**
 * Number of baskets (the `N` used by support and lift).
 */
let N = data.N_BASKETS;

/**
 * @typedef {Object} Item
 * @property {string} stock       Stock code (the item identity).
 * @property {string} description Human-readable product description.
 */

/**
 * @typedef {Object} Rule
 * @property {string[]} antecedent    Item identities on the left-hand side (A).
 * @property {string[]} consequent    Item identities on the right-hand side (B).
 * @property {number} jointCount      count(A union B).
 * @property {number} antecedentCount count(A).
 * @property {number} consequentCount count(B).
 * @property {number} support         jointCount / N.
 * @property {number} confidence      jointCount / antecedentCount.
 * @property {number} lift            confidence / (consequentCount / N).
 */

/**
 * @typedef {Object} BasketIndex
 * @property {number} n                              Number of baskets.
 * @property {Map<string, Set<number>>} byStock      Stock code -> basket ids.
 */

// ---------------------------------------------------------------------------
// Provided helpers and student stubs
//
// Functions that were `TODO(hw4)` stubs in the starter are now implemented;
// every other function in this file is scaffolding and was left as provided.
// ---------------------------------------------------------------------------

/**
 * Read the item identity out of a basket entry. Accepts either a plain string or
 * an object with a `stock` property so the helpers work with both the real
 * dataset and the small fixtures used by the tests.
 *
 * @param {Item|string} item
 * @returns {string}
 */
function stockOf(item) {
  return typeof item === "string" ? item : item.stock;
}

/**
 * Build an inverted index from basket id to the set of baskets containing each
 * stock code. Counting a candidate itemset then becomes a set intersection.
 *
 * @param {Array<Array<Item|string>>} baskets
 * @returns {BasketIndex}
 */
function buildIndex(baskets) {
  /** @type {Map<string, Set<number>>} */
  const byStock = new Map();
  baskets.forEach((basket, basketId) => {
    for (const item of basket) {
      const stock = stockOf(item);
      let posting = byStock.get(stock);
      if (!posting) {
        posting = new Set();
        byStock.set(stock, posting);
      }
      posting.add(basketId);
    }
  });
  return { n: baskets.length, byStock };
}

/** Cache of lazily built indexes, keyed by the baskets array. */
const indexCache = new WeakMap();

/**
 * Accept either a ready-made index or a raw basket array and return an index.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @returns {BasketIndex}
 */
function asIndex(basketsOrIndex) {
  if (basketsOrIndex && !Array.isArray(basketsOrIndex) && basketsOrIndex.byStock) {
    return basketsOrIndex;
  }
  let index = indexCache.get(basketsOrIndex);
  if (!index) {
    index = buildIndex(basketsOrIndex);
    indexCache.set(basketsOrIndex, index);
  }
  return index;
}

/**
 * Count the baskets that contain every stock code in `stocks`.
 *
 * HW4 solution: build (or reuse) a basket -> stock inverted index, intersect the
 * posting lists of the requested stocks, and return the size of the
 * intersection.
 *
 * Contract:
 *  - Accept either a ready-made index or a raw basket array. The provided
 *    `asIndex` helper returns an index for either input.
 *  - Return `0` when `stocks` is empty, and also when any requested stock is
 *    absent from the dataset (never throw for an unknown stock).
 *  - A stock repeated within one basket must be counted at most once. Dedupe the
 *    request before intersecting.
 *
 * Algorithm:
 *  1. normalise the request through `dedupeBasket` (so `["A","A","B"]` behaves
 *     exactly like `["A","B"]`);
 *  2. look up each posting list — a missing one means an unknown stock, so the
 *     answer is `0`;
 *  3. sort the posting lists by ascending size so the accumulator stays small;
 *  4. intersect smallest-first and return the final size.
 *
 * A stock repeated *inside* one basket cannot inflate the count: posting lists
 * are `Set<number>` of basket ids, so a repeated code maps to the same id and
 * `Set.add` collapses it.
 *
 * Cost is O(sum of the posting-list lengths actually touched) instead of a
 * O(N * k) scan over all 17,080 baskets per candidate.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {Array<string|Item>} stocks
 * @returns {number} count(A) for a single-element `stocks`, count(A union B) for two.
 */
function countItemset(basketsOrIndex, stocks) {
  const index = asIndex(basketsOrIndex);
  if (!index || !index.byStock) return 0;

  // 1. Normalise + dedupe the request. Non-arrays / malformed entries degrade
  //    to an empty request, which the next step reports as 0.
  const requested = Array.isArray(stocks) ? dedupeBasket(stocks) : [];
  if (requested.length === 0) return 0;

  // 2. Look up every posting list first: one unknown stock makes the whole
  //    itemset uncountable, so answer 0 rather than intersecting partial data.
  const postingLists = [];
  for (const stock of requested) {
    const posting = index.byStock.get(stock);
    if (!posting) return 0;
    postingLists.push(posting);
  }

  // 3. Seed from the smallest posting list.
  postingLists.sort((a, b) => a.size - b.size);
  let intersection = new Set(postingLists[0]);

  // 4. Intersect smallest-first, bailing out as soon as it empties.
  for (let i = 1; i < postingLists.length && intersection.size > 0; i++) {
    const next = new Set();
    const other = postingLists[i];
    for (const basketId of intersection) {
      if (other.has(basketId)) next.add(basketId);
    }
    intersection = next;
  }
  return intersection.size;
}

/**
 * Count the baskets containing a single stock code.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stock
 * @returns {number}
 */
function countItem(basketsOrIndex, stock) {
  return countItemset(basketsOrIndex, [stock]);
}

/**
 * Count the baskets containing both `stockA` and `stockB`.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stockA
 * @param {string|Item} stockB
 * @returns {number}
 */
function countPair(basketsOrIndex, stockA, stockB) {
  return countItemset(basketsOrIndex, [stockA, stockB]);
}

/**
 * Remove repeated item identities from a raw basket, keeping first-appearance
 * order. A basket is a set of items, so duplicates must not be counted twice.
 *
 * HW4 solution: walk the input once, map each entry to its stock code with the
 * provided `stockOf` helper, and return each distinct code the first time it
 * appears.
 *
 * Contract:
 *  - An empty input returns `[]`.
 *  - The input may mix plain strings and `{ stock, description }` objects.
 *  - Only the stock code is returned; the description is dropped.
 *  - A malformed entry (anything whose stock identity is not a non-empty
 *    string) is SKIPPED rather than emitted. This is the one deterministic
 *    defensive behaviour chosen for bad input, so `undefined` / `null` / `""`
 *    codes can never reach the inverted index and silently become an
 *    `"undefined"` map key.
 *
 * @param {Array<string|Item>} rawItems
 * @returns {Array<string>} unique stock codes, in first-appearance order.
 */
function dedupeBasket(rawItems) {
  /** @type {string[]} */
  const unique = [];
  /** @type {Set<string>} */
  const seen = new Set();
  if (!Array.isArray(rawItems)) return unique;

  for (const entry of rawItems) {
    let stock;
    try {
      stock = stockOf(entry);
    } catch (error) {
      stock = undefined; // e.g. a null / undefined entry
    }
    // Reject anything that is not a usable stock-code identity.
    if (typeof stock !== "string" || stock === "") continue;
    if (seen.has(stock)) continue;
    seen.add(stock);
    unique.push(stock);
  }
  return unique;
}

/**
 * Compute support as `jointCount / n`.
 *
 * HW4 solution: return the fraction and flag the `n === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / n` when `n > 0`.
 *  - `defined: false` with `value = 0` when `n === 0` (never divide by zero).
 *
 * @param {number} jointCount count(A union B)
 * @param {number} n          number of baskets
 * @returns {{value: number, defined: boolean}} `defined` is false when `n === 0`.
 */
function computeSupport(jointCount, n) {
  if (typeof n !== "number" || !(n > 0)) return { value: 0, defined: false };
  return { value: jointCount / n, defined: true };
}

/**
 * Compute confidence as `jointCount / antecedentCount`.
 *
 * HW4 solution: return the fraction and flag the `antecedentCount === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / antecedentCount` when count(A) > 0.
 *  - `defined: false` with `value = 0` when count(A) === 0: a rule whose
 *    left-hand side never occurs has no confidence.
 *
 * @param {number} jointCount      count(A union B)
 * @param {number} antecedentCount count(A)
 * @returns {{value: number, defined: boolean}}
 */
function computeConfidence(jointCount, antecedentCount) {
  if (typeof antecedentCount !== "number" || !(antecedentCount > 0)) {
    return { value: 0, defined: false };
  }
  return { value: jointCount / antecedentCount, defined: true };
}

/**
 * Compute lift as `confidence / (consequentCount / n)`.
 *
 * HW4 solution: divide the incoming confidence by the consequent's baseline rate.
 *
 * Contract:
 *  - `defined: true` with `value = confidence.value / (consequentCount / n)`
 *    when every input is usable.
 *  - `defined: false` with `value = 0` when the incoming confidence is missing or
 *    undefined, when `n === 0`, or when the baseline `consequentCount / n` is 0
 *    (the consequent never occurs).
 *
 * @param {{value: number, defined: boolean}} confidence confidence(A -> B)
 * @param {number} consequentCount count(B)
 * @param {number} n               number of baskets
 * @returns {{value: number, defined: boolean}}
 */
function computeLift(confidence, consequentCount, n) {
  // The confidence never happened, so there is nothing to lift.
  if (!confidence || confidence.defined !== true) return { value: 0, defined: false };
  if (typeof n !== "number" || !(n > 0)) return { value: 0, defined: false };
  // The consequent never occurs, so its baseline rate is 0 and the division blows up.
  if (typeof consequentCount !== "number" || !(consequentCount > 0)) {
    return { value: 0, defined: false };
  }
  const baseline = consequentCount / n;
  if (!(baseline > 0)) return { value: 0, defined: false };
  return { value: confidence.value / baseline, defined: true };
}

/**
 * Validate the two slider values, expressed as fractions in `(0, 1]`.
 *
 * @param {number} minSupport    minimum support fraction
 * @param {number} minConfidence minimum confidence fraction
 * @returns {{ok: boolean, errors: string[]}}
 */
function validateThresholds(minSupport, minConfidence) {
  const errors = [];
  for (const [label, value] of [
    ["Minimum support", minSupport],
    ["Minimum confidence", minConfidence],
  ]) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      errors.push(`${label} must be a number.`);
    } else if (value <= 0 || value > 1) {
      errors.push(`${label} must be greater than 0 and at most 1.`);
    }
  }
  return { ok: errors.length === 0, errors };
}

/**
 * Mine all frequent itemsets whose support is at least `minSupport`.
 *
 * HW4 solution: genuine level-wise Apriori — join two frequent (k-1)-itemsets
 * that share their first k-2 items, canonicalise the candidate, drop duplicates,
 * prune any candidate whose (k-1)-subsets are not all frequent (downward
 * closure), then count survivors by posting-list intersection. A level that
 * yields no frequent itemset terminates the run.
 *
 * Contract:
 *  - Return one entry per frequent itemset: `{ items, count, support }`, where
 *    `items` is the (deduplicated) set of stock codes, `count` is the number of
 *    baskets containing every item, and `support = count / N`.
 *  - Every returned itemset must satisfy `support >= minSupport`.
 *  - You may call `countItemset` while developing, but a per-candidate scan over
 *    17,080 baskets is slow — build your own occurrence/index structures.
 *  - `generateRules` consumes this exact shape, so keep the field names stable.
 *
 * Threshold arithmetic (important). `Math.floor(minSupport * N)` must NOT be the
 * acceptance test: it rounds the cutoff *down* and admits below-threshold
 * itemsets (at 3% and N = 17,080, `floor(512.4) = 512`, which admits support
 * 2.9977% into a table advertised as ">= 3%"). So the integer cutoff is used
 * only as a cheap pruning hint, and the *final* guard re-derives the support as
 * `count / N` and compares that. Boundary equality is inclusive, so an itemset
 * whose support equals `minSupport` exactly survives.
 *
 * Determinism: `items` is always sorted with `localeCompare`, the canonical key
 * is derived from that sorted array, candidates are generated in sorted order,
 * and the result is sorted by (size, canonical key). Two runs on the same input
 * therefore produce byte-identical output.
 *
 * @param {Array<Array<Item|string>>} transactions baskets
 * @param {number} minSupport minimum support fraction in `(0, 1]`
 * @returns {Array<{items: string[], count: number, support: number}>} frequent itemsets
 */
function findFrequentItemsets(transactions, minSupport) {
  const threshold = validateThresholds(minSupport, 1);
  if (!threshold.ok) return [];

  const index = asIndex(transactions);
  const n = index.n;
  if (!(n > 0)) return [];

  /** Canonical, stable string key for an itemset (its items are pre-sorted). */
  const keyOf = (items) => items.join("\u0001");
  // Integer pruning hint only. The authoritative test is `count / n >= minSupport`.
  const minCountHint = Math.ceil(minSupport * n - 1e-9);

  // ---- Level 1 -----------------------------------------------------------
  // Derived from posting-list sizes, so no basket scan is needed.
  /** @type {Array<{items: string[], count: number}>} */
  let level = [];
  for (const [stock, posting] of index.byStock) {
    if (posting.size < minCountHint) continue;
    if (!(posting.size / n >= minSupport)) continue; // final guard
    level.push({ items: [stock], count: posting.size });
  }
  level.sort((a, b) => keyOf(a.items).localeCompare(keyOf(b.items)));

  /** @type {Array<{items: string[], count: number, support: number}>} */
  const frequent = [];
  for (const entry of level) {
    frequent.push({ ...entry, support: entry.count / n });
  }

  // ---- Levels k >= 2 -----------------------------------------------------
  for (let k = 2; level.length > 0; k++) {
    // Every (k-1)-subset of a candidate must be in the previous level.
    const previousKeys = new Set(level.map((entry) => keyOf(entry.items)));

    // --- join + canonicalise + dedupe + prune ---
    const candidates = new Map();
    for (let i = 0; i < level.length; i++) {
      for (let j = i + 1; j < level.length; j++) {
        const left = level[i].items;
        const right = level[j].items;

        // Apriori join: the pair must share its first k-2 items.
        let sharePrefix = true;
        for (let p = 0; p < k - 2; p++) {
          if (left[p] !== right[p]) {
            sharePrefix = false;
            break;
          }
        }
        if (!sharePrefix) continue;

        // The new item must extend the prefix; anything else was already joined.
        if (left[k - 2] === right[k - 2]) continue;

        // Canonicalise: a candidate is a SET of stock codes.
        const items = [...new Set([...left, right[k - 2]])].sort((a, b) =>
          a.localeCompare(b),
        );
        if (items.length !== k) continue;

        // Downward-closure pruning.
        let allSubsetsFrequent = true;
        for (let d = 0; d < items.length; d++) {
          const subset =
            d === 0 ? items.slice(1) : [...items.slice(0, d), ...items.slice(d + 1)];
          if (!previousKeys.has(keyOf(subset))) {
            allSubsetsFrequent = false;
            break;
          }
        }
        if (!allSubsetsFrequent) continue;

        candidates.set(keyOf(items), items); // dedupe
      }
    }
    if (candidates.size === 0) break;

    // --- count survivors by posting-list intersection only ---
    /** @type {Array<{items: string[], count: number}>} */
    const next = [];
    for (const items of candidates.values()) {
      const count = countItemset(index, items);
      if (count < minCountHint) continue; // cheap prune
      if (!(count / n >= minSupport)) continue; // FINAL guard
      next.push({ items, count });
    }
    if (next.length === 0) break; // terminate: nothing frequent at this level

    next.sort((a, b) => keyOf(a.items).localeCompare(keyOf(b.items)));
    for (const entry of next) {
      frequent.push({ ...entry, support: entry.count / n });
    }
    level = next;
  }

  // Deterministic overall order: size ascending, then canonical key.
  frequent.sort((a, b) => {
    if (a.items.length !== b.items.length) return a.items.length - b.items.length;
    return keyOf(a.items).localeCompare(keyOf(b.items));
  });
  return frequent;
}

/**
 * Turn frequent itemsets into association rules and keep the ones whose
 * confidence is at least `minConfidence`.
 *
 * HW4 solution: for each frequent itemset, split it into a non-empty antecedent
 * `A` and a non-empty, disjoint consequent `B = X \ A`, compute the confidence
 * for each split, and keep the rules that pass the threshold.
 *
 * EVERY non-empty proper antecedent is enumerated with a bitmask, so a 3-item
 * itemset yields all six directions:
 *
 *     A -> BC      B -> AC      C -> AB
 *     AB -> C      AC -> B      BC -> A
 *
 * A pairwise-only miner (only `A -> B` / `B -> A`) would silently drop the
 * multi-item antecedents and consequents, which is a large share of the real
 * rule table — so the mask runs from `1` to `2^k - 2`, excluding the empty and
 * the full subset and guaranteeing both sides are non-empty.
 *
 * Contract:
 *  - Each returned rule follows the `Rule` shape documented at the top of this
 *    module. At minimum it carries `antecedent` and `consequent`; the renderer
 *    fills in the counts and metrics with `enrichRule`, but returning them
 *    yourself is fine and faster.
 *  - Generate both `A -> B` and `B -> A`: they are separate rules with (usually)
 *    different confidence. Skip a direction whose consequent is empty.
 *  - Keep only rules with `confidence >= minConfidence`. `count(A)` is non-zero
 *    for every generated rule, so the confidence is always defined.
 *
 * NOTE: support and confidence are the only filters here. Rules with
 * `lift <= 1` are deliberately RETAINED so they can be inspected as
 * misleading rules; the `lift > 1` restriction is a downstream business-analysis
 * step, not part of mining.
 *
 * Counts come from the supplied `frequentItemsets` alone, never from a dataset
 * index or from whichever collection happened to be mined most recently.
 *
 * Downward closure guarantees that every non-empty proper subset of a frequent
 * itemset is itself frequent, so a canonical lookup built from the given list
 * already contains `count(A)` and `count(B)` for every split of every itemset X.
 * That makes this a pure function of its two arguments: it cannot be affected by
 * call order, by a later `findFrequentItemsets` call, or by any global state.
 * (An earlier version read counts from a recorded mining source, which silently
 * returned zero rules for the page dataset whenever a fixture had been mined in
 * between.)
 *
 * The three metrics then need no `N` at all:
 *   support(A -> B)    = count(A u B) / N = support(X)      (from the lookup)
 *   confidence(A -> B) = count(X) / count(A)                (from the lookup)
 *   lift(A -> B)       = confidence / [ count(B) / N ]
 *                     = confidence / support(B)             (from the lookup)
 * because support(B) is exactly count(B) / N as computed by the miner.
 *
 * If a required subset is unexpectedly absent (a caller passing a truncated
 * list), that split is SKIPPED deterministically rather than counted against
 * unrelated data. Every subset of a genuine Apriori result is present, so this
 * never fires for well-formed input.
 *
 * @param {Array<{items: string[], count: number, support: number}>} frequentItemsets
 * @param {number} minConfidence minimum confidence fraction in `(0, 1]`
 * @returns {Rule[]}
 */
function generateRules(frequentItemsets, minConfidence) {
  if (!Array.isArray(frequentItemsets) || frequentItemsets.length === 0) return [];
  if (!(typeof minConfidence === "number" && Number.isFinite(minConfidence))) return [];

  const keyOf = (items) => items.join("\u0001");

  // ---- canonical lookup: canonical(items) -> { count, support } -------------
  /** @type {Map<string, {items: string[], count: number, support: number}>} */
  const lookup = new Map();
  for (const itemset of frequentItemsets) {
    if (!itemset || typeof itemset.count !== "number" || !Number.isFinite(itemset.count)) {
      continue;
    }
    const canonical = Array.isArray(itemset.items)
      ? dedupeBasket(itemset.items)
      : [];
    if (canonical.length === 0) continue;
    canonical.sort((a, b) => a.localeCompare(b)); // canonical order
    lookup.set(keyOf(canonical), {
      items: canonical,
      count: itemset.count,
      support:
        typeof itemset.support === "number" && Number.isFinite(itemset.support)
          ? itemset.support
          : 0,
    });
  }
  if (lookup.size === 0) return [];

  /** @type {Rule[]} */
  const rules = [];
  const seen = new Set();

  for (const itemset of lookup.values()) {
    const items = itemset.items;
    const size = items.length;
    if (size < 2) continue; // a 1-itemset has no non-empty proper split

    for (let mask = 1; mask < (1 << size) - 1; mask++) {
      /** @type {string[]} */
      const antecedent = [];
      /** @type {string[]} */
      const consequent = [];
      for (let bit = 0; bit < size; bit++) {
        if (mask & (1 << bit)) antecedent.push(items[bit]);
        else consequent.push(items[bit]);
      }
      // Both sides are non-empty by construction; keep the guard explicit.
      if (antecedent.length === 0 || consequent.length === 0) continue;

      const ruleKey = `${keyOf(antecedent)}|${keyOf(consequent)}`;
      if (seen.has(ruleKey)) continue; // no duplicate rules

      // Counts come only from the supplied frequent itemsets.
      const antecedentEntry = lookup.get(keyOf(antecedent));
      const consequentEntry = lookup.get(keyOf(consequent));
      if (!antecedentEntry || !consequentEntry) continue; // truncated input

      const antecedentCount = antecedentEntry.count;
      const consequentCount = consequentEntry.count;
      if (!(antecedentCount > 0)) continue; // cannot happen for a frequent subset

      const confidence = computeConfidence(itemset.count, antecedentCount);
      if (!confidence.defined) continue;
      if (!(confidence.value >= minConfidence)) continue; // inclusive boundary

      // support(X) = count(X) / N, supplied by the miner.
      const support = itemset.support;
      // lift = confidence / support(B), and support(B) = count(B) / N.
      const baseline = consequentEntry.support;
      const lift =
        baseline > 0
          ? { value: confidence.value / baseline, defined: true }
          : { value: 0, defined: false };

      seen.add(ruleKey);
      rules.push({
        antecedent,
        consequent,
        jointCount: itemset.count,
        antecedentCount,
        consequentCount,
        support,
        confidence: confidence.value,
        lift: lift.value,
      });
    }
  }

  // Deterministic order: antecedent length, antecedent lexicographic,
  // consequent length, consequent lexicographic.
  rules.sort((a, b) => {
    if (a.antecedent.length !== b.antecedent.length) {
      return a.antecedent.length - b.antecedent.length;
    }
    const byA = keyOf(a.antecedent).localeCompare(keyOf(b.antecedent));
    if (byA !== 0) return byA;
    if (a.consequent.length !== b.consequent.length) {
      return a.consequent.length - b.consequent.length;
    }
    return keyOf(a.consequent).localeCompare(keyOf(b.consequent));
  });
  return rules;
}

// ---------------------------------------------------------------------------
// Tiny worked example (used by the automated tests and the readout panel)
// ---------------------------------------------------------------------------

/**
 * A five-basket fixture with hand-computed support / confidence / lift values.
 *
 * The fixture is deliberately small enough to check with a pencil:
 *
 *   T1: bread, milk, jam, ham
 *   T2: bread, milk, jam
 *   T3: bread, milk
 *   T4: bread, jam, eggs
 *   T5: bread, eggs
 *
 * Item counts: bread = 5, milk = 3, jam = 3, eggs = 2, ham = 1 (N = 5).
 *
 * Because `bread` appears in every basket, every rule with `bread` on either
 * side has lift exactly 1 — that is the teaching point of the fixture.
 *
 * @returns {{n: number, baskets: string[][], itemCounts: Object<string, number>, rules: Array<Object>}}
 */
function tinyWorkedExample() {
  const baskets = [
    ["bread", "milk", "jam", "ham"],
    ["bread", "milk", "jam"],
    ["bread", "milk"],
    ["bread", "jam", "eggs"],
    ["bread", "eggs"],
  ];
  const n = baskets.length;
  return {
    n,
    baskets,
    itemCounts: { bread: 5, milk: 3, jam: 3, eggs: 2, ham: 1 },
    rules: [
      // lift == 1: bread is in every basket, so confidence equals support(B).
      {
        antecedent: ["bread"],
        consequent: ["milk"],
        jointCount: 3,
        antecedentCount: 5,
        consequentCount: 3,
        support: 3 / n,
        confidence: 3 / 5,
        lift: 1,
      },
      // lift > 1: milk and jam co-occur more than independence predicts.
      {
        antecedent: ["milk"],
        consequent: ["jam"],
        jointCount: 2,
        antecedentCount: 3,
        consequentCount: 3,
        support: 2 / n,
        confidence: 2 / 3,
        lift: (2 / 3) / (3 / 5),
      },
      // lift < 1 (and not degenerate): jam and eggs co-occur less than expected.
      {
        antecedent: ["jam"],
        consequent: ["eggs"],
        jointCount: 1,
        antecedentCount: 3,
        consequentCount: 2,
        support: 1 / n,
        confidence: 1 / 3,
        lift: (1 / 3) / (2 / 5),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Rule normalisation
// ---------------------------------------------------------------------------

/**
 * Fill in any missing counts / metrics on a rule using the dataset index, so the
 * table can render a rule even when the student returns only `antecedent` and
 * `consequent`.
 *
 * @param {Partial<Rule>} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function enrichRule(rule, index) {
  const antecedent = (rule.antecedent || []).map(stockOf);
  const consequent = (rule.consequent || []).map(stockOf);
  // Nullish, not `||`: an empty index has n === 0, which is falsy but valid
  // and must not silently fall back to the global N.
  const n = index && index.n !== undefined ? index.n : N;
  const jointCount =
    typeof rule.jointCount === "number"
      ? rule.jointCount
      : countItemset(index, [...antecedent, ...consequent]);
  const antecedentCount =
    typeof rule.antecedentCount === "number"
      ? rule.antecedentCount
      : countItemset(index, antecedent);
  const consequentCount =
    typeof rule.consequentCount === "number"
      ? rule.consequentCount
      : countItemset(index, consequent);
  const support = computeSupport(jointCount, n);
  const confidence = computeConfidence(jointCount, antecedentCount);
  const lift = computeLift(confidence, consequentCount, n);
  return {
    antecedent,
    consequent,
    jointCount,
    antecedentCount,
    consequentCount,
    support: typeof rule.support === "number" ? rule.support : support.value,
    confidence:
      typeof rule.confidence === "number" ? rule.confidence : confidence.value,
    lift: typeof rule.lift === "number" ? rule.lift : lift.value,
    supportDefined: support.defined,
    confidenceDefined: confidence.defined,
    liftDefined: lift.defined,
  };
}

/**
 * Swap the antecedent and consequent of a rule and recompute the metrics.
 *
 * Confidence is NOT symmetric: `B -> A` usually has a different confidence from
 * `A -> B`, because the denominator `count(A)` becomes `count(B)`.
 *
 * Support and lift ARE symmetric and must not change:
 *   - `support(A -> B) = count(A union B) / N`, and `A union B === B union A`,
 *     so `support(A -> B) === support(B -> A)`.
 *   - `lift(A -> B) = (count(A union B) / count(A)) / (count(B) / N)`
 *                = count(A union B) * N / (count(A) * count(B)`,
 *     which is symmetric in A and B, so `lift(A -> B) === lift(B -> A)`.
 *
 * (An earlier version of this comment wrongly claimed support also changed with
 * direction. It does not, and the renderer below demonstrates exactly that.)
 *
 * @param {Rule} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function reverseRule(rule, index) {
  return enrichRule(
    {
      antecedent: rule.consequent,
      consequent: rule.antecedent,
    },
    index,
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format a fraction as a percentage with two decimals.
 *
 * Non-finite input (NaN / Infinity) returns `"n/a"`, matching `formatMetric`, so
 * an undefined metric can never surface as a literal `"NaN%"` in the table.
 *
 * @param {number} fraction
 * @returns {string}
 */
function formatPercent(fraction) {
  if (!Number.isFinite(fraction)) return "n/a";
  return `${(fraction * 100).toFixed(2)}%`;
}

/**
 * Format a number with four significant decimals for the results table.
 *
 * @param {number} value
 * @returns {string}
 */
function formatMetric(value) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(4);
}

/**
 * Render a list of stock codes as readable text.
 *
 * @param {string[]} stocks
 * @param {BasketIndex} index
 * @returns {string}
 */
function formatItemset(stocks, index) {
  if (!stocks || stocks.length === 0) return "(empty)";
  return stocks.map((stock) => describe(stock, index)).join(", ");
}

/**
 * Lookup a stock code's canonical description, falling back to the code. Filled
 * by `primeDescriptions()` once `init()` runs.
 */
const descriptionByStock = new Map();

/**
 * Render `STOCK — human readable description` for one item.
 *
 * @param {string} stock
 * @param {BasketIndex} index
 * @returns {string}
 */
function describe(stock, index) {
  const description = descriptionByStock.get(stock);
  return description ? `${stock} — ${description}` : stock;
}

/**
 * Render one itemset as HTML: the product description leads, and the stock code
 * sits beneath it as secondary monospace text. Presentation only — the text
 * content is exactly what `describe()` produces, so nothing is added or lost.
 *
 * @param {string[]} stocks
 * @param {BasketIndex} index
 * @param {boolean} [muted] render the description in muted ink
 * @returns {string} HTML fragment
 */
function formatItemsetHtml(stocks, index, muted) {
  if (!stocks || stocks.length === 0) {
    return '<span class="item-cell__desc">(empty)</span>';
  }
  const className = muted ? "item-cell item-cell--muted" : "item-cell";
  return stocks
    .map((stock) => {
      const description = descriptionByStock.get(stock);
      const descHtml = description
        ? `<span class="item-cell__desc">${escapeHtml(description)}</span>`
        : "";
      return `<span class="${className}"><span class="item-cell__code">${escapeHtml(
        stock,
      )}</span>${descHtml}</span>`;
    })
    .join(", ");
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Render the dataset summary at the top of the page: total baskets, distinct
 * items, and the top five items by basket count.
 *
 * @param {BasketIndex} index
 * @param {HTMLElement|null} container
 * @returns {void}
 */
function renderDatasetSummary(index, container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;

  const counts = [...index.byStock.entries()]
    .map(([stock, posting]) => ({ stock, count: posting.size }))
    .sort((a, b) => b.count - a.count || a.stock.localeCompare(b.stock));
  const topFive = counts.slice(0, 5);

  const stat = (label, value) =>
    `<div class="stat"><dt class="stat__label">${escapeHtml(
      label,
    )}</dt><dd class="stat__value">${escapeHtml(String(value))}</dd></div>`;

  const topRows = topFive
    .map(
      (entry, i) =>
        `<tr><td class="num">${i + 1}</td><td>${escapeHtml(
          describe(entry.stock, index),
        )}</td><td class="num">${entry.count}</td></tr>`,
    )
    .join("");

  const stats = [
    stat("Baskets", index.n.toLocaleString("en-US")),
    stat("Distinct items", index.byStock.size.toLocaleString("en-US")),
    stat("Most frequent item", `${counts.length ? counts[0].count : 0}`),
  ].join("");

  target.innerHTML = `
    <dl class="stats">${stats}</dl>
    <details class="top-items">
      <summary>Top 5 items by basket count</summary>
      <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th scope="col" class="num">#</th><th scope="col">Item</th><th scope="col" class="num">Baskets</th></tr></thead>
          <tbody>${topRows}</tbody>
        </table>
      </div>
    </details>
    <p class="provenance">${escapeHtml(data.dataset_provenance)}</p>
  `;
}

/**
 * Reset the "Selected rule" panel back to its empty state.
 *
 * Called whenever a new rule table is rendered, so a selection made under the
 * previous thresholds can never be left on screen describing a rule that is no
 * longer in the current table.
 *
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function resetRuleDetail(container) {
  const target = container || document.getElementById("rule-detail");
  if (!target) return;
  target.innerHTML =
    '<p class="empty-state">Click a rule row to inspect it.</p>';
}

/**
 * Render the rule list into a table. Each row is clickable and shows the rule in
 * the detail panel.
 *
 * Rendering a new table always resets the detail panel, because the previously
 * selected rule belongs to the previous threshold setting.
 *
 * @param {Rule[]} rules
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderResults(rules, index, container) {
  const target = container || document.getElementById("results");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;

  // A new table invalidates the previous selection.
  if (!container) resetRuleDetail();

  if (!rules || rules.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No rules passed the current thresholds. ' +
      "Lower the minimum support or confidence and run again.</p>";
    return;
  }

  const body = rules
    .map((rule, rowIndex) => {
      const enriched = enrichRule(rule, activeIndex);
      return `
        <tr tabindex="0" data-rule-index="${rowIndex}" aria-selected="false">
          <td class="cell-itemset">${formatItemsetHtml(enriched.antecedent, activeIndex)}</td>
          <td class="cell-itemset">${formatItemsetHtml(enriched.consequent, activeIndex, true)}</td>
          <td class="num">${enriched.jointCount}</td>
          <td class="num">${enriched.antecedentCount}</td>
          <td class="num">${enriched.consequentCount}</td>
          <td class="num">${formatPercent(enriched.support)}</td>
          <td class="num">${formatPercent(enriched.confidence)}</td>
          <td class="num">${formatMetric(enriched.lift)}</td>
        </tr>`;
    })
    .join("");

  target.innerHTML = `
    <p class="results-count">${rules.length} rule${rules.length === 1 ? "" : "s"}</p>
    <table class="data-table rules-table">
      <thead>
        <tr>
          <th scope="col">Antecedent (A)</th>
          <th scope="col">Consequent (B)</th>
          <th scope="col" class="num">Joint</th>
          <th scope="col" class="num">Count(A)</th>
          <th scope="col" class="num">Count(B)</th>
          <th scope="col" class="num">Support</th>
          <th scope="col" class="num">Confidence</th>
          <th scope="col" class="num">Lift</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>`;

  target.querySelectorAll("tr[data-rule-index]").forEach((row) => {
    const activate = () => {
      const rule = rules[Number(row.dataset.ruleIndex)];
      // Mark the active row so the selection is visible as well as announced.
      target.querySelectorAll("tr[data-rule-index]").forEach((other) => {
        other.setAttribute("aria-selected", other === row ? "true" : "false");
      });
      renderRuleDetail(rule, activeIndex);
    };
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });
}

/**
 * Render the detail panel for one selected rule, including a "Reverse direction"
 * button that swaps A and B and re-renders the panel.
 *
 * Shows an inline note when `count(A)` or `count(B)` is zero, because confidence
 * and lift are then undefined.
 *
 * @param {Rule} rule
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRuleDetail(rule, index, container) {
  const target = container || document.getElementById("rule-detail");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;
  const enriched = enrichRule(rule, activeIndex);
  const reversed = reverseRule(enriched, activeIndex);

  const zeroDenominatorNotes = [];
  if (enriched.antecedentCount === 0) {
    zeroDenominatorNotes.push(
      "count(A) = 0, so confidence(A \u2192 B) is undefined (division by zero).",
    );
  }
  if (enriched.consequentCount === 0) {
    zeroDenominatorNotes.push(
      "count(B) = 0, so lift(A \u2192 B) is undefined (division by zero).",
    );
  }
  const noteHtml = zeroDenominatorNotes.length
    ? `<p class="warning">${zeroDenominatorNotes.map(escapeHtml).join(" ")}</p>`
    : "";

  const metric = (label, value, note) => `
    <div class="metric">
      <p class="metric__label">${escapeHtml(label)}</p>
      <p class="metric__value">${escapeHtml(String(value))}</p>
      <p class="metric__note">${escapeHtml(note)}</p>
    </div>`;

  const evidence = (label, value) => `
    <div>
      <p class="evidence__label">${escapeHtml(label)}</p>
      <p class="evidence__value">${escapeHtml(String(value))}</p>
    </div>`;

  const side = (label, stocks) => `
    <div class="rule-side">
      <p class="rule-side__label">${escapeHtml(label)}</p>
      ${formatItemsetHtml(stocks, activeIndex, label.indexOf("B") !== -1)}
    </div>`;

  target.innerHTML = `
    <div class="rule-flow">
      ${side("Antecedent (A)", enriched.antecedent)}
      <p class="rule-arrow" aria-hidden="true">&#8595;</p>
      ${side("Consequent (B)", enriched.consequent)}
    </div>

    <div class="metric-grid">
      ${metric("Support", formatPercent(enriched.support), "Share of all baskets containing the full itemset.")}
      ${metric("Confidence", formatPercent(enriched.confidence), "P(B | A): share of A baskets that also contain B.")}
      ${metric("Lift", formatMetric(enriched.lift), "Confidence divided by the baseline frequency of B.")}
    </div>

    <dl class="evidence">
      ${evidence("Joint baskets", enriched.jointCount)}
      ${evidence("Count(A)", enriched.antecedentCount)}
      ${evidence("Count(B)", enriched.consequentCount)}
    </dl>

    <p class="comparison">
      Reverse direction (B \u2192 A): confidence
      <strong>${formatPercent(reversed.confidence)}</strong>, lift
      <strong>${formatMetric(reversed.lift)}</strong>.
      Confidence changes with direction; support and lift do not.
    </p>
    ${noteHtml}
    <p class="detail-actions">
      <button type="button" id="reverse-rule">Reverse direction (B \u2192 A)</button>
    </p>
  `;

  const button = target.querySelector("#reverse-rule");
  button.addEventListener("click", () => {
    renderRuleDetail(reversed, activeIndex, target);
  });
}

/**
 * Render the optional worked-example readout panel.
 *
 * @param {ReturnType<typeof tinyWorkedExample>} example
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderWorkedExample(example, container) {
  const target = container || document.getElementById("worked-example");
  if (!target) return;
  const rows = example.rules
    .map(
      (rule) => `
      <tr>
        <td>${rule.antecedent.join(", ")}</td>
        <td>${rule.consequent.join(", ")}</td>
        <td class="num">${rule.jointCount}</td>
        <td class="num">${rule.antecedentCount}</td>
        <td class="num">${rule.consequentCount}</td>
        <td class="num">${formatPercent(rule.support)}</td>
        <td class="num">${formatPercent(rule.confidence)}</td>
        <td class="num">${formatMetric(rule.lift)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <p class="worked-example-intro">
      Five hand-built baskets, N = ${example.n}. The top row has lift exactly 1
      because <code>bread</code> appears in every basket; the second has lift &gt; 1;
      the third has lift &lt; 1.
    </p>
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">A</th><th scope="col">B</th>
          <th scope="col">count(A∪B)</th><th scope="col">count(A)</th><th scope="col">count(B)</th>
          <th scope="col">support</th><th scope="col">confidence</th><th scope="col">lift</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker thrown by unimplemented `TODO(hw4)` stubs. */
const TODO_MARKER = "TODO(hw4)";

/**
 * Run the automated self-checks against the five-basket worked example and the
 * reference helpers. Unimplemented `TODO(hw4)` functions are reported as
 * `pending` rather than `fail`, so the harness is useful before and after the
 * assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const example = tinyWorkedExample();
  const checks = [];
  const basketObjects = example.baskets;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw4): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Item counts on the fixture.
  check("fixture: item counts match the hand-computed values", () => {
    for (const [stock, expected] of Object.entries(example.itemCounts)) {
      const actual = countItem(basketObjects, stock);
      if (actual !== expected) {
        throw new Error(`countItem(${stock}) = ${actual}, expected ${expected}`);
      }
    }
    return "all five item counts correct";
  });

  // 2. lift == 1 case.
  check("fixture: bread -> milk has lift exactly 1", () => {
    const [breadMilk] = example.rules;
    const joint = countPair(basketObjects, "bread", "milk");
    const support = computeSupport(joint, example.n);
    const confidence = computeConfidence(joint, countItem(basketObjects, "bread"));
    const lift = computeLift(confidence, countItem(basketObjects, "milk"), example.n);
    if (!close(support.value, breadMilk.support)) throw new Error("support mismatch");
    if (!close(confidence.value, breadMilk.confidence)) throw new Error("confidence mismatch");
    if (!close(lift.value, breadMilk.lift)) throw new Error(`lift = ${lift.value}, expected 1`);
    return `support=${support.value.toFixed(4)} confidence=${confidence.value.toFixed(4)} lift=${lift.value.toFixed(4)}`;
  });

  // 3. lift > 1 case.
  check("fixture: milk -> jam has lift greater than 1", () => {
    const rule = example.rules[1];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value > 1)) throw new Error("expected lift > 1");
    return `lift=${lift.value.toFixed(4)} (> 1)`;
  });

  // 4. lift < 1 case.
  check("fixture: jam -> eggs has lift less than 1", () => {
    const rule = example.rules[2];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value < 1 && lift.value > 0)) throw new Error("expected 0 < lift < 1");
    return `lift=${lift.value.toFixed(4)} (0 < lift < 1)`;
  });

  // 5. Reversed confidence differs.
  check("fixture: reversed confidence differs (jam <-> eggs)", () => {
    const forwardJoint = countPair(basketObjects, "jam", "eggs");
    const forward = computeConfidence(forwardJoint, countItem(basketObjects, "jam"));
    const reversed = computeConfidence(forwardJoint, countItem(basketObjects, "eggs"));
    if (close(forward.value, reversed.value)) {
      throw new Error("confidence should differ between jam -> eggs and eggs -> jam");
    }
    if (!close(forward.value, 1 / 3) || !close(reversed.value, 1 / 2)) {
      throw new Error(`unexpected values: ${forward.value} vs ${reversed.value}`);
    }
    return `jam->eggs=${forward.value.toFixed(4)} vs eggs->jam=${reversed.value.toFixed(4)}`;
  });

  // 6. Duplicate items in a raw basket are counted once.
  check("duplicate items in a raw basket are counted once", () => {
    const deduped = dedupeBasket(["milk", "bread", "milk", "bread", "jam"]);
    if (deduped.length !== 3) throw new Error(`expected 3 unique items, got ${deduped.length}`);
    const duplicateBasket = [["milk", "bread", "milk", "bread"], ["milk"]];
    if (countItem(duplicateBasket, "milk") !== 2) {
      throw new Error("countItem must count each basket once, not each row");
    }
    if (countPair(duplicateBasket, "milk", "bread") !== 1) {
      throw new Error("countPair must count each basket once");
    }
    return "dedupeBasket removed repeats; counting is per basket";
  });

  // 7. Empty result set renders an empty-state note.
  check("empty rule set renders an empty-state note", () => {
    const scratch = document.createElement("div");
    renderResults([], DATASET_INDEX, scratch);
    if (!/no rules passed/i.test(scratch.textContent)) {
      throw new Error("expected an empty-state message");
    }
    return "empty state rendered";
  });

  // 8. Invalid thresholds are rejected.
  check("invalid thresholds are rejected", () => {
    if (validateThresholds(0.01, 0.3).ok !== true) throw new Error("valid thresholds rejected");
    if (validateThresholds(Number.NaN, 0.3).ok !== false) throw new Error("NaN support accepted");
    if (validateThresholds(0.01, 1.5).ok !== false) throw new Error("confidence > 1 accepted");
    if (validateThresholds(0, 0.3).ok !== false) throw new Error("zero support accepted");
    return "valid accepted, invalid rejected";
  });

  // 9. Zero-denominator guards.
  check("zero-denominator guards return undefined metrics", () => {
    if (computeConfidence(0, 0).defined !== false) throw new Error("count(A)=0 must be undefined");
    if (computeLift({ value: 0.5, defined: true }, 0, 5).defined !== false) {
      throw new Error("count(B)=0 must be undefined");
    }
    if (computeSupport(0, 0).defined !== false) throw new Error("N=0 must be undefined");
    return "zero denominators return defined=false";
  });

  // 10. Student function: frequent itemsets (pending until implemented).
  check("findFrequentItemsets reproduces the fixture's frequent itemsets", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.4);
    const pairs = itemsets.filter((set) => set.items.length === 2);
    if (pairs.length === 0) throw new Error("no frequent pairs found at support >= 0.4");
    return `${itemsets.length} itemsets`;
  });

  // 11. Student function: rules (pending until implemented).
  check("generateRules reproduces the fixture's rules", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.2);
    const rules = generateRules(itemsets, 0.5);
    if (rules.length === 0) throw new Error("no rules found at support >= 0.2, confidence >= 0.5");
    return `${rules.length} rules`;
  });

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW4 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  // Visual summary only. The log text above is written unchanged.
  const summary = document.getElementById("test-summary");
  if (summary) {
    const pill = (kind, count, label) =>
      `<span class="qa-pill qa-pill--${kind}">${label} <span class="qa-pill__count">${count}</span></span>`;
    summary.innerHTML = [
      pill("pass", passed, "Passed"),
      pill("fail", failed, "Failed"),
      pill("pending", pendingCount, "Pending"),
      `<span class="qa-pill">Checks <span class="qa-pill__count">${checks.length}</span></span>`,
    ].join("");
  }

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/**
 * Populate the stock -> description lookup used by the renderers from the
 * dictionary-encoded tables embedded in `window.HW4`.
 *
 * @returns {void}
 */
function primeDescriptions() {
  data.stocks.forEach((stock, i) => {
    if (!descriptionByStock.has(stock)) {
      descriptionByStock.set(stock, data.descriptions[i]);
    }
  });
}

/**
 * Read the two sliders and return the thresholds as fractions in `(0, 1]`.
 *
 * @returns {{minSupport: number, minConfidence: number}}
 */
function readThresholds() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  return {
    minSupport: Number(supportInput.value) / 100,
    minConfidence: Number(confidenceInput.value) / 100,
  };
}

/** Update the `<output>` readouts next to the two sliders. */
function syncThresholdLabels() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  const supportOutput = document.getElementById("min-support-value");
  const confidenceOutput = document.getElementById("min-confidence-value");
  if (supportOutput) supportOutput.textContent = `${Number(supportInput.value).toFixed(1)}%`;
  if (confidenceOutput) confidenceOutput.textContent = `${Number(confidenceInput.value).toFixed(0)}%`;
}

/**
 * Read the thresholds, run the student pipeline, and render the results.
 *
 * @returns {void}
 */
function runPipeline() {
  const status = document.getElementById("status");
  if (!DATASET_INDEX) {
    if (status) {
      status.textContent = "Dataset not ready yet — reload the page.";
    }
    return;
  }
  const { minSupport, minConfidence } = readThresholds();
  const validation = validateThresholds(minSupport, minConfidence);
  if (!validation.ok) {
    if (status) status.textContent = validation.errors.join(" ");
    return;
  }
  try {
    if (status) status.textContent = "Mining frequent itemsets…";
    const itemsets = findFrequentItemsets(TRANSACTIONS, minSupport);
    const rules = generateRules(itemsets, minConfidence);
    renderResults(rules, DATASET_INDEX);
    if (status) status.textContent = `Done — ${rules.length} rule(s) at support \u2265 ${(minSupport * 100).toFixed(1)}% and confidence \u2265 ${(minConfidence * 100).toFixed(0)}%.`;
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    const resultsEl = document.getElementById("results");
    if (resultsEl) {
      resultsEl.innerHTML =
        '<p class="empty-state">Run failed &mdash; the rule miner did not complete. ' +
        "Implement the <code>TODO(hw4)</code> functions, then press &ldquo;Run rules&rdquo;. " +
        `Error: ${escapeHtml(message)}</p>`;
    }
    if (status) status.textContent = message;
  }
}

/**
 * Wire the controls once the DOM is ready, then build the dataset index from the
 * baskets already decoded from `window.HW4` and render the summary.
 *
 * The data is already in memory (embedded by `transactions.js`), so there is no
 * fetch and no error path beyond the `window.HW4` guard at the top of this file.
 *
 * @returns {void}
 */
function init() {
  renderWorkedExample(tinyWorkedExample());
  syncThresholdLabels();

  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  if (supportInput) supportInput.addEventListener("input", syncThresholdLabels);
  if (confidenceInput) confidenceInput.addEventListener("input", syncThresholdLabels);

  const runButton = document.getElementById("run-rules");
  if (runButton) runButton.addEventListener("click", runPipeline);

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  const status = document.getElementById("status");
  primeDescriptions();
  N = TRANSACTIONS.length;
  DATASET_INDEX = buildIndex(TRANSACTIONS);

  // Header metadata chips, read straight from the dataset global.
  const basketsChip = document.getElementById("chip-baskets");
  const itemsChip = document.getElementById("chip-items");
  if (basketsChip) basketsChip.textContent = N.toLocaleString("en-US");
  if (itemsChip) {
    itemsChip.textContent = (DATASET_INDEX.byStock.size).toLocaleString("en-US");
  }
  try {
    renderDatasetSummary(DATASET_INDEX);
  } catch (error) {
    // `renderDatasetSummary` is scaffolding, but guard it defensively so that an
    // unimplemented TODO(hw4) stub can never take the whole page down at load
    // time. The harness's "Run tests" button stays usable regardless.
    const summary = document.getElementById("dataset-summary-body");
    if (summary) {
      summary.innerHTML =
        '<p class="empty-state">Implement the TODO(hw4) functions to activate ' +
        "the dataset summary.</p>";
    }
  }
  if (status) {
    status.textContent = `Dataset ready: ${N.toLocaleString("en-US")} baskets, ${data.N_ITEMS.toLocaleString("en-US")} distinct items. Set the thresholds and press “Run rules”.`;
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}
