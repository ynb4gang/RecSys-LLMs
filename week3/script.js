'use strict';

// ===========================================================================
// script.js — UI and collaborative-filtering logic
//
// ===========================================================================
// CHOSEN MISSING-VALUE STRATEGY (EXACTLY ONE)
// ===========================================================================
//
//   SUPPORT-WEIGHTED CO-RATED COSINE
//
//   Missing ratings stay MISSING. Nothing is imputed, nothing is mean-centred,
//   and there is no latent-factor model. Similarity is computed only from
//   entries that BOTH sides actually rated, then discounted by how many
//   co-rated entries backed that estimate:
//
//       rawCosine       = dot(A_common, B_common)
//                        / ( ||A_common|| * ||B_common|| )
//
//       supportWeight   = nCommon / (nCommon + LAMBDA)
//
//       finalSimilarity = rawCosine * supportWeight        (0 if nCommon < MIN_COMMON)
//
//   Why the support weight is needed on MovieLens 100K: ratings live in 1..5,
//   so every rating is strictly positive. Cosine is scale invariant, and for a
//   single co-rated entry the numerator and the denominator are the same two
//   numbers, so cosine is EXACTLY 1.0 no matter how large the disagreement is.
//   A user who rated a movie 1 and another who rated it 5 score as perfectly
//   similar. nCommon/(nCommon+LAMBDA) is a significance weight: it is a
//   monotone shrinkage heuristic, near 0 for a one-entry coincidence and
//   approaching 1 as evidence accumulates, so thin evidence is discounted
//   rather than trusted.
//
//   MIN_COMMON does a different job from the weight. The weight is continuous
//   and order preserving (n=1 still scores 1/11 = 0.0909, which can outrank a
//   legitimate n=3 pair with a low raw cosine). The MIN_COMMON gate is
//   categorical: below the floor, similarity is exactly 0 because no claim is
//   made at all. Together they remove the degenerate tail and rank the rest.
//
//   The two rejected alternatives are NOT implemented and must not be mixed in:
//     mean imputation      fabricates values in the 93.7% of cells that are
//                          empty, destroys the 0 = missing sentinel, and
//                          creates artificial agreement;
//     matrix factorization out of scope for similarity-based CF, and a hand
//                          rolled SGD trainer is far beyond this assignment.
//
// ===========================================================================

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const LAMBDA = 10;            // significance weight denominator
const MIN_COMMON = 3;         // minimum co-rated entries to trust a similarity
const MIN_PREDICTORS = 2;     // minimum contributors to trust a prediction
const USER_NEIGHBOURS = 20;   // neighbour cut for User-Based CF
const TOP_K = 5;              // recommendations returned per approach

// ---------------------------------------------------------------------------
// Runtime is a DIAGNOSTIC ONLY. It is never part of any ranking decision and
// never influences a predicted score.
// ---------------------------------------------------------------------------
let lastRun = { user: null, item: null };

// ===========================================================================
// Similarity
// ===========================================================================

/**
 * Support-weighted co-rated cosine between two sparse rating maps.
 *
 * Both arguments are Map-like: keys are the co-rated entity (a movieId for
 * user-user similarity, a userId for item-item similarity) and values are the
 * 1..5 ratings. Only genuinely shared keys are used; nothing is imputed.
 *
 * Iterates the SMALLER map and probes the larger one, so cost is
 * O(min(|A|, |B|)) rather than O(I) or O(U).
 *
 * @returns {{rawCosine: number, similarity: number, commonCount: number}}
 *   rawCosine is reported even below MIN_COMMON so that the diagnostics can
 *   show the degenerate one-overlap value of 1.0 that the gate then discards.
 *   `similarity` is the gated, support-weighted value actually used for ranking.
 */
function getSimilarityDetails(mapA, mapB) {
    const empty = { rawCosine: 0, similarity: 0, commonCount: 0 };
    if (!mapA || !mapB || mapA.size === 0 || mapB.size === 0) return empty;

    // Iterate the smaller side, probe the larger side.
    const small = (mapA.size <= mapB.size) ? mapA : mapB;
    const large = (mapA.size <= mapB.size) ? mapB : mapA;

    let dot = 0;
    let sumSquaresSmall = 0;
    let sumSquaresLarge = 0;
    let commonCount = 0;

    for (const [key, valueSmall] of small) {
        const valueLarge = large.get(key);
        if (valueLarge === undefined) continue;
        commonCount++;
        dot += valueSmall * valueLarge;
        sumSquaresSmall += valueSmall * valueSmall;
        sumSquaresLarge += valueLarge * valueLarge;
    }

    if (commonCount === 0) return empty;

    const normProduct = Math.sqrt(sumSquaresSmall) * Math.sqrt(sumSquaresLarge);

    let rawCosine = 0;
    if (normProduct > 0) {
        rawCosine = dot / normProduct;
        if (!Number.isFinite(rawCosine)) rawCosine = 0;
    }

    let similarity = 0;
    if (commonCount >= MIN_COMMON && normProduct > 0) {
        const supportWeight = commonCount / (commonCount + LAMBDA);
        similarity = rawCosine * supportWeight;
        if (!Number.isFinite(similarity)) similarity = 0;
    }

    return { rawCosine: rawCosine, similarity: similarity, commonCount: commonCount };
}

/**
 * Assignment-facing similarity entry point.
 *
 * Returns the FINAL support-weighted similarity by delegating to
 * getSimilarityDetails. The production hot path calls getSimilarityDetails
 * directly, because this wrapper also has to accept the dense-array form
 * documented in the assignment ("two arrays of equal length, slice the rating
 * matrix column or row"). Dense arrays use 0 to mean "not rated", so the
 * non-zero entries are exactly the co-rated entries.
 *
 * @param {Map|Array} a  sparse map, or a dense array using 0 for "not rated"
 * @param {Map|Array} b  the other side, same form
 * @returns {number} final support-weighted similarity in [0, 1]
 */
function cosineSimilarity(a, b) {
    if (a instanceof Map && b instanceof Map) {
        return getSimilarityDetails(a, b).similarity;
    }
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) {
            throw new Error('cosineSimilarity: arrays must have equal length.');
        }
        const sparseA = new Map();
        const sparseB = new Map();
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== 0) sparseA.set(i, a[i]);
            if (b[i] !== 0) sparseB.set(i, b[i]);
        }
        return getSimilarityDetails(sparseA, sparseB).similarity;
    }
    throw new Error('cosineSimilarity: expected two Maps or two equal-length arrays.');
}

// ===========================================================================
// User-Based CF
// ===========================================================================

/**
 * Top-K recommendations from the most similar USERS.
 *
 *   1. support-weighted cosine against every other actual user id
 *   2. keep similarity > 0, sort deterministically, take the top USER_NEIGHBOURS
 *   3. for every movie the active user has NOT rated, predict
 *          score = SUM(similarity * neighbourRating) / SUM(similarity)
 *      counting how many neighbours contributed as evidenceCount
 *   4. require denominator > 0 and evidenceCount >= MIN_PREDICTORS
 *   5. sort by score DESC, evidenceCount DESC, movieId ASC and take TOP_K
 *
 * Already-rated movies are excluded structurally: the accumulator is only ever
 * touched for movie ids that are not in the active user's row.
 *
 * @returns {Array<{movieId, title, score, evidenceCount}>} up to TOP_K rows.
 *   Runtime and counters are published on lastRun.user for the diagnostics panel.
 */
function getUserBasedRecommendations(activeUserId, topK) {
    const k = (topK === undefined) ? TOP_K : topK;
    const startedAt = performance.now();

    const activeRatings = userRatings.get(activeUserId);
    if (!activeRatings || activeRatings.size === 0) {
        // Cold start: no rating history means no rating vector to compare.
        lastRun.user = {
            runtimeMs: performance.now() - startedAt,
            eligibleNeighbours: 0,
            neighboursUsed: 0,
            candidatesConsidered: 0,
            coldStart: true
        };
        return [];
    }

    // --- 1 & 2: neighbours ------------------------------------------------
    const neighbours = [];
    let eligibleNeighbours = 0;
    for (const otherUserId of userIds) {
        if (otherUserId === activeUserId) continue;
        const details = getSimilarityDetails(activeRatings, userRatings.get(otherUserId));
        if (details.similarity > 0) {
            eligibleNeighbours++;
            neighbours.push({
                userId: otherUserId,
                similarity: details.similarity,
                commonCount: details.commonCount
            });
        }
    }

    // Total order (userId is unique) so the result is deterministic.
    neighbours.sort(function (a, b) {
        return (b.similarity - a.similarity)
            || (b.commonCount - a.commonCount)
            || (a.userId - b.userId);
    });

    const topNeighbours = neighbours.slice(0, USER_NEIGHBOURS);

    // --- 3 & 4: accumulate over unseen movies only -----------------------
    const numerator = new Map();
    const denominator = new Map();
    const evidence = new Map();

    for (const neighbour of topNeighbours) {
        const theirRatings = userRatings.get(neighbour.userId);
        for (const [movieId, theirRating] of theirRatings) {
            // Leakage exclusion: never predict something already watched.
            if (activeRatings.has(movieId)) continue;

            numerator.set(movieId,
                (numerator.get(movieId) || 0) + neighbour.similarity * theirRating);
            denominator.set(movieId,
                (denominator.get(movieId) || 0) + neighbour.similarity);
            evidence.set(movieId,
                (evidence.get(movieId) || 0) + 1);
        }
    }

    const candidates = [];
    let candidatesConsidered = 0;
    for (const [movieId, num] of numerator) {
        candidatesConsidered++;
        const den = denominator.get(movieId);
        const evidenceCount = evidence.get(movieId);
        if (!(den > 0)) continue;
        if (evidenceCount < MIN_PREDICTORS) continue;

        const score = num / den;
        if (!Number.isFinite(score)) continue;

        candidates.push({ movieId: movieId, score: score, evidenceCount: evidenceCount });
    }

    // --- 5: deterministic ranking ----------------------------------------
    candidates.sort(function (a, b) {
        return (b.score - a.score)
            || (b.evidenceCount - a.evidenceCount)
            || (a.movieId - b.movieId);
    });

    const results = [];
    for (let i = 0; i < candidates.length && results.length < k; i++) {
        const candidate = candidates[i];
        const movie = itemById.get(candidate.movieId);
        results.push({
            movieId: candidate.movieId,
            title: movie ? movie.title : '(unknown movie ' + candidate.movieId + ')',
            score: candidate.score,
            evidenceCount: candidate.evidenceCount
        });
    }

    lastRun.user = {
        runtimeMs: performance.now() - startedAt,
        eligibleNeighbours: eligibleNeighbours,
        neighboursUsed: topNeighbours.length,
        candidatesConsidered: candidatesConsidered,
        candidatesRanked: candidates.length,
        coldStart: false
    };

    return results;
}

// ===========================================================================
// Item-Based CF
// ===========================================================================

/**
 * Top-K recommendations from similar MOVIES.
 *
 *   1. for every movie the active user has NOT rated, compare it against every
 *      movie the active user HAS rated, using support-weighted cosine over the
 *      movies' common USERS
 *   2. predict
 *          score = SUM(itemSimilarity * activeUserRating) / SUM(itemSimilarity)
 *      counting rated movies with similarity > 0 as evidenceCount
 *   3. require denominator > 0 and evidenceCount >= MIN_PREDICTORS
 *   4. sort by score DESC, evidenceCount DESC, movieId ASC and take TOP_K
 *
 * No global item-item or user-user similarity matrix is precomputed; every
 * similarity is derived from the sparse maps on this request.
 *
 * @returns {Array<{movieId, title, score, evidenceCount}>} up to TOP_K rows.
 *   Runtime and the rated x unseen pair census are published on lastRun.item.
 */
function getItemBasedRecommendations(activeUserId, topK) {
    const k = (topK === undefined) ? TOP_K : topK;
    const startedAt = performance.now();

    const activeRatings = userRatings.get(activeUserId);
    if (!activeRatings || activeRatings.size === 0) {
        // Cold start: nothing to propagate from.
        lastRun.item = {
            runtimeMs: performance.now() - startedAt,
            pairsExamined: 0,
            pairsByCommonCount: {},
            pairsUsable: 0,
            candidatesConsidered: 0,
            oneOverlapExample: null,
            coldStart: true
        };
        return [];
    }

    const scores = new Map();
    const evidence = new Map();

    // Census of the rated x unseen pairs actually examined. These are the
    // pairs that matter for a recommendation, not pairs among already-rated
    // movies, so this is the honest measure of item-side support.
    let pairsExamined = 0;
    let pairsZeroOverlap = 0;
    let pairsOneOverlap = 0;
    let pairsTwoOverlap = 0;
    let pairsUsable = 0;
    let oneOverlapRawCosineOne = 0;
    let oneOverlapExample = null;

    for (const candidateMovieId of itemIds) {
        // Leakage exclusion: skip everything the user already rated.
        if (activeRatings.has(candidateMovieId)) continue;
        pairsExamined++;

        const candidateColumn = itemRatings.get(candidateMovieId);

        let num = 0;
        let den = 0;
        let evidenceCount = 0;

        for (const [ratedMovieId, activeUserRating] of activeRatings) {
            const details = getSimilarityDetails(
                itemRatings.get(ratedMovieId), candidateColumn);

            if (details.commonCount === 0) pairsZeroOverlap++;
            else if (details.commonCount === 1) {
                pairsOneOverlap++;
                // With one shared rater cosine is exactly 1.0 whatever the two
                // ratings were. Record one concrete instance as evidence.
                if (Math.abs(details.rawCosine - 1) < 1e-12) {
                    oneOverlapRawCosineOne++;
                    if (oneOverlapExample === null) {
                        oneOverlapExample = {
                            ratedMovieId: ratedMovieId,
                            candidateMovieId: candidateMovieId,
                            rawCosine: details.rawCosine,
                            supportWeight: 1 / (1 + LAMBDA),
                            similarity: details.similarity
                        };
                    }
                }
            } else if (details.commonCount === 2) pairsTwoOverlap++;
            else pairsUsable++;

            if (!(details.similarity > 0)) continue;

            num += details.similarity * activeUserRating;
            den += details.similarity;
            evidenceCount++;
        }

        if (!(den > 0)) continue;
        if (evidenceCount < MIN_PREDICTORS) continue;

        const score = num / den;
        if (!Number.isFinite(score)) continue;

        scores.set(candidateMovieId, score);
        evidence.set(candidateMovieId, evidenceCount);
    }

    // --- deterministic ranking -------------------------------------------
    const candidates = [];
    for (const [movieId, score] of scores) {
        candidates.push({
            movieId: movieId,
            score: score,
            evidenceCount: evidence.get(movieId)
        });
    }
    candidates.sort(function (a, b) {
        return (b.score - a.score)
            || (b.evidenceCount - a.evidenceCount)
            || (a.movieId - b.movieId);
    });

    const results = [];
    for (let i = 0; i < candidates.length && results.length < k; i++) {
        const candidate = candidates[i];
        const movie = itemById.get(candidate.movieId);
        results.push({
            movieId: candidate.movieId,
            title: movie ? movie.title : '(unknown movie ' + candidate.movieId + ')',
            score: candidate.score,
            evidenceCount: candidate.evidenceCount
        });
    }

    lastRun.item = {
        runtimeMs: performance.now() - startedAt,
        pairsExamined: pairsExamined,
        pairsByCommonCount: {
            0: pairsZeroOverlap,
            1: pairsOneOverlap,
            2: pairsTwoOverlap,
            '3+': pairsUsable
        },
        oneOverlapRawCosineOne: oneOverlapRawCosineOne,
        oneOverlapExample: oneOverlapExample,
        candidatesConsidered: scores.size,
        candidatesRanked: candidates.length,
        coldStart: false
    };

    return results;
}

// ===========================================================================
// Sparse-similarity diagnostics for the selected user
// ===========================================================================

/**
 * Real overlap census between the active user and every other user.
 * Everything here is measured from userRatings; nothing is hard-coded.
 */
function computeUserOverlapDiagnostics(activeUserId) {
    const startedAt = performance.now();
    const activeRatings = userRatings.get(activeUserId);
    if (!activeRatings) return null;

    let withOverlap = 0;
    let exactlyOne = 0;
    let exactlyTwo = 0;
    let gatedByMinCommon = 0;
    let oneOverlapRawCosineOne = 0;
    let oneOverlapExample = null;
    let highestRawCosineLowSupport = null;

    for (const otherUserId of userIds) {
        if (otherUserId === activeUserId) continue;

        const details = getSimilarityDetails(activeRatings, userRatings.get(otherUserId));
        if (details.commonCount === 0) continue;

        withOverlap++;

        if (details.commonCount === 1) {
            exactlyOne++;
            if (Math.abs(details.rawCosine - 1) < 1e-12) {
                oneOverlapRawCosineOne++;
                if (oneOverlapExample === null) {
                    oneOverlapExample = {
                        otherUserId: otherUserId,
                        sharedMovieId: firstSharedKey(activeRatings, userRatings.get(otherUserId)),
                        rawCosine: details.rawCosine,
                        supportWeight: 1 / (1 + LAMBDA),
                        similarity: details.similarity
                    };
                }
            }
        } else if (details.commonCount === 2) {
            exactlyTwo++;
        }

        if (details.commonCount < MIN_COMMON) gatedByMinCommon++;

        if (details.commonCount < MIN_COMMON && details.rawCosine > 0
            && (highestRawCosineLowSupport === null
                || details.rawCosine > highestRawCosineLowSupport.rawCosine)) {
            highestRawCosineLowSupport = {
                otherUserId: otherUserId,
                commonCount: details.commonCount,
                rawCosine: details.rawCosine
            };
        }
    }

    return {
        runtimeMs: performance.now() - startedAt,
        withOverlap: withOverlap,
        exactlyOne: exactlyOne,
        exactlyTwo: exactlyTwo,
        gatedByMinCommon: gatedByMinCommon,
        oneOverlapRawCosineOne: oneOverlapRawCosineOne,
        oneOverlapExample: oneOverlapExample,
        highestRawCosineLowSupport: highestRawCosineLowSupport
    };
}

function firstSharedKey(mapA, mapB) {
    if (!mapA || !mapB) return null;
    const small = (mapA.size <= mapB.size) ? mapA : mapB;
    const large = (mapA.size <= mapB.size) ? mapB : mapA;
    for (const key of small.keys()) {
        if (large.has(key)) return key;
    }
    return null;
}

// ===========================================================================
// UI
//
// Everything below is presentation and orchestration only. It never computes a
// similarity, a prediction or a ranking: it calls the two recommendation
// functions above and formats what they return.
//
// DATA SAFETY: every value that originates in u.data / u.item (movie titles,
// ids, counts) is written with textContent. No innerHTML, no insertAdjacentHTML,
// no outerHTML, no document.write anywhere in this file.
// ===========================================================================

window.onload = async function () {
    const userPanel = document.getElementById('user-based-result');
    const itemPanel = document.getElementById('item-based-result');

    try {
        setPanelMessage(userPanel, 'Loading movie data\u2026');
        setPanelMessage(itemPanel, 'Loading movie data\u2026');
        setGlobalStatus('Loading movie data\u2026');

        await loadData();

        populateUserDropdown();
        renderDatasetChips();

        setPanelMessage(userPanel,
            'Data loaded. Select a user, then press "Generate recommendations".');
        setPanelMessage(itemPanel,
            'Data loaded. Select a user, then press "Generate recommendations".');
        setGlobalStatus('');
        renderIdleRuntimeBadges();
        renderDiagnostics(null);
        renderComparison(null);
    } catch (error) {
        // data.js has already reported the failure to BOTH result panels and
        // the diagnostics panel, and logged it. Startup is abandoned here.
        setGlobalStatus('');
        setBusy(false);
    }
};

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

/** One option per user id actually present in u.data. */
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');
    if (!selectElement) return;

    // Keep option 0 (the disabled placeholder), drop the rest.
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (const userId of userIds) {
        const option = document.createElement('option');
        option.value = String(userId);
        option.textContent = 'User ' + userId;
        selectElement.appendChild(option);
    }
}

/**
 * UI entry point, bound to the button's onclick.
 *
 * The two recommendation functions stay synchronous — that is the verified API
 * and nothing here changes it. This wrapper only does three UI things:
 *   1. clears lastRun so a failed run can never display the previous run's
 *      numbers,
 *   2. guards against double-click while a run is in flight,
 *   3. yields one paint frame so the busy state is visible before the
 *      synchronous computation blocks the main thread.
 */
function getRecommendations() {
    if (isComputing) return;

    const selectElement = document.getElementById('user-select');
    const rawValue = selectElement ? selectElement.value : '';
    const activeUserId = Number(rawValue);

    // Reachable: the button can be pressed while the placeholder is still shown.
    if (rawValue === '' || !Number.isInteger(activeUserId) || !userRatings.has(activeUserId)) {
        lastRun = { user: null, item: null };
        renderList('user-based-result', [], 'Please select a user first.');
        renderList('item-based-result', [], 'Please select a user first.');
        setCardStatus('user-based-status', '');
        setCardStatus('item-based-status', '');
        setRuntimeBadge('user-based-runtime', null);
        setRuntimeBadge('item-based-runtime', null);
        setGlobalStatus('');
        renderComparison(null);
        renderDiagnostics(null);
        return;
    }

    setBusy(true);
    afterNextPaint(function () {
        try {
            runRecommendations(activeUserId);
        } finally {
            setBusy(false);
        }
    });
}

/**
 * The whole run, synchronously. Kept separate from getRecommendations so the
 * paint/busy wrapper stays out of the measurement path.
 */
function runRecommendations(activeUserId) {
    // Clear first: if either call throws, the diagnostics must not show the
    // previous run's numbers as though they belonged to this user.
    lastRun = { user: null, item: null };

    const userResults = getUserBasedRecommendations(activeUserId, TOP_K);
    const userStats = lastRun.user;
    const itemResults = getItemBasedRecommendations(activeUserId, TOP_K);
    const itemStats = lastRun.item;
    const overlapStats = computeUserOverlapDiagnostics(activeUserId);

    const userLeakage = countLeakage(userResults, activeUserId);
    const itemLeakage = countLeakage(itemResults, activeUserId);

    const sharedTopIds = userResults
        .map(function (r) { return r.movieId; })
        .filter(function (id) {
            return itemResults.some(function (r) { return r.movieId === id; });
        });

    renderMethodPanel('user', activeUserId, userResults, userStats, userLeakage);
    renderMethodPanel('item', activeUserId, itemResults, itemStats, itemLeakage);

    setGlobalStatus('Generated recommendations for user ' + activeUserId + '.');

    const context = {
        activeUserId: activeUserId,
        userResults: userResults,
        itemResults: itemResults,
        userStats: userStats,
        itemStats: itemStats,
        overlapStats: overlapStats,
        userLeakage: userLeakage,
        itemLeakage: itemLeakage,
        sharedTopIds: sharedTopIds
    };

    renderComparison(context);
    renderDiagnostics(context);
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/**
 * Returns an explanatory message when there is genuinely nothing to show, and
 * an empty string when there ARE results (so renderList draws the list).
 * An explicit message is always better than a blank panel, so cold start and
 * thin-evidence outcomes are explained rather than looking like a failure.
 */
function emptyMessage(activeUserId, stats, approach, resultCount) {
    if (resultCount > 0) return '';

    if (stats && stats.coldStart) {
        return 'User ' + activeUserId + ' has no ratings. Collaborative filtering has '
            + 'no rating history to place this user in rating space, so neither approach '
            + 'can produce a recommendation. This is the user cold-start problem: pure CF '
            + 'needs at least a few ratings before it can say anything.';
    }

    const ranked = stats ? (stats.candidatesRanked || 0) : 0;
    if (ranked === 0) {
        return 'No candidate reached the evidence floor for user ' + activeUserId + '. '
            + 'This is what thin evidence looks like: too few neighbours or rated movies '
            + 'survived the similarity gate to support MIN_PREDICTORS = ' + MIN_PREDICTORS
            + ' contributors.';
    }

    return 'Only ' + ranked + ' candidate(s) survived the evidence floor for user '
        + activeUserId + ' (' + approach + '-based), which is fewer than the TOP_K = '
        + TOP_K + ' requested.';
}

function countLeakage(results, activeUserId) {
    const watched = userRatings.get(activeUserId);
    if (!watched) return 0;
    let leaked = 0;
    for (const result of results) {
        if (watched.has(result.movieId)) leaked++;
    }
    return leaked;
}

// ---------------------------------------------------------------------------
// Busy state
// ---------------------------------------------------------------------------

let isComputing = false;

/**
 * Shows a visible, non-colour-dependent busy state. The button is disabled so a
 * double click cannot start a second run, and aria-busy tells assistive tech
 * that the region is being replaced.
 */
function setBusy(on) {
    isComputing = on;

    for (const id of ['user-based-result', 'item-based-result']) {
        const body = document.getElementById(id);
        if (!body) continue;
        body.setAttribute('aria-busy', on ? 'true' : 'false');
        const card = body.closest ? body.closest('.result-column') : null;
        if (card && card.classList) card.classList.toggle('is-computing', on);
        if (on) body.insertBefore(makeSweep(), body.firstChild);
    }

    const button = document.getElementById('recommend-btn');
    if (button) {
        button.disabled = on;
        const label = button.querySelector ? button.querySelector('.btn-label') : null;
        if (label) {
            label.textContent = on ? 'Computing\u2026' : 'Generate recommendations';
        }
    }
}

function makeSweep() {
    const sweep = document.createElement('div');
    sweep.className = 'progress-sweep';
    return sweep;
}

/**
 * Run `callback` after the browser has had a chance to paint the busy state.
 * Falls back to a macrotask where requestAnimationFrame is unavailable, and to
 * a direct call if neither exists, so the app still works headless.
 */
function afterNextPaint(callback) {
    let done = false;
    const run = function () {
        if (done) return;
        done = true;
        callback();
    };

    const raf = (typeof requestAnimationFrame === 'function')
        ? requestAnimationFrame
        : (typeof globalThis !== 'undefined' && typeof globalThis.requestAnimationFrame === 'function'
            ? globalThis.requestAnimationFrame.bind(globalThis)
            : null);

    if (raf) {
        // Two frames: the first lets the busy state be styled, the second runs
        // after that style has actually been painted.
        raf(function () { raf(run); });
    }

    // Safety net. A backgrounded or throttled tab may never fire rAF, which
    // would leave the button permanently disabled. Whichever fires first wins.
    if (typeof setTimeout === 'function') {
        setTimeout(run, raf ? 120 : 0);
    } else {
        run();
    }
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function clearElement(element) {
    if (!element) return;
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

function makeElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
}

function setPanelMessage(element, message) {
    if (!element) return;
    clearElement(element);
    element.appendChild(makeElement('p', 'panel-message', message));
}

function setCardStatus(elementId, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    element.textContent = message || '';
}

function setGlobalStatus(message) {
    const element = document.getElementById('global-status');
    if (element) element.textContent = message || '';
}

function setRuntimeBadge(elementId, runtimeMs) {
    const element = document.getElementById(elementId);
    if (!element) return;
    if (runtimeMs === null || runtimeMs === undefined) {
        element.textContent = 'not run';
        if (element.classList) element.classList.add('is-idle');
        return;
    }
    element.textContent = runtimeMs.toFixed(1) + ' ms';
    if (element.classList) element.classList.remove('is-idle');
}

function renderIdleRuntimeBadges() {
    setRuntimeBadge('user-based-runtime', null);
    setRuntimeBadge('item-based-runtime', null);
}

function evidenceLabel(evidenceCount) {
    if (evidenceCount === undefined || evidenceCount === null) return 'no evidence';
    return evidenceCount === 1 ? '1 contributor' : evidenceCount + ' contributors';
}

// ---------------------------------------------------------------------------
// Method panel (one per approach)
// ---------------------------------------------------------------------------

function renderMethodPanel(side, activeUserId, results, stats, leakage) {
    const panelId = side === 'user' ? 'user-based-result' : 'item-based-result';
    const statusId = side === 'user' ? 'user-based-status' : 'item-based-status';
    const badgeId = side === 'user' ? 'user-based-runtime' : 'item-based-runtime';

    setRuntimeBadge(badgeId, stats ? stats.runtimeMs : null);
    renderList(panelId, results,
        emptyMessage(activeUserId, stats, side, results.length));

    if (results.length > 0) {
        setCardStatus(statusId,
            results.length + (results.length === 1 ? ' recommendation' : ' recommendations')
            + ' \u00b7 leakage ' + (leakage === 0 ? 'PASS' : 'FAIL'));
    } else {
        setCardStatus(statusId, '');
    }
}

// ---------------------------------------------------------------------------
// Recommendation list
// ---------------------------------------------------------------------------

/**
 * Render one Top-K list. Each row shows rank, title, id badge, predicted score
 * and evidence count. Titles are written with textContent, so a title
 * containing & or ' is displayed literally and can never be parsed as markup.
 */
function renderList(elementId, items, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    clearElement(element);

    if (message) {
        element.appendChild(makeElement('p', 'panel-message', message));
        return;
    }

    if (!items || items.length === 0) {
        element.appendChild(makeElement('p', 'panel-message',
            'No recommendations could be produced.'));
        return;
    }

    const list = makeElement('ol', 'rec-list');

    items.forEach(function (item, index) {
        const row = makeElement('li', 'rec-row');
        row.appendChild(makeElement('span', 'rec-rank', '#' + (index + 1)));

        const body = makeElement('span', 'rec-body');
        body.appendChild(makeElement('span', 'rec-title', item.title));

        const meta = makeElement('span', 'rec-meta');
        meta.appendChild(makeElement('span', 'tag-id', 'ID ' + item.movieId));
        meta.appendChild(makeElement('span', 'tag-evidence',
            evidenceLabel(item.evidenceCount)));
        body.appendChild(meta);

        row.appendChild(body);

        const score = makeElement('span', 'score');
        score.appendChild(makeElement('span', 'score-value',
            Number(item.score).toFixed(3)));
        score.appendChild(makeElement('span', 'score-caption', 'predicted'));
        row.appendChild(score);

        list.appendChild(row);
    });

    element.appendChild(list);
}

// ---------------------------------------------------------------------------
// Header chips — values come from the loaded dataset, never hard-coded
// ---------------------------------------------------------------------------

function renderDatasetChips() {
    const list = document.getElementById('dataset-chips');
    if (!list) return;
    clearElement(list);

    const userCount = userIds.length;
    const itemCount = numMovies;
    const density = (ratings.length / (userCount * itemCount)) * 100;

    addChip(list, compactNumber(ratings.length), 'ratings');
    addChip(list, userCount.toLocaleString('en-US'), 'users');
    addChip(list, itemCount.toLocaleString('en-US'), 'movies');
    addChip(list, density.toFixed(2) + '%', 'dense');
    addChip(list, (100 - density).toFixed(2) + '%', 'sparse');
}

function addChip(list, value, label) {
    const item = makeElement('li', 'chip');
    item.appendChild(makeElement('b', null, value));
    item.appendChild(makeElement('span', null, label));
    list.appendChild(item);
}

function compactNumber(value) {
    if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
    if (value >= 1000) return (value / 1000).toFixed(0) + 'K';
    return String(value);
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

function renderComparison(context) {
    const grid = document.getElementById('comparison-grid');
    if (!grid) return;
    clearElement(grid);

    if (context === null) {
        grid.appendChild(makeElement('p', 'placeholder',
            'Select a user and press "Generate recommendations" to compare both approaches.'));
        return;
    }

    const userStats = context.userStats || {};
    const itemStats = context.itemStats || {};
    const activeRatingCount = (userRatings.get(context.activeUserId) || new Map()).size;

    appendMetric(grid, 'Active user', 'User ' + context.activeUserId,
        activeRatingCount + ' ratings \u00b7 ' + (numMovies - activeRatingCount)
        + ' unseen of ' + numMovies);

    appendMetric(grid, 'Top-5 overlap',
        context.sharedTopIds.length === 0 ? '0 shared' : context.sharedTopIds.length + ' shared',
        context.sharedTopIds.length === 0
            ? 'the two neighbourhoods are disjoint'
            : context.sharedTopIds.map(idMovieTitle).join(', '));

    appendMetric(grid, 'User-Based runtime', formatMs(userStats.runtimeMs),
        (userStats.neighboursUsed || 0) + ' of ' + (userStats.eligibleNeighbours || 0)
        + ' eligible neighbours \u00b7 ' + (userStats.candidatesRanked || 0) + ' candidates ranked');

    appendMetric(grid, 'Item-Based runtime', formatMs(itemStats.runtimeMs),
        (itemStats.candidatesRanked || 0) + ' candidates ranked \u00b7 '
        + ((itemStats.pairsExamined || 0) * activeRatingCount)
            .toLocaleString('en-US') + ' rated \u00d7 unseen pairs');

    appendMetric(grid, 'Mean evidence \u00b7 user', meanEvidence(context.userResults),
        'contributors per recommendation');
    appendMetric(grid, 'Mean evidence \u00b7 item', meanEvidence(context.itemResults),
        'contributors per recommendation');

    appendStatusMetric(grid, 'Leakage \u00b7 User-Based', context.userLeakage === 0,
        context.userLeakage + ' already-rated in Top-5');
    appendStatusMetric(grid, 'Leakage \u00b7 Item-Based', context.itemLeakage === 0,
        context.itemLeakage + ' already-rated in Top-5');
}

function appendMetric(parent, label, value, sub) {
    const metric = makeElement('div', 'metric');
    metric.appendChild(makeElement('div', 'metric-label', label));
    metric.appendChild(makeElement('div', 'metric-value', value));
    if (sub) metric.appendChild(makeElement('div', 'metric-sub', sub));
    parent.appendChild(metric);
}

/** PASS/FAIL is always spelled out, so colour is never the only signal. */
function appendStatusMetric(parent, label, ok, sub) {
    const metric = makeElement('div', 'metric');
    metric.appendChild(makeElement('div', 'metric-label', label));

    const value = makeElement('div', 'metric-value');
    const badge = makeElement('span', ok ? 'status-pass' : 'status-fail');
    badge.appendChild(makeElement('span', 'status-mark', ok ? '\u2713' : '\u2717'));
    badge.appendChild(makeElement('span', null, ok ? 'PASS' : 'FAIL'));
    value.appendChild(badge);
    metric.appendChild(value);

    if (sub) metric.appendChild(makeElement('div', 'metric-sub', sub));
    parent.appendChild(metric);
}

// ---------------------------------------------------------------------------
// Technical diagnostics
// ---------------------------------------------------------------------------

function renderDiagnostics(context) {
    const body = document.getElementById('diagnostics-body');
    if (!body) return;
    clearElement(body);

    if (ratingMatrix === null) {
        body.appendChild(makeElement('p', 'placeholder',
            'Diagnostics appear here once the dataset has loaded.'));
        return;
    }

    const groups = makeElement('div', 'diag-groups');
    const userCount = userIds.length;
    const itemCount = numMovies;
    const logicalCells = userCount * itemCount;
    const densityPct = (ratings.length / logicalCells) * 100;
    const overlap = context ? context.overlapStats : null;
    const activeRatings = context ? userRatings.get(context.activeUserId) : null;
    const ratingCount = activeRatings ? activeRatings.size : 0;
    const userStats = context ? (context.userStats || {}) : {};
    const itemStats = context ? (context.itemStats || {}) : {};

    // --- Dataset ---------------------------------------------------------
    appendDiagGroup(groups, 'Dataset', [
        ['Ratings parsed', ratings.length.toLocaleString('en-US')],
        ['Users', userCount.toLocaleString('en-US')],
        ['Items', itemCount.toLocaleString('en-US')],
        ['Logical matrix U \u00d7 I', userCount + ' \u00d7 ' + itemCount
            + ' = ' + logicalCells.toLocaleString('en-US')],
        ['Density', densityPct.toFixed(6) + ' %'],
        ['Sparsity', (100 - densityPct).toFixed(6) + ' %'],
        ['Allocated matrix', (numUsers + 1) + ' \u00d7 ' + matrixColumns
            + ' = ' + ((numUsers + 1) * matrixColumns).toLocaleString('en-US')],
        ['u.item encoding', uItemEncoding]
    ], 'Row/column 0 of the allocated matrix are unused, so its storage density '
        + '(' + (((ratings.length / ((numUsers + 1) * matrixColumns)) * 100)).toFixed(6)
        + ' %) is lower than the dataset density above. '
        + (uItemEncoding === 'iso-8859-1'
            ? 'u.item is not valid UTF-8, so it is decoded explicitly as '
                + 'ISO-8859-1 to preserve the 9 non-ASCII titles.'
            : ''));

    // --- Selected user ---------------------------------------------------
    if (context === null) {
        appendDiagGroup(groups, 'Selected user', [
            ['Status', 'no user selected']
        ], 'Choose a user above and press "Generate recommendations".');
    } else {
        appendDiagGroup(groups, 'Selected user', [
            ['User ID', context.activeUserId],
            ['Ratings by this user', ratingCount],
            ['Movies never rated', (numMovies - ratingCount) + ' of ' + numMovies],
            ['Other users with \u22651 common rating', overlap ? overlap.withOverlap : '\u2014'],
            ['\u2026 with exactly 1 common rating', overlap ? overlap.exactlyOne : '\u2014'],
            ['\u2026 with exactly 2 common ratings', overlap ? overlap.exactlyTwo : '\u2014'],
            ['\u2026 gated out by MIN_COMMON = ' + MIN_COMMON,
                overlap ? overlap.gatedByMinCommon : '\u2014']
        ], 'Overlap counts are measured from the sparse user index, not assumed.');
    }

    // --- Similarity strategy ---------------------------------------------
    appendDiagGroup(groups, 'Similarity strategy', [
        ['Strategy', 'support-weighted co-rated cosine'],
        ['Lambda (\u03bb)', LAMBDA],
        ['Minimum common ratings', MIN_COMMON],
        ['Minimum predictors', MIN_PREDICTORS],
        ['Neighbour cut', USER_NEIGHBOURS],
        ['Top-K', TOP_K]
    ], 'rawCosine = dot(A,B) / (\u2016A\u2016 \u00b7 \u2016B\u2016) over co-rated entries only; '
        + 'supportWeight = nCommon / (nCommon + lambda); '
        + 'finalSimilarity = rawCosine \u00d7 supportWeight, and 0 when nCommon < MIN_COMMON. '
        + 'Missing ratings stay missing: no mean imputation, no mean-centring, no matrix factorization.');

    // --- Sparsity evidence ------------------------------------------------
    if (context !== null) {
        const census = itemStats.pairsByCommonCount || {};
        const total = census[0] + census[1] + census[2] + census['3+'];
        const usableShare = total > 0 ? (census['3+'] / total) * 100 : 0;
        appendDiagGroup(groups, 'Sparsity evidence', [
            ['Rated \u00d7 unseen pairs scored', total.toLocaleString('en-US')],
            ['\u2026 0 common raters', (census[0] || 0).toLocaleString('en-US')],
            ['\u2026 1 common rater', (census[1] || 0).toLocaleString('en-US')],
            ['\u2026 2 common raters', (census[2] || 0).toLocaleString('en-US')],
            ['\u2026 3+ common raters (usable)', (census['3+'] || 0).toLocaleString('en-US')],
            ['Usable share', usableShare.toFixed(4) + ' %'],
            ['nCommon=1 pairs at raw cosine 1.0',
                (itemStats.oneOverlapRawCosineOne || 0) + ' of ' + (census[1] || 0)]
        ], overlap && overlap.oneOverlapExample
            ? 'One-overlap example: ' + overlap.oneOverlapExample.otherUserId
                + ' vs ' + context.activeUserId + ' share only movie '
                + overlap.oneOverlapExample.sharedMovieId
                + ' \u2192 rawCosine 1.000000, final similarity 0.000000.'
            : 'A single shared rating always yields raw cosine 1.0, so the gate and the '
                + 'support weight are what make those pairs harmless.');
    }

    // --- Validation --------------------------------------------------------
    if (context !== null) {
        appendDiagGroup(groups, 'Validation', [
            ['Leakage \u00b7 user', context.userLeakage === 0
                ? 'PASS (0 leaked)' : 'FAIL (' + context.userLeakage + ' leaked)'],
            ['Leakage \u00b7 item', context.itemLeakage === 0
                ? 'PASS (0 leaked)' : 'FAIL (' + context.itemLeakage + ' leaked)'],
            ['Top-5 overlap', context.sharedTopIds.length + ' shared movie(s)'],
            ['Candidates ranked \u00b7 user', userStats.candidatesRanked || 0],
            ['Candidates ranked \u00b7 item', itemStats.candidatesRanked || 0]
        ], 'Ranking is a total order: score DESC, then evidenceCount DESC, then movieId ASC, '
            + 'so repeated runs are byte-identical. Runtimes are recorded in this panel but are '
            + 'never used for ranking.');
    }

    body.appendChild(groups);
}

function appendDiagGroup(parent, title, rows, note) {
    const group = makeElement('section', 'diag-group');
    group.appendChild(makeElement('h3', 'diag-group-title', title));

    const table = makeElement('dl', 'diag-table');
    for (const row of rows) {
        table.appendChild(makeElement('dt', 'diag-label', row[0]));
        table.appendChild(makeElement('dd', 'diag-value', row[1]));
    }
    if (note) {
        table.appendChild(makeElement('p', 'diag-note', note));
    }
    group.appendChild(table);
    parent.appendChild(group);
}

// ---------------------------------------------------------------------------
// Small formatters
// ---------------------------------------------------------------------------

function idMovieTitle(movieId) {
    if (movieId === null || movieId === undefined) return 'unknown movie';
    const movie = itemById.get(movieId);
    return movie ? ('movie ' + movieId + ' "' + movie.title + '"') : ('movie ' + movieId);
}

function meanEvidence(results) {
    if (!results || results.length === 0) return 'n/a';
    let total = 0;
    for (const result of results) total += result.evidenceCount;
    return (total / results.length).toFixed(2);
}

function formatMs(value) {
    if (value === undefined || value === null) return 'n/a';
    return value.toFixed(1) + ' ms';
}
