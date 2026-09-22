// ---------------------------------------------------------------------------
// HW A02 - Content-Based Movie Recommender (vanilla JS, no dependencies)
//
// Ranking uses cosine similarity ONLY:
//     cos(a, b) = dot(a, b) / (||a|| * ||b||)
//
// Popularity (rating count from u.data) is computed for evaluation/display
// only and never enters the ranking score.
// ---------------------------------------------------------------------------

// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        // Load data
        await loadData();

        // Populate dropdowns and update status
        populateMoviesDropdowns();
        document.getElementById('results').hidden = true;
        resultElement.textContent =
            `Data loaded: ${movies.length} movies, ${ratings.length} ratings. ` +
            `Select 3 distinct movies, then press Get Recommendations.`;
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the three watched-movie dropdowns with the same sorted movie list.
function populateMoviesDropdowns() {
    const selectIds = ['watched-1', 'watched-2', 'watched-3'];

    // Duplicate titles exist in MovieLens (same title, different ids), so append
    // the id to duplicated titles. Identity is always the numeric id, never text.
    const titleCounts = new Map();
    for (const movie of movies) {
        titleCounts.set(movie.title, (titleCounts.get(movie.title) || 0) + 1);
    }

    // Sort alphabetically by title with id as a deterministic secondary key.
    const sortedMovies = [...movies].sort(
        (a, b) => a.title.localeCompare(b.title) || a.id - b.id
    );

    for (const selectId of selectIds) {
        const selectElement = document.getElementById(selectId);
        selectElement.innerHTML = '';

        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.disabled = true;
        placeholder.selected = true;
        placeholder.textContent = 'Select a movie';
        selectElement.appendChild(placeholder);

        for (const movie of sortedMovies) {
            const option = document.createElement('option');
            option.value = String(movie.id);
            option.textContent = titleCounts.get(movie.title) > 1
                ? `${movie.title} [id: ${movie.id}]`
                : movie.title;
            selectElement.appendChild(option);
        }
    }
}

// ---------------------------------------------------------------------------
// Vector math
// ---------------------------------------------------------------------------

function dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

function vectorNorm(v) {
    return Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
}

// cos(a, b) = dot(a, b) / (||a|| * ||b||)
//
// Both vector norms are used in the denominator. If either vector has zero
// norm the cosine is mathematically undefined; we define it as 0 (no
// content-overlap evidence) so the score can never become NaN or Infinity.
function cosineSimilarity(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
        return 0;
    }
    const normA = vectorNorm(a);
    const normB = vectorNorm(b);
    if (normA === 0 || normB === 0) {
        return 0;
    }
    return dotProduct(a, b) / (normA * normB);
}

// Aggregate user profile from exactly 3 watched movies:
//     profile = (v1 + v2 + v3) / 3   (element-wise mean)
function buildProfile(v1, v2, v3) {
    const profile = [];
    for (let i = 0; i < v1.length; i++) {
        profile.push((v1[i] + v2[i] + v3[i]) / 3);
    }
    return profile;
}

// Score every candidate with cosine similarity and return Top-5.
// Deterministic ordering:
//   1. cosine score descending
//   2. movie id ascending for equal scores
function rankCandidates(queryVector, candidates) {
    return candidates
        .map(movie => ({ movie, score: cosineSimilarity(queryVector, movie.vector) }))
        .sort((a, b) => (b.score - a.score) || (a.movie.id - b.movie.id))
        .slice(0, 5);
}

// ---------------------------------------------------------------------------
// Popularity (evaluation metadata ONLY - never part of any score)
// ---------------------------------------------------------------------------

// Derive ratingCount per movie from the parsed u.data ratings.
function buildRatingCounts() {
    const counts = new Map();
    for (const r of ratings) {
        counts.set(r.itemId, (counts.get(r.itemId) || 0) + 1);
    }
    return counts;
}

function ratingCountFor(counts, movieId) {
    // Movies with zero ratings (none in this dataset, but handled safely) -> 0.
    return counts.get(movieId) || 0;
}

function median(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Long-tail threshold = median rating count across the whole catalog,
// computed from the loaded data (never hard-coded).
// A movie is long-tail iff ratingCount <= threshold.
function longTailThreshold(counts) {
    return median(movies.map(m => ratingCountFor(counts, m.id)));
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

// Read the three watched selections. Exactly 3 movies, all DISTINCT.
function readWatchedSelection() {
    const values = ['watched-1', 'watched-2', 'watched-3']
        .map(id => document.getElementById(id).value);

    if (values.some(v => v === '' || v === null || v === undefined)) {
        return {
            error: 'Please select all three movies (Watched Movie 1 / Active Movie, Movie 2, Movie 3).'
        };
    }

    const ids = values.map(v => parseInt(v, 10));
    if (ids.some(id => !Number.isInteger(id))) {
        return { error: 'Invalid movie selection.' };
    }

    const selected = ids.map(id => movies.find(m => m.id === id));
    if (selected.some(m => !m)) {
        return { error: 'Error: Selected movie not found in database.' };
    }

    if (new Set(ids).size !== 3) {
        return {
            error: 'The three watched movies must be DISTINCT - no movie may be selected twice.'
        };
    }

    return { watched: selected };
}

// ---------------------------------------------------------------------------
// Core computation (no DOM access - verifiable headlessly)
// ---------------------------------------------------------------------------

// Movies leaked into a Top-5 list that the user already watched (must be empty).
function findLeakage(entries, watchedIds) {
    return entries.filter(e => watchedIds.has(e.movie.id)).map(e => e.movie);
}

// Movies present in both Top-5 lists.
function overlapTitles(listA, listB) {
    const idsB = new Set(listB.map(e => e.movie.id));
    return listA.filter(e => idsB.has(e.movie.id)).map(e => e.movie);
}

// Evaluation metrics for one Top-5 list.
function listMetrics(entries, counts, threshold) {
    const total = entries.length || 1;
    const popularity = entries.map(e => ratingCountFor(counts, e.movie.id));
    const genreSet = new Set();
    let genreSum = 0;
    for (const entry of entries) {
        genreSum += entry.movie.genres.length;
        entry.movie.genres.forEach(g => genreSet.add(g));
    }
    const tailCount = popularity.filter(c => c <= threshold).length;
    return {
        avgPopularity: popularity.reduce((s, c) => s + c, 0) / total,
        tailCount,
        tailShare: tailCount / total,
        distinctGenres: genreSet.size,
        meanGenresPerMovie: genreSum / total
    };
}

// Compute both Top-5 lists and every verification/analysis value for them.
function computeRecommendations(watched) {
    const [movie1, movie2, movie3] = watched;
    const watchedIds = new Set(watched.map(m => m.id));

    // Exclude ALL 3 watched movies from BOTH candidate sets so already-watched
    // items can never leak into either recommendation list.
    const candidates = movies.filter(m => !watchedIds.has(m.id));

    // A. Item-to-item: query vector = Watched Movie 1 (the active movie).
    const i2iTop5 = rankCandidates(movie1.vector, candidates);

    // B. Profile-based: profile = (v1 + v2 + v3) / 3.
    const profile = buildProfile(movie1.vector, movie2.vector, movie3.vector);
    const profileTop5 = rankCandidates(profile, candidates);

    // Popularity data (evaluation metadata only).
    const counts = buildRatingCounts();
    const threshold = longTailThreshold(counts);

    return {
        watched,
        watchedIds,
        candidateCount: candidates.length,
        i2iTop5,
        profileTop5,
        profile,
        counts,
        threshold,
        // Profile-collapse diagnostic: cosine between the average and each input.
        profileCosines: watched.map(movie => ({ movie, score: cosineSimilarity(profile, movie.vector) })),
        // Already-watched leakage checks (both must be empty).
        i2iLeakage: findLeakage(i2iTop5, watchedIds),
        profileLeakage: findLeakage(profileTop5, watchedIds)
    };
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

// Main recommendation function (validates input, then renders both Top-5 lists).
function getRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        const input = readWatchedSelection();
        if (input.error) {
            document.getElementById('results').hidden = true;
            resultElement.textContent = input.error;
            resultElement.className = 'error';
            return;
        }

        // Show loading message while processing
        resultElement.textContent = 'Calculating recommendations...';
        resultElement.className = 'loading';

        // Use setTimeout to allow the UI to update before the computation
        setTimeout(() => {
            try {
                const rec = computeRecommendations(input.watched);
                renderResults(rec);
                const titles = input.watched.map(m => `"${m.title}"`).join(', ');
                resultElement.textContent = `Top-5 lists computed for watched movies: ${titles}.`;
                resultElement.className = 'success';
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = 'An error occurred while calculating recommendations.';
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = 'An unexpected error occurred.';
        resultElement.className = 'error';
    }
}

function appendTextCell(tr, text, className) {
    const td = document.createElement('td');
    td.textContent = text;
    if (className) td.className = className;
    tr.appendChild(td);
    return td;
}

function renderTop5Table(tbodyId, entries, rec) {
    const tbody = document.getElementById(tbodyId);
    tbody.innerHTML = '';

    entries.forEach((entry, index) => {
        const tr = document.createElement('tr');
        appendTextCell(tr, String(index + 1), 'num');
        appendTextCell(tr, entry.movie.title);
        appendTextCell(tr, String(entry.movie.id), 'num');
        appendTextCell(tr, entry.score.toFixed(6), 'num');
        appendTextCell(tr, entry.movie.genres.join(', ') || '(no genres)');

        const count = ratingCountFor(rec.counts, entry.movie.id);
        appendTextCell(tr, String(count), 'num');

        const isTail = count <= rec.threshold;
        appendTextCell(tr, isTail ? 'yes' : 'no', isTail ? 'tail-yes' : 'tail-no');

        tbody.appendChild(tr);
    });
}

function addMetricRow(container, label, value) {
    const row = document.createElement('div');
    row.className = 'metric-row';

    const labelEl = document.createElement('span');
    labelEl.className = 'label';
    labelEl.textContent = label;

    const valueEl = document.createElement('span');
    valueEl.className = 'value';
    valueEl.textContent = value;

    row.appendChild(labelEl);
    row.appendChild(valueEl);
    container.appendChild(row);
}

function genresOfList(entries) {
    const genreSet = new Set();
    for (const entry of entries) {
        entry.movie.genres.forEach(g => genreSet.add(g));
    }
    return genreSet;
}

function renderAnalysis(rec) {
    const container = document.getElementById('analysis');
    container.innerHTML = '';

    const i2iM = listMetrics(rec.i2iTop5, rec.counts, rec.threshold);
    const profM = listMetrics(rec.profileTop5, rec.counts, rec.threshold);
    const overlap = overlapTitles(rec.i2iTop5, rec.profileTop5);
    const unionGenres = new Set([...genresOfList(rec.i2iTop5), ...genresOfList(rec.profileTop5)]);

    addMetricRow(
        container,
        `Long-tail threshold (median rating count, computed from ${movies.length} movies)`,
        String(rec.threshold)
    );

    addMetricRow(container, 'Item-to-Item - average popularity@5', i2iM.avgPopularity.toFixed(1));
    addMetricRow(container, 'Item-to-Item - long-tail share@5',
        `${i2iM.tailCount}/5 (${Math.round(i2iM.tailShare * 100)}%)`);
    addMetricRow(container, 'Item-to-Item - genre coverage',
        `${i2iM.distinctGenres}/18 distinct genres (mean ${i2iM.meanGenresPerMovie.toFixed(1)} genres/title)`);

    addMetricRow(container, 'Profile-Based - average popularity@5', profM.avgPopularity.toFixed(1));
    addMetricRow(container, 'Profile-Based - long-tail share@5',
        `${profM.tailCount}/5 (${Math.round(profM.tailShare * 100)}%)`);
    addMetricRow(container, 'Profile-Based - genre coverage',
        `${profM.distinctGenres}/18 distinct genres (mean ${profM.meanGenresPerMovie.toFixed(1)} genres/title)`);

    addMetricRow(container, 'Top-5 overlap between the two approaches', `${overlap.length} movie(s)`);
    addMetricRow(container, 'Overlapping titles',
        overlap.length ? overlap.map(m => m.title).join('; ') : 'none');
    addMetricRow(container, 'Genre coverage of the union of both lists', `${unionGenres.size}/18`);

    const note = document.createElement('p');
    note.className = 'metric-note';
    note.textContent = 'Measured values for this selection only - no approach is claimed to be universally better.';
    container.appendChild(note);
}

function formatLeakageList(movieList) {
    if (movieList.length === 0) return 'empty';
    return movieList.map(m => `${m.title} [id=${m.id}]`).join('; ');
}

function renderChecks(rec) {
    const container = document.getElementById('checks');
    container.innerHTML = '';

    // A. Already-watched leakage: PASS only when BOTH intersections are empty.
    const bothEmpty = rec.i2iLeakage.length === 0 && rec.profileLeakage.length === 0;
    const leakDiv = document.createElement('div');
    leakDiv.className = 'check ' + (bothEmpty ? 'pass' : 'fail');
    leakDiv.textContent =
        `WATCHED-LEAKAGE CHECK: ${bothEmpty ? 'PASS' : 'FAIL'} - ` +
        `intersection(itemTop5 IDs, watched IDs) = ${formatLeakageList(rec.i2iLeakage)}; ` +
        `intersection(profileTop5 IDs, watched IDs) = ${formatLeakageList(rec.profileLeakage)}`;
    container.appendChild(leakDiv);

    // C. Profile-averaging diagnostic: report the three values only.
    const diagHead = document.createElement('div');
    diagHead.className = 'check neutral';
    diagHead.textContent =
        'Profile-averaging diagnostic: cosine between the averaged profile and each watched movie ' +
        '(values are reported for interpretation - no universal pass/fail threshold is applied):';
    container.appendChild(diagHead);

    for (const { movie, score } of rec.profileCosines) {
        addMetricRow(
            container,
            `cos(profile, ${movie.title} [id=${movie.id}])`,
            score.toFixed(6)
        );
    }
}

function renderResults(rec) {
    renderTop5Table('i2i-tbody', rec.i2iTop5, rec);
    renderTop5Table('profile-tbody', rec.profileTop5, rec);
    renderAnalysis(rec);
    renderChecks(rec);
    document.getElementById('results').hidden = false;
}
