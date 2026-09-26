'use strict';

// ===========================================================================
// data.js — MovieLens 100K: loading, strict parsing, and matrix construction
//
// Responsibilities (kept strictly separate from script.js, per the assignment):
//   * fetch u.item and u.data
//   * decode them with the correct character encoding
//   * validate every field
//   * build the required dense raw-ID rating matrix
//   * build the sparse index structures the CF code iterates over
// ===========================================================================

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
let movies = [];            // [{ id, title, genres, unknownGenre }] in file order
let ratings = [];           // [{ userId, itemId, rating, timestamp }]
let numUsers = 0;           // highest raw user id in u.data (matrix row bound)
let numMovies = 0;          // number of parsed movies in u.item
let ratingMatrix = null;    // ratingMatrix[userId][movieId] === rating, 0 = missing

// Sparse indexes. Similarity work iterates these, never the dense zeros.
let userRatings = new Map();   // userId  -> Map(movieId, rating)
let itemRatings = new Map();   // movieId -> Map(userId,  rating)
let itemById = new Map();      // movieId -> { id, title, genres }
let userIds = [];              // actual sorted user ids present in u.data
let itemIds = [];              // actual sorted movie ids that have >= 1 rating

// Diagnostics only.
let uItemEncoding = 'unknown'; // encoding actually used to decode u.item
let matrixColumns = 0;         // allocated column count (max raw movie id + 1)

// ---------------------------------------------------------------------------
// u.item column layout — 24 '|' separated fields
//
//   [0] movie id          [1] title            [2] release date
//   [3] video release date[4] IMDb URL         [5] "unknown" genre flag
//   [6 .. 23] the 18 named genre flags, Action .. Western
//
// The "unknown" flag at index 5 is NOT a genre. Pairing fields[5..23] with the
// 18 genre names shifts every name one column left: "Action" picks up the
// unknown flag, "Western" reads the Thriller column, and the real Western
// column is never read. See GENRE_FLAG_START below.
// ---------------------------------------------------------------------------
const U_ITEM_FIELD_COUNT = 24;   // 5 metadata + 1 unknown flag + 18 genre flags
const UNKNOWN_FLAG_INDEX = 5;     // "unknown" genre flag: validated, never a genre
const GENRE_FLAG_START = 6;       // first named genre flag ("Action")
const GENRE_COUNT = 18;           // exactly 18 named genres

const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// u.data column layout — exactly 4 tab separated fields.
const RATING_FIELD_COUNT = 4;     // userId, itemId, rating, timestamp
const MIN_RATING = 1;
const MAX_RATING = 5;

// ---------------------------------------------------------------------------
// Encoding
//
// Verified for this dataset:
//   u.item  is NOT valid UTF-8. It is ISO-8859-1 (Latin-1); the only high bytes
//           present are 0xC1 0xE8 0xE9 0xF6, all in the Latin-1 letter range.
//   u.data  is pure ASCII, so it is valid UTF-8.
//
// Decoding u.item with the default UTF-8 TextDecoder silently corrupts the 9
// affected titles: the invalid bytes are replaced by U+FFFD REPLACEMENT
// CHARACTER, so "Miserables, Les (1995)" becomes "Mis<U+FFFD>rables, Les (1995)".
// We therefore sniff with a fatal UTF-8 decode and fall back explicitly.
//
// Per the WHATWG Encoding Standard the label "iso-8859-1" maps to windows-1252.
// That is byte-identical to true ISO-8859-1 for every byte value actually
// present in this file, because none of them fall in the 0x80-0x9F range where
// the two encodings differ.
// ---------------------------------------------------------------------------
function decodeText(buffer) {
    const bytes = new Uint8Array(buffer);

    try {
        // fatal: true makes an invalid sequence throw instead of inserting U+FFFD.
        const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
        return { text: text, encoding: 'utf-8' };
    } catch (invalidUtf8) {
        const text = new TextDecoder('iso-8859-1').decode(bytes);
        return { text: text, encoding: 'iso-8859-1' };
    }
}

/**
 * Clear every data structure so a rebuild starts from nothing.
 *
 * Called at the top of loadData() AND at the top of buildRatingMatrix(). That
 * second call is what makes buildRatingMatrix() idempotent on its own: without
 * it, invoking buildRatingMatrix() twice would find every row already present
 * in userRatings and report the whole file as duplicates.
 */
function resetDataStructures() {
    movies = [];
    ratings = [];
    numUsers = 0;
    numMovies = 0;
    ratingMatrix = null;
    userRatings = new Map();
    itemRatings = new Map();
    itemById = new Map();
    userIds = [];
    itemIds = [];
    uItemEncoding = 'unknown';
    matrixColumns = 0;
}

// ---------------------------------------------------------------------------
// loadData — fetch, decode, parse, build. Idempotent.
// ---------------------------------------------------------------------------
async function loadData() {
    // Reset first so repeated calls cannot accumulate data. Without this, a
    // second loadData() would leave movies.length = 3364, ratings.length =
    // 200000 and a matrix with duplicated rows and columns.
    resetDataStructures();

    try {
        // --- u.item first, then u.data (assignment section 4.2) ---
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load u.item: HTTP ${moviesResponse.status}`);
        }
        const moviesDecoded = decodeText(await moviesResponse.arrayBuffer());
        uItemEncoding = moviesDecoded.encoding;
        parseItemData(moviesDecoded.text);

        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load u.data: HTTP ${ratingsResponse.status}`);
        }
        parseRatingData(decodeText(await ratingsResponse.arrayBuffer()).text);

        // buildRatingMatrix() derives every dimension (numUsers, numMovies,
        // matrixColumns) and builds the dense matrix plus the sparse indexes.
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        reportFatalError(error);
        throw error; // re-throw so the caller can stop its own startup sequence
    }
}

// ---------------------------------------------------------------------------
// Error reporting — both result panels must show the failure.
// ---------------------------------------------------------------------------
function reportFatalError(error) {
    const message = 'Error: ' + error.message +
        ' Make sure u.item and u.data sit in the same directory as index.html.';

    // Both result panels must show the failure, never just one.
    writeErrorMessage('user-based-result', message);
    writeErrorMessage('item-based-result', message);

    // The diagnostics panel would otherwise stay stuck on "Loading..." forever.
    writeErrorMessage('diagnostics',
        'Diagnostics unavailable: the dataset failed to load.');
}

function writeErrorMessage(elementId, message) {
    const target = document.getElementById(elementId);
    if (!target) return;
    target.textContent = '';
    const paragraph = document.createElement('p');
    paragraph.className = 'error';
    paragraph.textContent = message;
    target.appendChild(paragraph);
}

// ---------------------------------------------------------------------------
// parseItemData — strict u.item parser
// ---------------------------------------------------------------------------
function parseItemData(text) {
    const lines = text.split('\n');
    let lineNumber = 0;

    for (const line of lines) {
        lineNumber++;
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length !== U_ITEM_FIELD_COUNT) {
            throw new Error(
                `u.item line ${lineNumber}: expected ${U_ITEM_FIELD_COUNT} ` +
                `'|' separated fields, found ${fields.length}.`);
        }

        const id = Number(fields[0]);
        if (!Number.isInteger(id) || id <= 0) {
            throw new Error(
                `u.item line ${lineNumber}: movie id must be a positive integer, ` +
                `found "${fields[0]}".`);
        }

        const title = fields[1].trim();
        if (title === '') {
            throw new Error(`u.item line ${lineNumber}: movie ${id} has an empty title.`);
        }

        // fields[5] is the "unknown" flag. Validate it, but never treat it as a genre.
        const unknownFlag = fields[UNKNOWN_FLAG_INDEX];
        if (unknownFlag !== '0' && unknownFlag !== '1') {
            throw new Error(
                `u.item line ${lineNumber}: movie ${id} unknown-genre flag must be ` +
                `0 or 1, found "${unknownFlag}".`);
        }

        // fields[6..23] are the 18 named genre flags, in genreNames order.
        const genres = [];
        for (let g = 0; g < GENRE_COUNT; g++) {
            const flag = fields[GENRE_FLAG_START + g];
            if (flag !== '0' && flag !== '1') {
                throw new Error(
                    `u.item line ${lineNumber}: genre flag "${genreNames[g]}" for movie ` +
                    `${id} must be 0 or 1, found "${flag}".`);
            }
            if (flag === '1') genres.push(genreNames[g]);
        }

        movies.push({ id: id, title: title, genres: genres, unknownGenre: unknownFlag === '1' });
    }
}

// ---------------------------------------------------------------------------
// parseRatingData — strict u.data parser
// ---------------------------------------------------------------------------
function parseRatingData(text) {
    const lines = text.split('\n');
    let lineNumber = 0;

    for (const line of lines) {
        lineNumber++;
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length !== RATING_FIELD_COUNT) {
            throw new Error(
                `u.data line ${lineNumber}: expected ${RATING_FIELD_COUNT} ` +
                `tab separated fields, found ${fields.length}.`);
        }

        // Number() (not parseInt) so that "3abc" is rejected rather than read as 3.
        const userId = Number(fields[0]);
        const itemId = Number(fields[1]);
        const rating = Number(fields[2]);
        const timestamp = Number(fields[3]);

        if (!Number.isInteger(userId) || userId <= 0) {
            throw new Error(
                `u.data line ${lineNumber}: userId must be a positive integer, ` +
                `found "${fields[0]}".`);
        }
        if (!Number.isInteger(itemId) || itemId <= 0) {
            throw new Error(
                `u.data line ${lineNumber}: itemId must be a positive integer, ` +
                `found "${fields[1]}".`);
        }
        if (!Number.isInteger(rating) || rating < MIN_RATING || rating > MAX_RATING) {
            throw new Error(
                `u.data line ${lineNumber}: rating must be an integer in ` +
                `${MIN_RATING}..${MAX_RATING}, found "${fields[2]}".`);
        }
        if (!Number.isInteger(timestamp) || timestamp < 0) {
            throw new Error(
                `u.data line ${lineNumber}: timestamp must be a non-negative ` +
                `integer, found "${fields[3]}".`);
        }

        ratings.push({ userId: userId, itemId: itemId, rating: rating, timestamp: timestamp });
    }
}

// ---------------------------------------------------------------------------
// buildRatingMatrix
//
// Required shape and convention:
//     ratingMatrix[userId][movieId] === rating      (1..5)
//     a missing rating is 0
//     rows    = numUsers + 1        (row 0 unused)
//     columns = max raw movie id + 1 (column 0 unused)
//   => 944 x 1683 for this dataset.
//
// Values are 0..5, so each row is a Uint8Array: 1 588 752 bytes total instead of
// 1 588 752 JS number slots, with the same ratingMatrix[u][i] access pattern.
//
// The same single pass also fills the sparse indexes and detects duplicate
// (userId, itemId) pairs.
// ---------------------------------------------------------------------------
function buildRatingMatrix() {
    // Rebuild from scratch. buildRatingMatrix() is therefore idempotent and can
    // be re-invoked after ratings/movies change without corrupting the indexes.
    userRatings = new Map();
    itemRatings = new Map();
    itemById = new Map();
    userIds = [];
    itemIds = [];

    // --- dimensions -------------------------------------------------------
    // numUsers is the highest RAW user id. On this dataset ids are contiguous
    // 1..943 so it also equals the user count, but the matrix dimension depends
    // on the raw id while the dropdown uses the actually-present ids (userIds).
    let maxUserId = 0;
    for (const r of ratings) {
        if (r.userId > maxUserId) maxUserId = r.userId;
    }
    numUsers = maxUserId;

    numMovies = movies.length;

    let maxMovieId = 0;
    for (const m of movies) {
        if (m.id > maxMovieId) maxMovieId = m.id;
    }
    matrixColumns = maxMovieId + 1;   // always >= 1, so column 0 always exists

    // numUsers === 0 means u.data contributed no usable row. An empty u.item
    // cannot happen without also failing the itemId range check below.
    if (numUsers === 0) {
        throw new Error('No usable ratings were parsed from u.data.');
    }

    // --- dense required matrix -------------------------------------------
    ratingMatrix = new Array(numUsers + 1);
    for (let u = 0; u <= numUsers; u++) {
        ratingMatrix[u] = new Uint8Array(matrixColumns);
    }

    // --- sparse indexes + duplicate detection, one pass ------------------
    for (const movie of movies) {
        itemById.set(movie.id, movie);
    }

    const duplicates = [];
    for (const r of ratings) {
        if (r.itemId >= matrixColumns) {
            throw new Error(
                `u.data references movie id ${r.itemId}, which is outside the ` +
                `u.item id range 1..${maxMovieId}.`);
        }

        let row = userRatings.get(r.userId);
        if (row === undefined) {
            row = new Map();
            userRatings.set(r.userId, row);
        }

        if (row.has(r.itemId)) {
            if (duplicates.length < 10) {
                duplicates.push(`(user ${r.userId}, movie ${r.itemId})`);
            }
            continue; // first rating wins; the error below aborts the load anyway
        }

        row.set(r.itemId, r.rating);
        ratingMatrix[r.userId][r.itemId] = r.rating;

        let column = itemRatings.get(r.itemId);
        if (column === undefined) {
            column = new Map();
            itemRatings.set(r.itemId, column);
        }
        column.set(r.userId, r.rating);
    }

    if (duplicates.length > 0) {
        throw new Error(
            'u.data contains duplicate (userId, itemId) pairs: ' +
            duplicates.join(', ') + '. Each pair must appear at most once.');
    }

    // --- authoritative id lists ------------------------------------------
    userIds = Array.from(userRatings.keys()).sort(function (a, b) { return a - b; });
    itemIds = Array.from(itemRatings.keys()).sort(function (a, b) { return a - b; });
}
