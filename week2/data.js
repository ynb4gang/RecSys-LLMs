// Global variables for storing movie and rating data.
// Both are re-initialised inside loadData(), so calling loadData() more than
// once can never duplicate parsed records.
let movies = [];
let ratings = [];

// The 18 named MovieLens 100K genres, in u.item column order (Action .. Western).
//
// HW A02 starter defect fixed here: a u.item row contains 5 metadata fields
// followed by 19 binary flags:
//     1 "unknown" flag  +  these 18 named genre flags  =  19 flags, 24 fields total.
// The starter code sliced 19 flags but paired them with only 18 names, which
// shifted every label by one position (unknown was read as "Action", Action as
// "Adventure", ..., War as "Western") and dropped the real Western flag entirely.
// The "unknown" flag is NOT a genre and must never be mapped to "Action".
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// u.item layout constants (0-based field indices after splitting on '|').
const U_ITEM_FIELD_COUNT = 24;  // 5 metadata + 1 unknown flag + 18 genre flags
const UNKNOWN_FLAG_INDEX = 5;   // "unknown" flag - validated but never used as a genre
const GENRE_FLAG_START = 6;     // first named genre flag ("Action")
const GENRE_COUNT = 18;         // exactly 18 named genres => 18-dim feature vector

// Primary function to load data from files
async function loadData() {
    try {
        // Reset outputs first: repeated loadData() calls must not duplicate records.
        movies = [];
        ratings = [];

        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from the u.item format.
//
// Correct mapping (verified against the actual data file):
//   fields 0..4   -> id | title | release date | video release date | IMDb URL
//   field  5      -> "unknown" flag (skipped - not a genre)
//   fields 6..23  -> the 18 named genre flags, Action..Western, in order
//
// Every row is validated strictly: exactly 24 fields, a positive integer id,
// a non-empty title, and all 19 flag fields must be exactly '0' or '1'.
function parseItemData(text) {
    const lines = text.split('\n');
    let skipped = 0;

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length !== U_ITEM_FIELD_COUNT) {
            skipped++; // malformed row: refuse to index fields we cannot trust
            continue;
        }

        const id = parseInt(fields[0], 10);
        const title = fields[1].trim();
        if (!Number.isInteger(id) || id <= 0 || title === '') {
            skipped++;
            continue;
        }

        let flagsValid = true;
        for (let i = UNKNOWN_FLAG_INDEX; i < U_ITEM_FIELD_COUNT; i++) {
            if (fields[i] !== '0' && fields[i] !== '1') {
                flagsValid = false;
                break;
            }
        }
        if (!flagsValid) {
            skipped++;
            continue;
        }

        // 18-dimensional binary genre feature vector (unknown flag excluded).
        const vector = [];
        for (let g = 0; g < GENRE_COUNT; g++) {
            vector.push(fields[GENRE_FLAG_START + g] === '1' ? 1 : 0);
        }

        // Human-readable genres derived from the same vector, so the UI text and
        // the cosine calculation can never disagree.
        const genres = genreNames.filter((_, index) => vector[index] === 1);

        movies.push({ id, title, genres, vector });
    }

    if (skipped > 0) {
        console.warn(`parseItemData: skipped ${skipped} malformed u.item row(s).`);
    }
}

// Parse rating data from the u.data format: userId \t itemId \t rating \t timestamp.
// Strictly validates that each row has exactly 4 fields with parseable values.
// Ratings are used ONLY to derive per-movie rating counts (evaluation metadata);
// they never enter the content-based similarity score.
function parseRatingData(text) {
    const lines = text.split('\n');
    let skipped = 0;

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length !== 4) {
            skipped++;
            continue;
        }

        const userId = parseInt(fields[0], 10);
        const itemId = parseInt(fields[1], 10);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3], 10);

        if (!Number.isInteger(userId) || !Number.isInteger(itemId) ||
            !Number.isFinite(rating) || !Number.isInteger(timestamp)) {
            skipped++;
            continue;
        }

        ratings.push({ userId, itemId, rating, timestamp });
    }

    if (skipped > 0) {
        console.warn(`parseRatingData: skipped ${skipped} malformed u.data row(s).`);
    }
}
