require("dotenv").config();
const fs = require("fs").promises;
const { parse } = require("csv-parse");
const { stringify } = require("csv-stringify");
const spotifyPreviewFinder = require("spotify-preview-finder");

// Path to input and output files
const INPUT_FILE = "../data/processed/unique_tracks.csv";
const OUTPUT_FILE = "../data/processed/tracks_with_previews.csv";
const BATCH_SIZE = 10; // Number of tracks to process in parallel

async function readCsvFile(filePath) {
  const content = await fs.readFile(filePath, "utf-8");
  return new Promise((resolve, reject) => {
    parse(
      content,
      {
        columns: true,
        skip_empty_lines: true,
      },
      (err, data) => {
        if (err) reject(err);
        else resolve(data);
      }
    );
  });
}

async function writeCsvFile(filePath, data) {
  const csvContent = await new Promise((resolve, reject) => {
    stringify(data, { header: true }, (err, output) => {
      if (err) reject(err);
      else resolve(output);
    });
  });
  await fs.writeFile(filePath, csvContent);
}

async function getPreviewUrl(track) {
  try {
    const result = await spotifyPreviewFinder(
      track.track_name,
      track.artist_name,
      1
    );
    if (result.success && result.results.length > 0) {
      const matchedTrack = result.results[0];
      return {
        ...track,
        preview_url: matchedTrack.previewUrls[0] || null,
        spotify_popularity: matchedTrack.popularity,
        matched_spotify_uri: matchedTrack.trackId,
        match_confidence:
          matchedTrack.trackId === track.spotify_track_uri.split(":")[2]
            ? "high"
            : "low",
      };
    }
  } catch (error) {
    console.error(
      `Error fetching preview for ${track.track_name}: ${error.message}`
    );
  }
  return {
    ...track,
    preview_url: null,
    spotify_popularity: null,
    matched_spotify_uri: null,
    match_confidence: null,
  };
}

async function processBatch(tracks) {
  return Promise.all(tracks.map((track) => getPreviewUrl(track)));
}

async function main() {
  try {
    // Read unique tracks
    console.log("Reading unique tracks...");
    const tracks = await readCsvFile(INPUT_FILE);
    console.log(`Found ${tracks.length} tracks to process`);

    // Process tracks in batches
    const results = [];
    for (let i = 0; i < tracks.length; i += BATCH_SIZE) {
      const batch = tracks.slice(i, i + BATCH_SIZE);
      console.log(
        `Processing batch ${i / BATCH_SIZE + 1} of ${Math.ceil(
          tracks.length / BATCH_SIZE
        )}`
      );
      const batchResults = await processBatch(batch);
      results.push(...batchResults);

      // Save progress after each batch
      await writeCsvFile(OUTPUT_FILE, results);
      console.log(
        `Progress: ${results.length}/${tracks.length} tracks processed`
      );

      // Add a small delay to avoid rate limiting
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    console.log("All done! Results saved to:", OUTPUT_FILE);
  } catch (error) {
    console.error("Error:", error.message);
    process.exit(1);
  }
}

main();
