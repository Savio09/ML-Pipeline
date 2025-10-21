#!/usr/bin/env node

// Load environment variables from .env
require("dotenv").config();
const spotifyPreviewFinder = require("spotify-preview-finder");

// Support both SPOTIFY_CLIENT_ID/SECRET and CLIENT_ID/SECRET
if (!process.env.SPOTIFY_CLIENT_ID && process.env.CLIENT_ID) {
  process.env.SPOTIFY_CLIENT_ID = process.env.CLIENT_ID;
}
if (!process.env.SPOTIFY_CLIENT_SECRET && process.env.CLIENT_SECRET) {
  process.env.SPOTIFY_CLIENT_SECRET = process.env.CLIENT_SECRET;
}

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length < 2) {
  console.error(
    "Usage: node spotify_preview_finder.js <track_name> <artist_name> [limit]"
  );
  process.exit(1);
}
const trackName = args[0];
const artistName = args[1];
const limit = args[2] ? parseInt(args[2], 10) : 5;

async function main() {
  try {
    const result = await spotifyPreviewFinder(trackName, artistName, limit);
    if (
      result.success &&
      Array.isArray(result.results) &&
      result.results.length > 0
    ) {
      // Output a clear JSON structure for downstream use
      const tracks = result.results.map((song) => ({
        name: song.name,
        trackId: song.trackId,
        albumName: song.albumName,
        releaseDate: song.releaseDate,
        popularity: song.popularity,
        durationMs: song.durationMs,
        spotifyUrl: song.spotifyUrl,
        previewUrls: song.previewUrls,
      }));
      console.log(
        JSON.stringify({
          success: true,
          searchQuery: result.searchQuery,
          tracks: tracks,
        })
      );
    } else {
      console.log(
        JSON.stringify({
          success: false,
          error: result.error || "No preview found",
        })
      );
    }
  } catch (err) {
    console.log(
      JSON.stringify({
        success: false,
        error: err.message || "Unknown error",
      })
    );
  }
}

main();
