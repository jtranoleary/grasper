// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is a backend server that maps frontend requests to Google
// Cloud Storage.

import express from 'express';
import { Storage } from '@google-cloud/storage';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let titlesMap = {};
try {
  titlesMap = JSON.parse(fs.readFileSync('./titles.json', 'utf-8'));
} catch (e) {
  console.log('No titles.json found, skipping enhancement.');
}

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

const storage = new Storage();
const bucketName = process.env.GCS_BUCKET_NAME || 'YOUR_GCS_BUCKET_NAME';

app.get('/api/videos', async (req, res) => {
  try {
    const [files] = await storage.bucket(bucketName).getFiles();

    // Map files to a structured response
    // Generate Signed URLs for each file to allow secure streaming
    // Map files to local proxy stream URL
    const videos = files
      .filter(file => file.name.endsWith('.mp4') || file.name.endsWith('.mov'))
      .map(file => {
        const fileKey = file.name.split('/').pop()?.replace('.mp4', '').replace('.mov', '') || '';
        let mappedTitle = titlesMap[fileKey];

        if (!mappedTitle) {
          const match = fileKey.match(/^clip_(\d+)_/);
          if (match) {
            const epNum = match[1];
            mappedTitle = titlesMap[`episode_${epNum}`];
          }
        }

        if (!mappedTitle) {
          mappedTitle = file.name.split('/').pop()?.replace('.mp4', '').replace(/_/g, ' ') || file.name;
        }

        return {
          id: file.id,
          name: file.name,
          title: mappedTitle,
          // Pointing to our local proxy stream endpoint using query parameter
          url: `/api/video?name=${encodeURIComponent(file.name)}`,
          size: file.metadata.size,
          updated: file.metadata.updated
        };
      });

    res.json(videos);
  } catch (error) {
    console.error('Error fetching files from bucket:', error);
    res.status(500).json({ error: 'Failed to fetch videos from bucket' });
  }
});

// Stream endpoint to pipe data directly from GCP to client via query param
app.get('/api/video', async (req, res) => {
  console.log('--- HIT /api/video ---', req.query.name);
  const fileName = req.query.name;
  if (!fileName) {
    return res.status(400).send('Missing name parameter');
  }
  const file = storage.bucket(bucketName).file(fileName);

  try {
    const [metadata] = await file.getMetadata();

    // Set proper headers for streaming
    res.setHeader('Content-Type', metadata.contentType || 'video/mp4');
    res.setHeader('Content-Length', metadata.size);
    res.setHeader('Accept-Ranges', 'bytes');

    // Pipe the read stream directly to response
    file.createReadStream()
      .on('error', (err) => {
        console.error('Stream error:', err);
        if (!res.headersSent) {
          res.status(500).send('Error streaming video');
        }
      })
      .pipe(res);

  } catch (error) {
    console.error('Error fetching metadata for stream:', error);
    res.status(404).send('Video not found');
  }
});

// Mock Analysis Endpoint
app.post('/api/analyze', (req, res) => {
  const { filename } = req.body;
  const choices = [
    {
      overallAnalysis: "The thermal profile suggests excessive sofi-ing load constraints leading to lateral profile distortion.",
      hotspots: [
        {
          id: 'excessive-sofiing',
          top: '51%',
          left: '29%',
          label: 'Excessive Sofi-ing',
          clipKey: 'clip_g3_17',
          sequenceKey: 'incorrect-sofietta',
          correctSequenceKey: 'correct-sofietta',
          desc: "Excessive air and pressure with the sofietta created a bulge in the vessel wall. That part was never fixed before the lip was opened."
        }
      ]
    },
    {
      overallAnalysis: "Thermal diagnostics reveal rim overheating leading to concentric angle drift profile.",
      hotspots: [
        {
          id: 'overheated-flared-lip',
          top: '33%',
          left: '50%',
          label: 'Overheated Flared Lip',
          clipKey: 'clip_1_10',
          sequenceKey: 'over-flare',
          correctSequenceKey: 'correct-flare',
          desc: "The cup was overheated and the jack angles were not in line with the rest of the cup."
        }
      ]
    },
    {
      overallAnalysis: "Analysis shows that the base became too thin during the inflation stage, likely due to blowing while the base was too hot.",
      hotspots: [
        {
          id: 'broken-thin-base',
          top: '25%',
          left: '55%',
          label: 'Broken Thin Base',
          clipKey: 'clip_7_1',
          sequenceKey: 'too-thin-base',
          correctSequenceKey: 'correct-base-thickness',
          desc: "The base needs to be kept cool and slightly thicker than the rest of the vessel during inflation, either by using the marver, the strap of the jacks, newspaper, or related cooling tools."
        }
      ]
    },
    {
      overallAnalysis: "The thermal profile suggests uneven heating on the blowpipe and excessive blowing before the top part was sufficiently hot.",
      hotspots: [
        {
          id: 'waist-inwards',
          top: '45%',
          left: '24%',
          label: 'Waist Going Inwards',
          clipKey: 'clip_g3_8',
          sequenceKey: 'shallow-heat-puff',
          correctSequenceKey: 'correct-heat-puff',
          desc: "The waist is going inwards due to not heating the cup evenly when still on the blow pipe, blowing too much when only the bottom part was hot, and not using the jacks while inflating to keep the side edges straight."
        }
      ]
    },
  ];

  // Deterministic selection to simulate "analysis" stability
  let choiceIndex = 0;
  if (filename && filename.toLowerCase().includes('flared')) {
    choiceIndex = 1;
  } else if (filename && filename.toLowerCase().includes('expanded')) {
    choiceIndex = 0;
  } else if (filename && filename.toLowerCase().includes('broken')) {
    choiceIndex = 2;
  } else if (filename && filename.toLowerCase().includes('waist')) {
    choiceIndex = 3;
  } else {
    // Fallback item
    choiceIndex = Math.floor(Math.random() * choices.length);
  }
  res.json(choices[choiceIndex]);
});

// --- GCS Sequences Shared Storage Endpoints ---

// 1. Save a sequence payload
app.post('/api/sequences', async (req, res) => {
  const { name, data } = req.body;
  if (!name || !data) {
    return res.status(400).send('Missing name or sequence data');
  }

  try {
    const file = storage.bucket(bucketName).file(`sequences/${name}.json`);
    await file.save(JSON.stringify(data), {
      contentType: 'application/json',
      resumable: false
    });
    res.json({ success: true, message: `Saved ${name}` });
  } catch (error) {
    console.error('Error saving sequence:', error);
    res.status(500).json({ error: 'Failed to write sequence to GCS' });
  }
});

// 2. List all saved sequences names
app.get('/api/sequences', async (req, res) => {
  try {
    const [files] = await storage.bucket(bucketName).getFiles({ prefix: 'sequences/' });
    const names = files
      .filter(file => file.name.endsWith('.json'))
      .map(file => file.name.replace('sequences/', '').replace('.json', ''));
    res.json(names);
  } catch (error) {
    console.error('Error listing sequences:', error);
    res.status(500).json({ error: 'Failed to list sequences from GCS' });
  }
});

// 3. Load a specific sequence payload
app.get('/api/sequences/:name', async (req, res) => {
  const { name } = req.params;
  try {
    const file = storage.bucket(bucketName).file(`sequences/${name}.json`);
    const [content] = await file.download();
    const sequenceData = JSON.parse(content.toString());
    res.json(sequenceData);
  } catch (error) {
    console.error('Error loading sequence:', error);
    res.status(404).json({ error: `Sequence ${name} not found` });
  }
});

// 4. Delete a sequence payload
app.delete('/api/sequences/:name', async (req, res) => {
  const { name } = req.params;
  try {
    const file = storage.bucket(bucketName).file(`sequences/${name}.json`);
    await file.delete();
    res.json({ success: true, message: `Deleted ${name}` });
  } catch (error) {
    console.error('Error deleting sequence:', error);
    res.status(500).json({ error: `Failed to delete sequence ${name}` });
  }
});

// Serve static files from the React app build directory
app.use(express.static(path.join(__dirname, 'dist')));

// The "catchall" handler: for any request that doesn't
// match one above, send back React's index.html file.
app.get(/.*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(port, () => {
  console.log(`Backend proxy listening on port ${port}`);
});
