const express = require('express');
const multer = require('multer');
const { createClient } = require('redis');
const cors = require('cors');
const path = require('path');
const crypto = require('crypto');

const app = express();
app.use(cors());
app.use(express.json()); // IMPORTANT: This allows the /upload-link route to read JSON

const upload = multer({ dest: './data/' }); 
const dataDir = process.env.DATA_DIR || path.resolve('./data');
const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
const redisClient = createClient({ url: redisUrl });
redisClient.connect().catch(console.error);

const AVAILABLE_METHODS = ['Mean', 'KNN', 'AKNN_TIQR', 'MICE', 'Isolation Forest'];

function safeFilename(value) {
    return String(value).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '') || 'imputed';
}

// --- HELPER FUNCTION: Manages the Redis Queue for both Files and Links ---
async function queueTournament(filepath, tournamentId, requestedMethod, targetCol, res) {
    targetCol = typeof targetCol === 'string' ? targetCol.trim() : targetCol;
    if (!targetCol) {
        return res.status(400).json({ error: "You must specify a 'targetCol'." });
    }

    let methodsToRun = [];
    if (requestedMethod.toLowerCase() === 'all') {
        methodsToRun = AVAILABLE_METHODS;
    } else if (AVAILABLE_METHODS.includes(requestedMethod)) {
        methodsToRun = [requestedMethod];
    } else {
        return res.status(400).json({ 
            error: `Invalid method. Choose 'all' or: ${AVAILABLE_METHODS.join(', ')}` 
        });
    }

    // Store the expected number of methods in Redis
    await redisClient.set(`expected:${tournamentId}`, methodsToRun.length);

    // Push tasks to the Python workers
    for (const method of methodsToRun) {
        const task = JSON.stringify({ 
            filepath: filepath, 
            method, 
            tournamentId,
            targetCol,
            outputDir: dataDir
        });
        await redisClient.lPush('impute_queue', task);
    }

    res.status(202).json({
        message: `Tournament Started! Workers notified.`,
        id: tournamentId,
        methodsQueued: methodsToRun
    });
}

// ==========================================
// ROUTE 1: UPLOAD PHYSICAL FILE (form-data)
// ==========================================
app.post('/upload', upload.single('dataset'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded." });
        }

        const absolutePath = path.resolve(req.file.path); 
        const tournamentId = req.file.filename;
        const targetCol = req.body.targetCol;
        const requestedMethod = req.body.method || 'all';

        await queueTournament(absolutePath, tournamentId, requestedMethod, targetCol, res);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ==========================================
// ROUTE 2: UPLOAD VIA URL (JSON body)
// ==========================================
app.post('/upload-link', async (req, res) => {
    try {
        const datasetUrl = req.body.url;
        const targetCol = req.body.targetCol;
        const requestedMethod = req.body.method || 'all';
        
        if (!datasetUrl) {
            return res.status(400).json({ error: "You must provide a 'url'." });
        }

        // Generate a random ID since we don't have a physical filename
        const tournamentId = crypto.randomBytes(8).toString('hex'); 

        await queueTournament(datasetUrl, tournamentId, requestedMethod, targetCol, res);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/status/:tournamentId', async (req, res) => {
    try {
        const tournamentId = req.params.tournamentId;
        
        const expectedCountStr = await redisClient.get(`expected:${tournamentId}`);
        if (!expectedCountStr) {
            return res.status(404).json({ error: "Tournament not found" });
        }
        const expectedCount = parseInt(expectedCountStr, 10);

        const results = await redisClient.hGetAll(`results:${tournamentId}`);
        
        const parsedResults = [];
        for (const key in results) {
            try {
                const safeJsonStr = results[key].replace(/Infinity/g, '999999999');
                parsedResults.push(JSON.parse(safeJsonStr));
            } catch (parseError) {
                console.error(`Failed to parse result for ${key}:`, results[key]);
            }
        }

        if (parsedResults.length === 0) return res.json({ status: "Processing..." });

        parsedResults.sort((a, b) => a.rmse - b.rmse);
        const isComplete = parsedResults.length === expectedCount;

        res.json({
            status: isComplete ? "Complete" : `Partial Results (${parsedResults.length}/${expectedCount})`,
            winner: isComplete ? parsedResults[0].method : "Calculating...",
            leaderboard: parsedResults
        });

    } catch (err) {
        console.error("Status Route Error:", err);
        res.status(500).json({ error: "Internal server error fetching status." });
    }
});

app.get('/download/:tournamentId/:method', async (req, res) => {
    try {
        const tournamentId = req.params.tournamentId;
        const method = decodeURIComponent(req.params.method);

        const resultStr = await redisClient.hGet(`results:${tournamentId}`, method);
        if (!resultStr) {
            return res.status(404).json({ error: "Imputed CSV not found for this method." });
        }

        const result = JSON.parse(resultStr.replace(/Infinity/g, '999999999'));
        if (!result.csvPath) {
            return res.status(404).json({ error: "This method has not produced an imputed CSV yet." });
        }

        const csvPath = path.resolve(result.csvPath);
        const filename = `${safeFilename(tournamentId)}_${safeFilename(method)}_imputed.csv`;
        res.download(csvPath, filename, (err) => {
            if (err && !res.headersSent) {
                console.error("Download Route Error:", err);
                res.status(404).json({ error: "Imputed CSV file is missing on disk." });
            }
        });
    } catch (err) {
        console.error("Download Route Error:", err);
        res.status(500).json({ error: "Internal server error downloading imputed CSV." });
    }
});

app.listen(5000, () => console.log('Orchestrator running on port 5000'));
