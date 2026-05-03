const express = require('express');
const multer = require('multer');
const { createClient } = require('redis');
const cors = require('cors');
const path = require('path'); // Added this

const app = express();
app.use(cors());

// Changed destination to a local folder
const upload = multer({ dest: './data/' }); 

// Changed URL to localhost
const redisClient = createClient({ url: 'redis://localhost:6379' });
redisClient.connect().catch(console.error);

app.post('/upload', upload.single('dataset'), async (req, res) => {
    try {
        // Convert the relative path to an absolute path for Python
        const absolutePath = path.resolve(req.file.path); 
        const tournamentId = req.file.filename;

        const methods = ['Mean', 'KNN', 'AKNN_TIQR'];
        for (const method of methods) {
            // Pass the absolute path to Redis
            const task = JSON.stringify({ filepath: absolutePath, method, tournamentId });
            await redisClient.lPush('impute_queue', task);
        }

        res.status(202).json({
            message: "Tournament Started! Workers notified.",
            tournamentId: tournamentId
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/status/:tournamentId', async (req, res) => {
    const tournamentId = req.params.tournamentId;
    const results = await redisClient.hGetAll(`results:${tournamentId}`);
    const parsedResults = Object.values(results).map(r => JSON.parse(r));

    if (parsedResults.length === 0) return res.json({ status: "Processing..." });

    parsedResults.sort((a, b) => a.rmse - b.rmse);

    res.json({
        status: parsedResults.length === 3 ? "Complete" : `Partial Results (${parsedResults.length}/3)`,
        winner: parsedResults.length === 3 ? parsedResults[0].method : "Calculating...",
        leaderboard: parsedResults
    });
});

app.listen(5000, () => console.log('Orchestrator running on port 5000'));