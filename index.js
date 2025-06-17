// index.js — RiskCircuit Express API Wrapper
import express from 'express';
import cors from 'cors';
import { ethers } from 'ethers';
import { RiskEngine } from './src/lib/riskEngine.js'; // Assumes main engine lives here

const app = express();
const PORT = process.env.PORT || 3001;
app.use(cors());

// Configure provider (use your preferred RPC endpoint)
const provider = new ethers.JsonRpcProvider("https://mainnet.infura.io/v3/YOUR_INFURA_KEY");
const engine = new RiskEngine(provider);

app.get('/analyze/:address', async (req, res) => {
  const address = req.params.address;
  try {
    const result = await engine.analyzeWallet(address);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`RiskCircuit API live on port ${PORT}`);
});

// To run:
// 1. npm install express cors ethers
// 2. node index.js
