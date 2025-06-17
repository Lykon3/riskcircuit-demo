# 🧠 RiskCircuit

> ⚠️ **Predictive Compliance Engine for DeFi**  
> “Entropy doesn’t lie. Regulators just show up late.”

RiskCircuit is a real-time, entropy-weighted compliance engine that detects wallet-based risk **before it becomes obvious**.  
We don’t just score behavior. We **model instability**, map **adjacency to bad actors**, and apply **Bayesian reasoning** to decode the future of DeFi.

Built for:
- Institutional allocators entering a fragmented space
- Founders who want to stay 10 steps ahead of regulators
- Auditors who don’t have time to scan 50k wallets manually

---

## 🔍 What It Does

| Capability | Description |
|------------|-------------|
| 🧮 **Entropy Engine** | Measures chaos across time, gas, protocol, and tx value. Composite scores + Lyapunov stability. |
| 🌐 **Adjacency Graph** | Maps wallet relationships to flagged actors up to 3 hops out. Decay-based threat propagation. |
| 🧠 **Bayesian Risk Model** | Wallet age × funding source × network priors → probabilistic future risk. |
| 🧾 **PDF Audit Report** | Downloadable report w/ glyph, threat summary, and recommendations (coming soon). |
| 🎯 **Risk Glyphs** | Entropy-based SVG fingerprint for sharing, scanning, and social flexing. |

---

## 🔥 Sample Output

- `Risk Score: 87.4/100`
- `Interpretation: CRITICAL RISK — adjacency to flagged Tornado fork + unstable entropy trajectory`
- `Recommendations:`  
  - Freeze funds  
  - Flag for review  
  - Monitor new inbound txs

---

## 🧱 Stack

- **Node.js / Express** backend
- **Ethers.js** for wallet history
- **TensorFlow.js** for entropy modeling
- **Bayesian CPT engine** for forward prediction
- **Cytoscape.js** for adjacency graphing
- **TailwindCSS** UI
- **SVG Glyph Generator** for visual output
- **jsPDF (soon)** for printable reports

---

## 🚀 Run It Locally

```bash
# Install
npm install

# Set your Ethereum RPC endpoint in index.js
# (e.g. Infura, Alchemy, or your own node)

# Start API
node index.js

# Open public/index.html in your browser
