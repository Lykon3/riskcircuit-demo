// RiskCircuit Core Engine - Enhanced Production Version
// Advanced blockchain risk analysis with real-time monitoring and ML capabilities


import { ethers } from 'ethers';
import * as tf from '@tensorflow/tfjs';
import { ApolloClient, InMemoryCache, gql } from '@apollo/client';
import { WebSocketProvider } from '@ethersproject/providers';
import Redis from 'ioredis';
import pino from 'pino';


// Logger setup
const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true
    }
  }
});


// ============================================
// CONFIGURATION
// ============================================


const CONFIG = {
  ALCHEMY_API_KEY: process.env.ALCHEMY_API_KEY,
  GRAPH_API_URL: process.env.GRAPH_API_URL || 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3',
  REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379',
  OFAC_API_URL: 'https://api.ofac-api.com/v3/sanctions',
  CACHE_TTL: 3600, // 1 hour
  RISK_THRESHOLDS: {
    CRITICAL: 85,
    HIGH: 60,
    MODERATE: 35,
    LOW: 10
  },
  ANOMALY_DETECTION: {
    WINDOW_SIZE: 100,
    STD_THRESHOLD: 3,
    MIN_SAMPLES: 10
  }
};


// ============================================
// DATA SOURCE INTEGRATIONS
// ============================================


class DataSourceManager {
  constructor() {
    this.alchemyProvider = new ethers.AlchemyProvider('mainnet', CONFIG.ALCHEMY_API_KEY);
    this.wsProvider = new WebSocketProvider(`wss://eth-mainnet.g.alchemy.com/v2/${CONFIG.ALCHEMY_API_KEY}`);
    this.graphClient = new ApolloClient({
      uri: CONFIG.GRAPH_API_URL,
      cache: new InMemoryCache()
    });
    this.redis = new Redis(CONFIG.REDIS_URL);
    this.ofacCache = new Map();
    this.lastOFACUpdate = 0;
  }


  async getEnhancedTransactionHistory(address, limit = 1000) {
    const cacheKey = `txHistory:${address}:${limit}`;
    const cached = await this.redis.get(cacheKey);
    if (cached) return JSON.parse(cached);


    try {
      // Use Alchemy's enhanced API for better performance
      const response = await fetch(
        `https://eth-mainnet.g.alchemy.com/v2/${CONFIG.ALCHEMY_API_KEY}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0',
            id: 1,
            method: 'alchemy_getAssetTransfers',
            params: [{
              fromBlock: '0x0',
              toBlock: 'latest',
              fromAddress: address,
              category: ['external', 'internal', 'erc20', 'erc721', 'erc1155'],
              maxCount: `0x${limit.toString(16)}`,
              excludeZeroValue: false
            }]
          })
        }
      );


      const data = await response.json();
      const transfers = data.result.transfers;


      // Enrich with protocol identification
      const enrichedTransfers = await this.enrichTransfersWithProtocols(transfers);


      await this.redis.setex(cacheKey, CONFIG.CACHE_TTL, JSON.stringify(enrichedTransfers));
      return enrichedTransfers;
    } catch (error) {
      logger.error({ error, address }, 'Failed to fetch enhanced transaction history');
      throw error;
    }
  }


  async enrichTransfersWithProtocols(transfers) {
    // Query The Graph for protocol identification
    const addresses = [...new Set(transfers.map(t => t.to).filter(Boolean))];
    const PROTOCOL_QUERY = gql`
      query GetProtocols($addresses: [String!]!) {
        protocols(where: { id_in: $addresses }) {
          id
          name
          type
          totalValueLockedUSD
        }
      }
    `;


    try {
      const { data } = await this.graphClient.query({
        query: PROTOCOL_QUERY,
        variables: { addresses: addresses.map(a => a.toLowerCase()) }
      });


      const protocolMap = new Map(data.protocols.map(p => [p.id, p]));
      
      return transfers.map(transfer => ({
        ...transfer,
        protocol: protocolMap.get(transfer.to?.toLowerCase()) || null
      }));
    } catch (error) {
      logger.warn({ error }, 'Failed to enrich transfers with protocol data');
      return transfers;
    }
  }


  async checkOFACSanctions(address) {
    // Update OFAC cache every 24 hours
    if (Date.now() - this.lastOFACUpdate > 86400000) {
      await this.updateOFACCache();
    }


    return this.ofacCache.has(address.toLowerCase());
  }


  async updateOFACCache() {
    try {
      const response = await fetch(`${CONFIG.OFAC_API_URL}/addresses`);
      const data = await response.json();
      
      this.ofacCache.clear();
      data.addresses.forEach(addr => {
        this.ofacCache.set(addr.toLowerCase(), true);
      });
      
      this.lastOFACUpdate = Date.now();
      logger.info({ count: this.ofacCache.size }, 'Updated OFAC sanctions cache');
    } catch (error) {
      logger.error({ error }, 'Failed to update OFAC cache');
    }
  }


  setupRealtimeMonitoring(address, callback) {
    // Monitor incoming/outgoing transactions in real-time
    const filter = {
      topics: [
        null,
        [ethers.zeroPadValue(address, 32), null],
        [null, ethers.zeroPadValue(address, 32)]
      ]
    };


    this.wsProvider.on(filter, (log) => {
      callback({
        type: 'transaction',
        address,
        log,
        timestamp: Date.now()
      });
    });


    logger.info({ address }, 'Real-time monitoring established');
    
    return () => {
      this.wsProvider.off(filter);
      logger.info({ address }, 'Real-time monitoring stopped');
    };
  }
}


// ============================================
// MACHINE LEARNING ANOMALY DETECTION
// ============================================


class AnomalyDetector {
  constructor() {
    this.model = null;
    this.scaler = { mean: 0, std: 1 };
    this.featureBuffer = [];
  }


  async initialize() {
    // Load pre-trained autoencoder model for anomaly detection
    try {
      this.model = await tf.loadLayersModel('/models/anomaly_detector.json');
      logger.info('Anomaly detection model loaded');
    } catch (error) {
      logger.warn('No pre-trained model found, creating new one');
      this.model = this.createAutoencoder();
    }
  }


  createAutoencoder() {
    const encoder = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [10], units: 8, activation: 'relu' }),
        tf.layers.dense({ units: 4, activation: 'relu' }),
        tf.layers.dense({ units: 2, activation: 'relu' })
      ]
    });


    const decoder = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [2], units: 4, activation: 'relu' }),
        tf.layers.dense({ units: 8, activation: 'relu' }),
        tf.layers.dense({ units: 10, activation: 'sigmoid' })
      ]
    });


    const autoencoder = tf.sequential({
      layers: [...encoder.layers, ...decoder.layers]
    });


    autoencoder.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError'
    });


    return autoencoder;
  }


  extractFeatures(transactionData) {
    // Extract 10 key features for anomaly detection
    const features = [
      transactionData.valueStd / (transactionData.valueMean + 1e-8), // Coefficient of variation
      transactionData.timeIntervalStd / (transactionData.timeIntervalMean + 1e-8),
      transactionData.gasStd / (transactionData.gasMean + 1e-8),
      transactionData.uniqueProtocolRatio,
      transactionData.nightTransactionRatio,
      transactionData.failedTransactionRatio,
      transactionData.contractCreationRatio,
      transactionData.selfTransferRatio,
      Math.log1p(transactionData.maxSingleValue / (transactionData.valueMean + 1e-8)),
      transactionData.velocityScore // Value transferred per time unit
    ];


    return features;
  }


  normalizeFeatures(features) {
    // Z-score normalization
    return features.map(f => (f - this.scaler.mean) / this.scaler.std);
  }


  async detectAnomaly(transactionData) {
    if (!this.model) await this.initialize();


    const features = this.extractFeatures(transactionData);
    const normalized = this.normalizeFeatures(features);
    
    const input = tf.tensor2d([normalized]);
    const reconstructed = this.model.predict(input);
    const reconstructionError = tf.mean(tf.square(tf.sub(input, reconstructed)));
    
    const errorValue = await reconstructionError.data();
    
    // Clean up tensors
    input.dispose();
    reconstructed.dispose();
    reconstructionError.dispose();


    // Dynamic threshold based on historical errors
    const threshold = this.calculateDynamicThreshold();
    
    return {
      isAnomaly: errorValue[0] > threshold,
      anomalyScore: errorValue[0],
      threshold,
      features: {
        raw: features,
        normalized
      }
    };
  }


  calculateDynamicThreshold() {
    if (this.featureBuffer.length < CONFIG.ANOMALY_DETECTION.MIN_SAMPLES) {
      return 0.5; // Default threshold
    }


    const errors = this.featureBuffer.map(f => f.error);
    const mean = errors.reduce((a, b) => a + b) / errors.length;
    const std = Math.sqrt(
      errors.reduce((sum, e) => sum + Math.pow(e - mean, 2), 0) / errors.length
    );


    return mean + (CONFIG.ANOMALY_DETECTION.STD_THRESHOLD * std);
  }


  async updateModel(features, label) {
    // Online learning capability
    this.featureBuffer.push({ features, label, error: 0 });
    
    if (this.featureBuffer.length >= CONFIG.ANOMALY_DETECTION.WINDOW_SIZE) {
      // Retrain with new data
      const trainingData = this.prepareTrainingData();
      await this.model.fit(trainingData.inputs, trainingData.outputs, {
        epochs: 10,
        batchSize: 32,
        verbose: 0
      });
      
      // Keep only recent data
      this.featureBuffer = this.featureBuffer.slice(-50);
    }
  }


  prepareTrainingData() {
    const normalSamples = this.featureBuffer.filter(s => !s.label);
    const inputs = normalSamples.map(s => s.features.normalized);
    
    return {
      inputs: tf.tensor2d(inputs),
      outputs: tf.tensor2d(inputs) // Autoencoder reconstructs input
    };
  }
}


// ============================================
// ENHANCED ENTROPY ANALYZER
// ============================================


class EnhancedEntropyAnalyzer {
  constructor() {
    this.timeWindows = [3600, 86400, 604800, 2592000]; // Added 30-day window
    this.anomalyDetector = new AnomalyDetector();
  }


  async analyzeWithML(walletData) {
    // Original entropy calculations
    const basicEntropy = this.computeCompositeEntropy(walletData);
    
    // ML-based anomaly detection
    const anomalyResult = await this.anomalyDetector.detectAnomaly({
      valueStd: this.calculateStd(walletData.values),
      valueMean: this.calculateMean(walletData.values),
      timeIntervalStd: this.calculateIntervalStd(walletData.timestamps),
      timeIntervalMean: this.calculateIntervalMean(walletData.timestamps),
      gasStd: this.calculateStd(walletData.gasPrices),
      gasMean: this.calculateMean(walletData.gasPrices),
      uniqueProtocolRatio: this.calculateUniqueRatio(walletData.protocols),
      nightTransactionRatio: this.calculateNightRatio(walletData.timestamps),
      failedTransactionRatio: walletData.failedRatio || 0,
      contractCreationRatio: walletData.contractCreationRatio || 0,
      selfTransferRatio: walletData.selfTransferRatio || 0,
      maxSingleValue: Math.max(...walletData.values),
      velocityScore: this.calculateVelocityScore(walletData)
    });


    return {
      ...basicEntropy,
      anomaly: anomalyResult,
      mlEnhancedScore: this.combineEntropyWithML(basicEntropy.composite, anomalyResult.anomalyScore)
    };
  }


  calculateVelocityScore(walletData) {
    if (walletData.timestamps.length < 2) return 0;
    
    const timeSpan = walletData.timestamps[walletData.timestamps.length - 1] - walletData.timestamps[0];
    const totalValue = walletData.values.reduce((a, b) => a + b, 0);
    
    return totalValue / (timeSpan / 86400); // Value per day
  }


  combineEntropyWithML(entropyScore, anomalyScore) {
    // Weighted combination with emphasis on ML detection for edge cases
    const normalizedAnomaly = Math.min(anomalyScore * 2, 1); // Scale anomaly score
    return entropyScore * 0.7 + normalizedAnomaly * 0.3;
  }


  // Additional statistical calculations
  calculateStd(values) {
    if (values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }


  calculateMean(values) {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }


  calculateIntervalStd(timestamps) {
    if (timestamps.length < 2) return 0;
    const intervals = [];
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i-1]);
    }
    return this.calculateStd(intervals);
  }


  calculateIntervalMean(timestamps) {
    if (timestamps.length < 2) return 0;
    const intervals = [];
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i-1]);
    }
    return this.calculateMean(intervals);
  }


  calculateUniqueRatio(protocols) {
    if (protocols.length === 0) return 0;
    const unique = new Set(protocols.map(p => p.protocol)).size;
    return unique / protocols.length;
  }


  calculateNightRatio(timestamps) {
    if (timestamps.length === 0) return 0;
    const nightTxs = timestamps.filter(ts => {
      const hour = new Date(ts * 1000).getUTCHours();
      return hour >= 0 && hour < 6; // UTC 0-6 AM
    });
    return nightTxs.length / timestamps.length;
  }


  // Include all original entropy methods from base class...
  calculateShannon(distribution) {
    const normalized = this.normalizeDistribution(distribution);
    return -normalized.reduce((sum, p) => {
      return sum + (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
  }


  calculateTemporalEntropy(timestamps) {
    if (timestamps.length < 2) return 0;
    
    const intervals = [];
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i-1]);
    }
    
    const bins = this.createLogBins(Math.min(...intervals), Math.max(...intervals), 20);
    const distribution = this.binData(intervals, bins);
    
    return this.calculateShannon(distribution);
  }


  calculateValueEntropy(values) {
    if (values.length === 0) return 0;
    
    const cleaned = this.removeOutliers(values);
    if (cleaned.length === 0) return 0;
    
    const minVal = Math.min(...cleaned);
    const maxVal = Math.max(...cleaned);
    const bins = this.createLogBins(minVal, maxVal, 15);
    const distribution = this.binData(cleaned, bins);
    
    return this.calculateShannon(distribution);
  }


  calculateProtocolEntropy(interactions) {
    const protocolCounts = {};
    interactions.forEach(int => {
      const protocolName = int.protocol || 'Unknown';
      protocolCounts[protocolName] = (protocolCounts[protocolName] || 0) + 1;
    });
    
    const distribution = Object.values(protocolCounts);
    return this.calculateShannon(distribution);
  }


  calculateGasEntropy(gasPrices) {
    if (gasPrices.length === 0) return 0;
    const bins = this.createLinearBins(Math.min(...gasPrices), Math.max(...gasPrices), 10);
    const distribution = this.binData(gasPrices, bins);
    return this.calculateShannon(distribution);
  }


  computeCompositeEntropy(walletData) {
    const weights = {
      temporal: 0.3,
      value: 0.25,
      protocol: 0.25,
      gas: 0.2
    };
    
    const entropies = {
      temporal: this.calculateTemporalEntropy(walletData.timestamps),
      value: this.calculateValueEntropy(walletData.values),
      protocol: this.calculateProtocolEntropy(walletData.protocols),
      gas: this.calculateGasEntropy(walletData.gasPrices)
    };
    
    const composite = Object.entries(entropies).reduce((sum, [key, value]) => {
      return sum + (weights[key] * value);
    }, 0);
    
    const stability = this.checkLyapunovStability(walletData.historicalEntropies);
    
    return {
      composite,
      components: entropies,
      stability,
      interpretation: this.interpretEntropy(composite, stability)
    };
  }


  checkLyapunovStability(timeSeries) {
    if (!timeSeries || timeSeries.length < 10) return { stable: true, exponent: 0, interpretation: 'stable' };
    
    const differences = [];
    for (let i = 1; i < timeSeries.length; i++) {
      differences.push(Math.abs(timeSeries[i] - timeSeries[i-1]));
    }
    
    const avgDivergence = differences.reduce((a, b) => a + b) / differences.length;
    if (avgDivergence <= 0) return { stable: true, exponent: -Infinity, interpretation: 'stable' };
    const lyapunovExponent = Math.log(avgDivergence);
    
    return {
      stable: lyapunovExponent < 0.1,
      exponent: lyapunovExponent,
      interpretation: lyapunovExponent > 0.5 ? 'chaotic' : 
                     lyapunovExponent > 0.1 ? 'unstable' : 'stable'
    };
  }


  normalizeDistribution(distribution) {
    const sum = distribution.reduce((a, b) => a + b, 0);
    if (sum === 0) return distribution.map(() => 0);
    return distribution.map(val => val / sum);
  }


  createLogBins(min, max, numBins) {
    if (min <= 0) min = 0.0001;
    const logMin = Math.log10(min);
    const logMax = Math.log10(max + 1);
    const step = (logMax - logMin) / numBins;
    
    return Array.from({length: numBins + 1}, (_, i) => 
      Math.pow(10, logMin + i * step)
    );
  }


  createLinearBins(min, max, numBins) {
    if (min === max) return [min, max];
    const step = (max - min) / numBins;
    return Array.from({length: numBins + 1}, (_, i) => min + i * step);
  }


  binData(data, bins) {
    const counts = new Array(bins.length - 1).fill(0);
    data.forEach(val => {
      for (let i = 0; i < bins.length - 1; i++) {
        if (val >= bins[i] && val < bins[i+1]) {
          counts[i]++;
          return;
        }
      }
      if(val === bins[bins.length-1]) {
        counts[counts.length-1]++;
      }
    });
    return counts;
  }


  removeOutliers(data) {
    if (data.length < 4) return data;
    const sorted = [...data].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lower = q1 - 1.5 * iqr;
    const upper = q3 + 1.5 * iqr;
    
    return data.filter(val => val >= lower && val <= upper);
  }


  interpretEntropy(entropy, stability) {
    if (!stability.stable) {
      return `UNSTABLE: Rapidly changing behavior patterns detected (${stability.interpretation})`;
    }
    
    if (entropy < 1.5) return 'LOW: Highly predictable, regular behavior';
    if (entropy < 2.5) return 'MODERATE: Some behavioral diversity';
    if (entropy < 3.5) return 'HIGH: Complex and diverse interaction patterns';
    return 'VERY HIGH: Potentially chaotic or obfuscated behavior';
  }
}


// ============================================
//COMPLIANCE MODULE
// ============================================


class ComplianceModule {
  constructor(dataSourceManager) {
    this.dataSourceManager = dataSourceManager;
    this.reportQueue = [];
    this.reportBatchSize = 100;
    this.reportInterval = 300000; // 5 minutes
    this.startReportingLoop();
  }


  async checkCompliance(address, riskScore) {
    const compliance = {
      ofacSanctioned: await this.dataSourceManager.checkOFACSanctions(address),
      requiresReporting: false,
      reportingReasons: [],
      complianceScore: 100
    };


    // OFAC check
    if (compliance.ofacSanctioned) {
      compliance.requiresReporting = true;
      compliance.reportingReasons.push('OFAC Sanctioned Entity');
      compliance.complianceScore = 0;
    }


    // High risk threshold
    if (riskScore > CONFIG.RISK_THRESHOLDS.HIGH) {
      compliance.requiresReporting = true;
      compliance.reportingReasons.push(`High Risk Score: ${riskScore.toFixed(2)}`);
      compliance.complianceScore = Math.max(0, compliance.complianceScore - riskScore);
    }


    // Suspicious patterns
    if (riskScore > CONFIG.RISK_THRESHOLDS.MODERATE) {
      const patterns = await this.checkSuspiciousPatterns(address);
      if (patterns.length > 0) {
        compliance.requiresReporting = true;
        compliance.reportingReasons.push(...patterns);
        compliance.complianceScore -= patterns.length * 10;
      }
    }


    return compliance;
  }


  async checkSuspiciousPatterns(address) {
    const patterns = [];
    
    // This would connect to more sophisticated pattern matching
    // For now, basic checks
    const history = await this.dataSourceManager.getEnhancedTransactionHistory(address, 100);
    
    // Rapid fund movement
    if (this.detectRapidMovement(history)) {
      patterns.push('Rapid fund movement detected');
    }


    // Layering behavior
    if (this.detectLayering(history)) {
      patterns.push('Potential layering behavior');
    }


    // Round amount transfers
    if (this.detectRoundAmounts(history)) {
      patterns.push('Suspicious round amount transfers');
    }


    return patterns;
  }


  detectRapidMovement(history) {
    // Funds received and sent within 1 hour
    const rapidTransfers = history.filter((tx, idx) => {
      if (idx === 0) return false;
      const timeDiff = tx.timestamp - history[idx - 1].timestamp;
      return timeDiff < 3600 && tx.value > 0;
    });
    
    return rapidTransfers.length > history.length * 0.3;
  }


  detectLayering(history) {
    // Multiple small transfers to different addresses
    const outgoing = history.filter(tx => tx.from === tx.address);
    const uniqueRecipients = new Set(outgoing.map(tx => tx.to)).size;
    
    return uniqueRecipients > 10 && outgoing.length > 20;
  }


  detectRoundAmounts(history) {
    // Check for suspicious round numbers
    const roundAmounts = history.filter(tx => {
      const value = parseFloat(tx.value);
      return value > 0 && (value % 1 === 0 || value % 0.1 === 0);
    });
    
    return roundAmounts.length > history.length * 0.5;
  }


  async generateComplianceReport(address, riskAnalysis, compliance) {
    const report = {
      reportId: `RC-${Date.now()}-${address.slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      address,
      riskScore: riskAnalysis.overallRisk.score,
      complianceScore: compliance.complianceScore,
      flags: compliance.reportingReasons,
      details: {
        entropy: riskAnalysis.components.entropy,
        adjacency: riskAnalysis.components.adjacency,
        bayesian: riskAnalysis.components.bayesian,
        anomaly: riskAnalysis.components.entropy.anomaly
      },
      recommendations: riskAnalysis.recommendations,
      status: 'PENDING_REVIEW'
    };


    // Queue for batch reporting
    this.reportQueue.push(report);
    
    if (compliance.ofacSanctioned || riskAnalysis.overallRisk.score > CONFIG.RISK_THRESHOLDS.CRITICAL) {
      // Immediate reporting for critical cases
      await this.submitReports([report]);
    }


    return report;
  }


  startReportingLoop() {
    setInterval(async () => {
      if (this.reportQueue.length > 0) {
        const reports = this.reportQueue.splice(0, this.reportBatchSize);
        await this.submitReports(reports);
      }
    }, this.reportInterval);
  }


  async submitReports(reports) {
    try {
      // In production, this would submit to regulatory reporting system
      logger.info({ count: reports.length }, 'Submitting compliance reports');
      
      // Store in database for audit trail
      await this.dataSourceManager.redis.lpush(
        'compliance:reports',
        ...reports.map(r => JSON.stringify(r))
      );
      
      // Trigger alerts for high-priority cases
      const critical = reports.filter(r => r.riskScore > CONFIG.RISK_THRESHOLDS.CRITICAL);
      if (critical.length > 0) {
        await this.sendAlerts(critical);
      }
    } catch (error) {
      logger.error({ error, reportCount: reports.length }, 'Failed to submit compliance reports');
    }
  }


  async sendAlerts(criticalReports) {
    // Integration with alerting system (Slack, email, etc.)
    logger.warn({ addresses: criticalReports.map(r => r.address) }, 'Critical risk alerts triggered');
  }
}


// ============================================
// ALERT SYSTEM
// ============================================


class AlertSystem {
  constructor() {
    this.subscribers = new Map();
    this.alertHistory = [];
    this.alertThrottles = new Map();
  }


  subscribe(address, config) {
    this.subscribers.set(address, {
      ...config,
      subscriptionId: `SUB-${Date.now()}-${address.slice(2, 8)}`,
      createdAt: Date.now()
    });
    
    logger.info({ address, config }, 'Alert subscription created');
    return this.subscribers.get(address).subscriptionId;
  }


  unsubscribe(address) {
    const removed = this.subscribers.delete(address);
    if (removed) {
      logger.info({ address }, 'Alert subscription removed');
    }
    return removed;
  }


  async processRiskChange(address, previousRisk, currentRisk) {
    const subscription = this.subscribers.get(address);
    if (!subscription) return;


    const riskDelta = currentRisk.score - previousRisk.score;
    const shouldAlert = this.shouldTriggerAlert(subscription, riskDelta, currentRisk);


    if (shouldAlert && !this.isThrottled(address)) {
      await this.sendAlert(address, {
        type: 'RISK_CHANGE',
        severity: this.calculateSeverity(currentRisk.score),
        previousScore: previousRisk.score,
        currentScore: currentRisk