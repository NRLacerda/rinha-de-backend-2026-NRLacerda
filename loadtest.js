import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 10 },
    { duration: "30s", target: 50 },
    { duration: "20s", target: 100 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(99)<1000"],
    http_req_failed: ["rate<0.01"],
  },
};

export function setup() {
  const maxWait = 30 * 60; // 30 minutes
  let elapsed = 0;
  while (elapsed < maxWait) {
    const res = http.get("http://localhost:9999/ready");
    if (res.status === 200) {
      console.log(`API ready after ${elapsed}s`);
      return;
    }
    console.log(`Waiting for API... (${elapsed}s elapsed)`);
    sleep(5);
    elapsed += 5;
  }
  throw new Error("API did not become ready in time");
}

const PAYLOADS = [
  {
    id: "tx-legit-1",
    transaction: { amount: 150.0, installments: 1, requested_at: "2024-06-10T14:30:00Z" },
    customer: { avg_amount: 200.0, tx_count_24h: 3, known_merchants: ["merchant-supermarket"] },
    merchant: { id: "merchant-supermarket", mcc: "5411", avg_amount: 120.0 },
    terminal: { is_online: false, card_present: true, km_from_home: 2.5 },
    last_transaction: { timestamp: "2024-06-10T10:00:00Z", km_from_current: 1.2 },
  },
  {
    id: "tx-fraud-1",
    transaction: { amount: 9500.0, installments: 1, requested_at: "2024-06-10T03:15:00Z" },
    customer: { avg_amount: 100.0, tx_count_24h: 18, known_merchants: [] },
    merchant: { id: "merchant-casino", mcc: "7995", avg_amount: 5000.0 },
    terminal: { is_online: true, card_present: false, km_from_home: 800.0 },
    last_transaction: { timestamp: "2024-06-10T03:10:00Z", km_from_current: 750.0 },
  },
  {
    id: "tx-no-last",
    transaction: { amount: 300.0, installments: 3, requested_at: "2024-06-10T09:00:00Z" },
    customer: { avg_amount: 500.0, tx_count_24h: 1, known_merchants: ["merchant-electronics"] },
    merchant: { id: "merchant-electronics", mcc: "5944", avg_amount: 800.0 },
    terminal: { is_online: true, card_present: false, km_from_home: 50.0 },
    last_transaction: null,
  },
  {
    id: "tx-mid-risk",
    transaction: { amount: 1200.0, installments: 6, requested_at: "2024-06-10T20:45:00Z" },
    customer: { avg_amount: 600.0, tx_count_24h: 8, known_merchants: ["merchant-airline"] },
    merchant: { id: "merchant-airline", mcc: "4511", avg_amount: 2000.0 },
    terminal: { is_online: true, card_present: false, km_from_home: 300.0 },
    last_transaction: { timestamp: "2024-06-09T20:00:00Z", km_from_current: 280.0 },
  },
];

const headers = { "Content-Type": "application/json" };

export default function () {
  const payload = PAYLOADS[Math.floor(Math.random() * PAYLOADS.length)];
  const res = http.post("http://localhost:9999/fraud-score", JSON.stringify(payload), { headers });

  check(res, {
    "status 200": (r) => r.status === 200,
    "has approved field": (r) => JSON.parse(r.body).approved !== undefined,
    "has fraud_score field": (r) => JSON.parse(r.body).fraud_score !== undefined,
  });
}
