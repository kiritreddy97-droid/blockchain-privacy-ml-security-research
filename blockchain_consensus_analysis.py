"""
Blockchain Consensus & Privacy Security Research Suite
=======================================================
Author: Kirit Reddy Daida — Graduate Researcher, Cleveland State University
Advisor: Prof. Iftikhar U. Sikder
Description:
    Implementation and comparative analysis of blockchain consensus algorithms
    (PoW, PoS, DPoS, PBFT), ML privacy attack simulations (membership inference,
    data poisoning, model inversion), and adversarial defense evaluation.
    Supports the published academic paper: 'Future of Brain-Computer Interfaces'
    and two survey papers on blockchain security and ML privacy.

Usage:
    python blockchain_consensus_analysis.py --mode all --output research/
    python blockchain_consensus_analysis.py --mode consensus
    python blockchain_consensus_analysis.py --mode ml_privacy
    python blockchain_consensus_analysis.py --mode bci

Requirements:
    pip install numpy pandas scipy matplotlib seaborn hashlib-plus
"""

import os
import sys
import time
import json
import hashlib
import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# MODULE 1: BLOCKCHAIN CONSENSUS ALGORITHM SIMULATIONS
# =============================================================================

@dataclass
class Block:
    index: int
    data: str
    previous_hash: str
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    hash: str = field(default="")

    def compute_hash(self) -> str:
        content = f"{self.index}{self.data}{self.previous_hash}{self.timestamp}{self.nonce}"
        return hashlib.sha256(content.encode()).hexdigest()

    def finalize(self) -> None:
        self.hash = self.compute_hash()


class ConsensusAlgorithm(ABC):
    """Abstract base for consensus algorithm implementations."""
    name: str = "Base"

    @abstractmethod
    def add_block(self, chain: List[Block], data: str) -> Tuple[Block, Dict]:
        """Add a block and return metrics."""
        pass

    def verify_chain(self, chain: List[Block]) -> bool:
        for i in range(1, len(chain)):
            if chain[i].previous_hash != chain[i - 1].hash:
                return False
            if chain[i].hash != chain[i].compute_hash():
                return False
        return True


class ProofOfWork(ConsensusAlgorithm):
    """Proof of Work — SHA-256 hash puzzle."""
    name = "PoW"

    def __init__(self, difficulty: int = 3) -> None:
        self.difficulty = difficulty
        self.target = "0" * difficulty

    def add_block(self, chain: List[Block], data: str) -> Tuple[Block, Dict]:
        prev = chain[-1]
        block = Block(index=len(chain), data=data, previous_hash=prev.hash)
        t0 = time.perf_counter()
        while True:
            block.hash = block.compute_hash()
            if block.hash.startswith(self.target):
                break
            block.nonce += 1
        elapsed = time.perf_counter() - t0
        metrics = {"algorithm": self.name, "nonce": block.nonce, "time_s": round(elapsed, 6),
                   "energy_cost": "High", "finality": "Probabilistic", "decentralisation": "High"}
        return block, metrics


class ProofOfStake(ConsensusAlgorithm):
    """Proof of Stake — validator selected by stake weight."""
    name = "PoS"

    def __init__(self, validators: Dict[str, float]) -> None:
        self.validators = validators
        self.rng = np.random.default_rng(42)

    def select_validator(self) -> str:
        names = list(self.validators.keys())
        stakes = np.array(list(self.validators.values()))
        probs = stakes / stakes.sum()
        return self.rng.choice(names, p=probs)

    def add_block(self, chain: List[Block], data: str) -> Tuple[Block, Dict]:
        prev = chain[-1]
        validator = self.select_validator()
        block = Block(index=len(chain), data=data, previous_hash=prev.hash)
        t0 = time.perf_counter()
        block.finalize()
        elapsed = time.perf_counter() - t0
        metrics = {"algorithm": self.name, "validator": validator, "time_s": round(elapsed, 6),
                   "energy_cost": "Low", "finality": "Probabilistic", "decentralisation": "Medium"}
        return block, metrics


class DelegatedProofOfStake(ConsensusAlgorithm):
    """DPoS — token holders elect witnesses who produce blocks."""
    name = "DPoS"

    def __init__(self, witnesses: List[str]) -> None:
        self.witnesses = witnesses
        self._idx = 0

    def add_block(self, chain: List[Block], data: str) -> Tuple[Block, Dict]:
        prev = chain[-1]
        witness = self.witnesses[self._idx % len(self.witnesses)]
        self._idx += 1
        block = Block(index=len(chain), data=data, previous_hash=prev.hash)
        t0 = time.perf_counter()
        block.finalize()
        elapsed = time.perf_counter() - t0
        metrics = {"algorithm": self.name, "witness": witness, "time_s": round(elapsed, 6),
                   "energy_cost": "Very Low", "finality": "Near-Instant", "decentralisation": "Low"}
        return block, metrics


class PBFT(ConsensusAlgorithm):
    """Practical Byzantine Fault Tolerance — 3-phase commit."""
    name = "PBFT"

    def __init__(self, n_nodes: int = 7) -> None:
        self.n_nodes = n_nodes
        self.faulty_tolerance = (n_nodes - 1) // 3
        self.rng = np.random.default_rng(99)

    def _simulate_phases(self) -> Dict:
        phases = {"pre_prepare": 1, "prepare": 2 * self.faulty_tolerance + 1, "commit": 2 * self.faulty_tolerance + 1}
        delays = {phase: round(self.rng.normal(0.002, 0.0005), 6) for phase in phases}
        return phases, delays

    def add_block(self, chain: List[Block], data: str) -> Tuple[Block, Dict]:
        prev = chain[-1]
        block = Block(index=len(chain), data=data, previous_hash=prev.hash)
        t0 = time.perf_counter()
        phases, delays = self._simulate_phases()
        block.finalize()
        elapsed = time.perf_counter() - t0 + sum(delays.values())
        metrics = {"algorithm": self.name, "nodes": self.n_nodes, "fault_tolerance": self.faulty_tolerance,
                   "time_s": round(elapsed, 6), "energy_cost": "Low",
                   "finality": "Deterministic", "decentralisation": "Medium"}
        return block, metrics


def benchmark_consensus(n_blocks: int = 20) -> pd.DataFrame:
    """Run all consensus algorithms and compare performance."""
    validators = {f"val_{i}": np.random.uniform(100, 10000) for i in range(10)}
    witnesses = [f"witness_{i}" for i in range(5)]
    algorithms = [
        ProofOfWork(difficulty=3),
        ProofOfStake(validators),
        DelegatedProofOfStake(witnesses),
        PBFT(n_nodes=7),
    ]
    results = []
    for algo in algorithms:
        genesis = Block(0, "Genesis", "0")
        genesis.finalize()
        chain = [genesis]
        for i in range(n_blocks):
            block, metrics = algo.add_block(chain, f"Tx-{i}: Alice→Bob: {np.random.uniform(0.1,10):.2f} BTC")
            chain.append(block)
            metrics["block_index"] = i + 1
            metrics["valid_chain"] = algo.verify_chain(chain)
            results.append(metrics)
        log.info("%s: %d blocks mined. Chain valid: %s", algo.name, n_blocks, algo.verify_chain(chain))
    df = pd.DataFrame(results)
    return df


# =============================================================================
# MODULE 2: ML PRIVACY ATTACKS SIMULATION
# =============================================================================

class MLPrivacyAttacks:
    """Simulates common ML privacy attacks and evaluates defenses."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def _generate_synthetic_model_outputs(self, n_train: int = 500, n_test: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate confidence vectors from a trained classification model."""
        n_classes = 10
        # Training members: higher confidence (model memorises)
        train_conf = self.rng.dirichlet(np.ones(n_classes) * 0.5, size=n_train)
        train_conf = np.sort(train_conf, axis=1)[:, ::-1]  # sort desc
        train_labels = np.ones(n_train, dtype=int)         # 1 = member

        # Non-members: lower, more uniform confidence
        test_conf = self.rng.dirichlet(np.ones(n_classes) * 2.0, size=n_test)
        test_conf = np.sort(test_conf, axis=1)[:, ::-1]
        test_labels = np.zeros(n_test, dtype=int)          # 0 = non-member

        return train_conf, test_conf, train_labels, test_labels

    # ── Attack 1: Membership Inference ─────────────────────────────────────
    def membership_inference_attack(self) -> Dict[str, Any]:
        """Threshold-based membership inference attack (Shokri et al. 2017)."""
        train_conf, test_conf, train_labels, test_labels = self._generate_synthetic_model_outputs()
        all_conf = np.vstack([train_conf, test_conf])
        all_labels = np.concatenate([train_labels, test_labels])
        max_conf = all_conf[:, 0]  # top-1 confidence as signal

        # Threshold sweep
        thresholds = np.linspace(0.1, 0.99, 50)
        best_acc, best_thresh = 0, 0.5
        for t in thresholds:
            preds = (max_conf >= t).astype(int)
            acc = (preds == all_labels).mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, t

        # True Positive Rate / False Positive Rate at best threshold
        preds_best = (max_conf >= best_thresh).astype(int)
        tp = ((preds_best == 1) & (all_labels == 1)).sum()
        fp = ((preds_best == 1) & (all_labels == 0)).sum()
        tn = ((preds_best == 0) & (all_labels == 0)).sum()
        fn = ((preds_best == 0) & (all_labels == 1)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        precision = tp / (tp + fp + 1e-9)

        # Defense: Differential Privacy noise reduces max_conf variance
        dp_noise = self.rng.normal(0, 0.05, size=len(max_conf))
        dp_conf = (max_conf + dp_noise).clip(0, 1)
        dp_preds = (dp_conf >= best_thresh).astype(int)
        dp_acc = (dp_preds == all_labels).mean()

        return {
            "attack": "Membership Inference",
            "best_threshold": round(best_thresh, 4),
            "attack_accuracy": round(best_acc, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "defense_dp_accuracy_drop": round(best_acc - dp_acc, 4),
            "defense_effective": (best_acc - dp_acc) > 0.03,
            "risk_level": "High" if best_acc > 0.65 else "Medium",
            "reference": "Shokri et al., 2017 — Membership Inference Attacks Against ML Models",
        }

    # ── Attack 2: Data Poisoning ────────────────────────────────────────────
    def data_poisoning_attack(self, poison_rate: float = 0.10) -> Dict[str, Any]:
        """Simulate label-flipping data poisoning on a training set."""
        n_samples = 1000
        # Original training data: binary classification
        X = self.rng.normal(0, 1, size=(n_samples, 20))
        y_clean = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Poison: flip labels for poison_rate% of samples
        n_poison = int(n_samples * poison_rate)
        poison_idx = self.rng.choice(n_samples, size=n_poison, replace=False)
        y_poisoned = y_clean.copy()
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

        # Simulate model accuracy with clean vs poisoned data (linear threshold proxy)
        def eval_accuracy(X_data, y_true, weights=None):
            if weights is None:
                weights = np.zeros(X_data.shape[1])
                weights[0], weights[1] = 1.0, 1.0
            logits = X_data @ weights
            preds = (logits > 0).astype(int)
            return (preds == y_true).mean()

        clean_acc = eval_accuracy(X, y_clean)
        poisoned_acc = eval_accuracy(X, y_clean)  # eval on clean labels
        # With poisoned training, weights deviate
        noise_weights = np.zeros(X.shape[1])
        noise_weights[0] = 1.0 - poison_rate * 1.5
        noise_weights[1] = 1.0 - poison_rate * 1.2
        degraded_acc = eval_accuracy(X, y_clean, noise_weights)

        # Defense: anomaly detection removes outliers
        mean, std = X.mean(axis=0), X.std(axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-9))
        clean_mask = z_scores.max(axis=1) < 3.0
        defended_acc = eval_accuracy(X[clean_mask], y_clean[clean_mask])

        return {
            "attack": "Data Poisoning (Label Flipping)",
            "poison_rate_pct": round(poison_rate * 100, 2),
            "n_poisoned_samples": n_poison,
            "clean_accuracy": round(clean_acc, 4),
            "degraded_accuracy": round(degraded_acc, 4),
            "accuracy_drop": round(clean_acc - degraded_acc, 4),
            "defended_accuracy": round(defended_acc, 4),
            "defense_method": "Z-score anomaly detection & sample filtering",
            "defense_effective": defended_acc > degraded_acc,
            "risk_level": "High" if poison_rate > 0.08 else "Medium",
            "reference": "Biggio et al., 2012 — Poisoning Attacks Against SVMs",
        }

    # ── Attack 3: Model Inversion ───────────────────────────────────────────
    def model_inversion_attack(self) -> Dict[str, Any]:
        """Simplified gradient-based model inversion simulation."""
        # Simulate reconstruction quality over iterations
        n_iterations = 100
        losses = []
        reconstruction_quality = []
        for i in range(n_iterations):
            loss = 10.0 * np.exp(-0.05 * i) + self.rng.normal(0, 0.3)
            losses.append(max(0, loss))
            quality = 1 - np.exp(-0.04 * i) + self.rng.normal(0, 0.02)
            reconstruction_quality.append(np.clip(quality, 0, 1))

        final_quality = reconstruction_quality[-1]
        # Defense: output perturbation reduces reconstruction quality
        defense_quality = final_quality * self.rng.uniform(0.40, 0.65)

        return {
            "attack": "Model Inversion",
            "iterations": n_iterations,
            "final_loss": round(losses[-1], 4),
            "reconstruction_quality": round(final_quality, 4),
            "defense_quality": round(defense_quality, 4),
            "quality_reduction_pct": round((1 - defense_quality / (final_quality + 1e-9)) * 100, 2),
            "defense_method": "Output perturbation + prediction confidence capping",
            "defense_effective": defense_quality < final_quality * 0.7,
            "risk_level": "High" if final_quality > 0.7 else "Medium",
            "reference": "Fredrikson et al., 2015 — Model Inversion Attacks",
        }

    def full_analysis(self) -> pd.DataFrame:
        """Run all attacks and return consolidated results."""
        results = []
        results.append(self.membership_inference_attack())
        results.append(self.data_poisoning_attack(0.10))
        results.append(self.data_poisoning_attack(0.20))
        results.append(self.model_inversion_attack())
        return pd.DataFrame(results)


# =============================================================================
# MODULE 3: BCI BLOCKCHAIN INTEGRATION MODEL
# =============================================================================

class BCIBlockchainModel:
    """
    Simulates the Brain-Computer Interface blockchain integration model
    co-authored in the published paper at Cleveland State University.
    Models neural data stream encryption, consent management, and
    tamper-evident audit logging via distributed ledger.
    """

    def __init__(self, n_patients: int = 50, seed: int = 11) -> None:
        self.n_patients = n_patients
        self.rng = np.random.default_rng(seed)
        self.consent_ledger: List[Dict] = []
        self.audit_chain: List[Block] = []
        self._init_chain()

    def _init_chain(self) -> None:
        genesis = Block(0, json.dumps({"event": "BCI Blockchain Genesis", "patients": self.n_patients}), "0")
        genesis.finalize()
        self.audit_chain.append(genesis)

    def record_consent(self, patient_id: str, data_types: List[str], granted: bool) -> Block:
        """Record patient consent for neural data usage on-chain."""
        consent_event = {
            "patient_id": patient_id,
            "data_types": data_types,
            "granted": granted,
            "timestamp": datetime.utcnow().isoformat(),
            "nonce": self.rng.integers(100000, 999999),
        }
        self.consent_ledger.append(consent_event)
        prev = self.audit_chain[-1]
        block = Block(len(self.audit_chain), json.dumps(consent_event), prev.hash)
        block.finalize()
        self.audit_chain.append(block)
        return block

    def simulate_neural_stream(self, patient_id: str, duration_s: float = 5.0) -> Dict:
        """Simulate EEG neural data stream with privacy-preserving hashing."""
        sample_rate = 256  # Hz
        n_samples = int(duration_s * sample_rate)
        channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]

        # Simulated raw EEG signal
        raw_data = {ch: self.rng.normal(0, 15, size=n_samples).tolist() for ch in channels}

        # Privacy: hash patient-identifiable fields, keep only aggregated features
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
        features = {
            "patient_hash": patient_hash,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "channels": channels,
            "mean_amplitude_uv": {ch: round(np.mean(raw_data[ch]), 4) for ch in channels},
            "std_amplitude_uv": {ch: round(np.std(raw_data[ch]), 4) for ch in channels},
            "band_power": {
                "delta_1_4hz": round(self.rng.uniform(10, 40), 4),
                "theta_4_8hz": round(self.rng.uniform(5, 25), 4),
                "alpha_8_13hz": round(self.rng.uniform(8, 35), 4),
                "beta_13_30hz": round(self.rng.uniform(3, 20), 4),
                "gamma_30_100hz": round(self.rng.uniform(1, 10), 4),
            },
        }

        # Record data access event on chain
        access_event = {"event": "neural_data_access", "patient_hash": patient_hash,
                        "timestamp": datetime.utcnow().isoformat()}
        prev = self.audit_chain[-1]
        block = Block(len(self.audit_chain), json.dumps(access_event), prev.hash)
        block.finalize()
        self.audit_chain.append(block)

        return features

    def generate_audit_report(self) -> pd.DataFrame:
        """Produce tamper-evident audit trail from the chain."""
        rows = []
        for block in self.audit_chain:
            rows.append({
                "block_index": block.index,
                "hash": block.hash[:20] + "…",
                "prev_hash": block.previous_hash[:20] + "…" if block.previous_hash != "0" else "GENESIS",
                "timestamp": block.timestamp,
                "data_preview": str(block.data)[:60],
            })
        return pd.DataFrame(rows)

    def integrity_check(self) -> Dict:
        """Verify chain integrity and detect tampering."""
        valid = True
        tamper_point = None
        for i in range(1, len(self.audit_chain)):
            b = self.audit_chain[i]
            if b.previous_hash != self.audit_chain[i - 1].hash:
                valid = False
                tamper_point = i
                break
            if b.hash != b.compute_hash():
                valid = False
                tamper_point = i
                break
        return {
            "chain_valid": valid,
            "total_blocks": len(self.audit_chain),
            "tamper_detected_at": tamper_point,
            "consent_records": len(self.consent_ledger),
        }


# =============================================================================
# REPORTING
# =============================================================================

class ResearchReporter:
    def __init__(self, output_dir: str = "research/") -> None:
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "charts").mkdir(exist_ok=True)

    def consensus_charts(self, df: pd.DataFrame) -> str:
        sns.set_theme(style="darkgrid")
        avg = df.groupby("algorithm")["time_s"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"PoW": "#e74c3c", "PoS": "#27ae60", "DPoS": "#3498db", "PBFT": "#f39c12"}
        bars = ax.bar(avg["algorithm"], avg["mean"] * 1000,
                      yerr=avg["std"] * 1000,
                      color=[colors.get(a, "#95a5a6") for a in avg["algorithm"]],
                      capsize=5, alpha=0.85)
        ax.set_ylabel("Avg Block Time (ms)")
        ax.set_title("Consensus Algorithm Performance Comparison", fontweight="bold")
        ax.set_xlabel("Algorithm")
        plt.tight_layout()
        p = str(self.out / "charts" / "consensus_benchmark.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p

    def privacy_charts(self, df: pd.DataFrame) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        attacks = df["attack"].tolist()
        risks = [1 if r == "High" else 0.5 for r in df["risk_level"].tolist()]
        defended = [1 if d else 0 for d in df["defense_effective"].tolist()]
        x = range(len(attacks))
        ax.bar(x, risks, label="Risk Level", color="#e74c3c", alpha=0.7, width=0.4)
        ax.bar([i + 0.4 for i in x], defended, label="Defense Effective", color="#27ae60", alpha=0.7, width=0.4)
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels([a[:25] for a in attacks], rotation=15, ha="right")
        ax.set_ylabel("Score (1=High/Effective)")
        ax.set_title("ML Privacy Attacks Risk vs Defense Effectiveness", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        p = str(self.out / "charts" / "ml_privacy_analysis.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return p

    def export_results(self, consensus_df: pd.DataFrame, privacy_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
        path = str(self.out / f"research_results_{date.today()}.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            consensus_df.to_excel(writer, sheet_name="Consensus Benchmark", index=False)
            privacy_df.to_excel(writer, sheet_name="ML Privacy Attacks", index=False)
            audit_df.to_excel(writer, sheet_name="BCI Audit Chain", index=False)
        log.info("Research results → %s", path)
        return path


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blockchain & ML Privacy Research Suite")
    p.add_argument("--mode", choices=["all", "consensus", "ml_privacy", "bci"], default="all")
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--patients", type=int, default=50)
    p.add_argument("--output", default="research/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reporter = ResearchReporter(args.output)

    consensus_df = pd.DataFrame()
    privacy_df = pd.DataFrame()
    audit_df = pd.DataFrame()

    if args.mode in ("all", "consensus"):
        log.info("Running consensus algorithm benchmarks (%d blocks each) …", args.blocks)
        consensus_df = benchmark_consensus(args.blocks)
        avg_times = consensus_df.groupby("algorithm")["time_s"].mean()
        print("
" + "="*55)
        print("  Consensus Algorithm Benchmark Results")
        print("="*55)
        for algo, t in avg_times.items():
            print(f"  {algo:<8} Avg Block Time: {t*1000:.3f} ms")
        print("="*55)
        reporter.consensus_charts(consensus_df)

    if args.mode in ("all", "ml_privacy"):
        log.info("Running ML privacy attack simulations …")
        attacks = MLPrivacyAttacks()
        privacy_df = attacks.full_analysis()
        print("
📌 ML Privacy Attack Results:")
        for _, row in privacy_df.iterrows():
            print(f"  [{row['risk_level']:6}] {row['attack'][:40]:<40} Defense: {'✅' if row['defense_effective'] else '❌'}")
        reporter.privacy_charts(privacy_df)

    if args.mode in ("all", "bci"):
        log.info("Simulating BCI blockchain integration for %d patients …", args.patients)
        bci = BCIBlockchainModel(args.patients)
        for i in range(min(args.patients, 10)):
            pid = f"PAT{str(i).zfill(4)}"
            bci.record_consent(pid, ["EEG", "EMG"], granted=True)
            bci.simulate_neural_stream(pid)
        integrity = bci.integrity_check()
        audit_df = bci.generate_audit_report()
        print("
🧠 BCI Blockchain Integrity Check:")
        for k, v in integrity.items():
            print(f"  {k:<30} {v}")

    if args.mode == "all":
        reporter.export_results(consensus_df, privacy_df, audit_df)
    log.info("Research pipeline complete. Output: %s", args.output)


if __name__ == "__main__":
    main()
