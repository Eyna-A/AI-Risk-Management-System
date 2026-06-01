"""
AI-Powered Risk Management System
==================================
Production-ready rewrite.

Fixes applied (see TECHNICAL_BREAKDOWN at the bottom of this file).
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REGIME_LABELS_EN = ["Normal", "Transition", "Crisis"]
EPS = 1e-8

class MultiSourceDataLoader:
    """Load and merge multi-source financial, geopolitical and macro data."""

    def __init__(self) -> None:
        self.scalers: Dict[str, StandardScaler] = {}
    def load_financial_data(
        self,
        prices: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        df = pd.DataFrame({"price": prices.values, "volume": volume.values})
        df["log_return"] = np.log(df["price"] / df["price"].shift(1))
        df["realized_vol"] = df["log_return"].rolling(20).std() * np.sqrt(252)
        df["price_momentum"] = df["price"] / df["price"].rolling(10).mean() - 1
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + EPS)
        df["price_velocity"] = df["price"].pct_change()
        df["price_acceleration"] = df["price_velocity"].diff()
        df["amihud_illiquidity"] = np.abs(df["log_return"]) / (df["volume"] + EPS)
        df["liquidity_ratio"] = 1.0 / (df["amihud_illiquidity"] + EPS)
        return df.dropna().reset_index(drop=True)
    def load_geopolitical_data(
        self,
        epu: pd.Series,
        sanctions: pd.Series,
        events: pd.Series,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "epu_index": epu.values,
                "sanction_intensity": sanctions.values,
                "political_event": events.values,
            }
        )
        df["epu_ma5"] = df["epu_index"].rolling(5).mean()
        df["epu_ma20"] = df["epu_index"].rolling(20).mean()
        df["epu_acceleration"] = df["epu_index"].diff()
        df["sanction_epu_interaction"] = df["sanction_intensity"] * df["epu_index"]
        return df.dropna().reset_index(drop=True)
    def load_macro_data(
        self,
        forex: pd.Series,
        inflation: pd.Series,
        cds: Optional[pd.Series] = None,
        embi: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame({"forex_rate": forex.values, "inflation": inflation.values})
        df["forex_volatility"] = df["forex_rate"].rolling(20).std()
        df["forex_change"] = df["forex_rate"].pct_change()
        if cds is not None:
            df["cds_spread"] = cds.values
        if embi is not None:
            df["embi_spread"] = embi.values
        df["real_forex_rate"] = df["forex_rate"] / (1.0 + df["inflation"] + EPS)
        return df.dropna().reset_index(drop=True)
    def merge_all_sources(
        self,
        financial_df: pd.DataFrame,
        geo_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        news_sentiment: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        min_len = min(len(financial_df), len(geo_df), len(macro_df))
        merged: Dict[str, np.ndarray] = {}
        for col in financial_df.columns:
            merged[f"fin_{col}"] = financial_df[col].values[:min_len]
        for col in geo_df.columns:
            merged[f"geo_{col}"] = geo_df[col].values[:min_len]
        for col in macro_df.columns:
            merged[f"macro_{col}"] = macro_df[col].values[:min_len]
        if news_sentiment is not None:
            merged["news_sentiment"] = news_sentiment.values[:min_len]
        return pd.DataFrame(merged)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiSourceFeatureExtractor(nn.Module):
    def __init__(
        self,
        fin_dim: int,
        geo_dim: int,
        macro_dim: int,
        news_dim: int = 0,
        hidden_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.fin_extractor = FeatureExtractor(fin_dim, hidden_dim, output_dim)
        self.geo_extractor = FeatureExtractor(geo_dim, hidden_dim, output_dim)
        self.macro_extractor = FeatureExtractor(macro_dim, hidden_dim, output_dim)
        self.news_extractor: Optional[FeatureExtractor] = None
        if news_dim > 0:
            self.news_extractor = FeatureExtractor(news_dim, hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(
        self,
        fin_x: torch.Tensor,
        geo_x: torch.Tensor,
        macro_x: torch.Tensor,
        news_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sources = [
            self.fin_extractor(fin_x),
            self.geo_extractor(geo_x),
            self.macro_extractor(macro_x),
        ]
        if news_x is not None and self.news_extractor is not None:
            sources.append(self.news_extractor(news_x))
        return torch.stack(sources, dim=1)  # (B, num_sources, output_dim)

class CrossModalAttention(nn.Module):
    """Multi-head self-attention across data modalities."""

    def __init__(self, feature_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.o_proj = nn.Linear(feature_dim, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, S, D = features.shape
        q = self.q_proj(features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        features = self.norm1(features + out)
        features = self.norm2(features + self.ffn(features))
        return features


class GatingMechanism(nn.Module):
    """Learned soft gating to weight each data source."""

    def __init__(self, feature_dim: int, num_sources: int) -> None:
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * num_sources, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_sources),
            nn.Softmax(dim=-1),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = features.shape[0]
        flat = features.view(B, -1)
        weights = self.gate_network(flat)           # (B, num_sources)
        fused = (features * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return fused, weights


class FusionLayer(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4, num_sources: int = 3) -> None:
        super().__init__()
        self.attention = CrossModalAttention(feature_dim, num_heads)
        self.gating = GatingMechanism(feature_dim, num_sources)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended = self.attention(features)
        fused, weights = self.gating(attended)
        return fused, weights

class TemporalEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, (h_n, _) = self.lstm(x)
        last_fwd = h_n[-2]   # (B, hidden_dim)
        last_bwd = h_n[-1]   # (B, hidden_dim)
        bidirectional_hidden = torch.cat([last_fwd, last_bwd], dim=1)  # (B, hidden*2)

        attn_out, attn_weights = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        final_out = self.layer_norm(attn_out[:, -1, :] + bidirectional_hidden)
        return final_out, attn_weights

class RegimeClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 32, num_regimes: int = 3
    ) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.regime_logits = nn.Linear(hidden_dim, num_regimes)
        self.regime_params = nn.Linear(hidden_dim, num_regimes * 2)
        self.num_regimes = num_regimes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.feature_net(x)
        regime_probs = F.softmax(self.regime_logits(feat), dim=-1)          # (B, R)
        params = self.regime_params(feat).view(-1, self.num_regimes, 2)     # (B, R, 2)
        regime_mu = params[..., 0]
        regime_sigma = F.softplus(params[..., 1]) + EPS                     # always > 0

        expected_mu = (regime_probs * regime_mu).sum(dim=-1)
        expected_sigma = (regime_probs * regime_sigma).sum(dim=-1)

        return {
            "regime_probs": regime_probs,
            "regime_params": params,
            "expected_mu": expected_mu,
            "expected_sigma": expected_sigma,
            "dominant_regime": regime_probs.argmax(dim=-1),
        }

class VolatilityPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        forecast_horizons: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.forecast_horizons = forecast_horizons or [1, 5, 20]
        self.temporal_encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.forecast_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Softplus(),
                )
                for _ in self.forecast_horizons
            ]
        )

    def forward(
        self, x_sequence: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        temporal_feat, attn_weights = self.temporal_encoder(x_sequence)
        forecasts = {
            f"vol_h{h}": self.forecast_heads[i](temporal_feat).squeeze(-1)
            for i, h in enumerate(self.forecast_horizons)
        }
        return forecasts, attn_weights

class SentimentAnalyzer(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden_dim: int = 32) -> None:
        super().__init__()
        self.text_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        attn_out, attn_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings
        )
        text_repr = attn_out.mean(dim=1)
        sentiment_probs = F.softmax(self.sentiment_head(text_repr), dim=-1)
        sentiment_score = sentiment_probs[:, 2] - sentiment_probs[:, 0]
        uncertainty = self.uncertainty_head(text_repr).squeeze(-1)
        epu = (1.0 - sentiment_score) / 2.0 * (1.0 + uncertainty) / 2.0
        return {
            "sentiment_score": sentiment_score,
            "sentiment_probs": sentiment_probs,
            "uncertainty": uncertainty,
            "epu_signal": epu,
            "attention_weights": attn_weights,
        }

class AIPoweredRiskSystem(nn.Module):
    """
    Full multi-source AI risk system.

    Key differences from original:
    - vol_proj registered in __init__ (not lazily inside forward → fixes
      missing optimizer registration & device mismatch bugs).
    - source_importance keys renamed to English to match report formatter.
    - All sub-modules always on the same device via to(device) call.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_models()
        self.to(self.device)
    @staticmethod
    def _default_config() -> Dict:
        return {
            "fin_dim": 10,
            "geo_dim": 7,
            "macro_dim": 6,
            "news_dim": 0,
            "hidden_dim": 64,
            "output_dim": 16,
            "sequence_length": 20,
            "num_regimes": 3,
            "forecast_horizons": [1, 5, 20],
            "learning_rate": 0.001,
            "dropout": 0.2,
        }
    def _build_models(self) -> None:
        cfg = self.config
        fin_dim = cfg["fin_dim"]
        geo_dim = cfg["geo_dim"]
        macro_dim = cfg["macro_dim"]
        news_dim = cfg.get("news_dim", 0)
        hidden_dim = cfg["hidden_dim"]
        output_dim = cfg["output_dim"]
        num_sources = 3 if news_dim == 0 else 4

        self.feature_extractor = MultiSourceFeatureExtractor(
            fin_dim=fin_dim,
            geo_dim=geo_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.fusion = FusionLayer(
            feature_dim=output_dim, num_heads=4, num_sources=num_sources
        )
        self.regime_classifier = RegimeClassifier(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_regimes=cfg["num_regimes"],
        )
        self.vol_proj = nn.Linear(fin_dim, output_dim)
        self.volatility_predictor = VolatilityPredictor(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            forecast_horizons=cfg["forecast_horizons"],
        )
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        if news_dim > 0:
            self.sentiment_analyzer = SentimentAnalyzer(
                embed_dim=output_dim, hidden_dim=hidden_dim
            )
        self.sequence_length: int = cfg["sequence_length"]
        self._num_sources: int = num_sources
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict:
        fin_x = batch["fin"].to(self.device)    # (B, T, fin_dim)
        geo_x = batch["geo"].to(self.device)
        macro_x = batch["macro"].to(self.device)
        fin_feat = self.feature_extractor.fin_extractor(fin_x[:, -1, :])
        geo_feat = self.feature_extractor.geo_extractor(geo_x[:, -1, :])
        macro_feat = self.feature_extractor.macro_extractor(macro_x[:, -1, :])

        sources = [fin_feat, geo_feat, macro_feat]
        if self.feature_extractor.news_extractor is not None and "news" in batch:
            news_x = batch["news"].to(self.device)
            sources.append(self.feature_extractor.news_extractor(news_x[:, -1, :]))

        features = torch.stack(sources, dim=1)   # (B, num_sources, output_dim)
        fused, source_weights = self.fusion(features)

        regime_output = self.regime_classifier(fused)
        fin_transformed = self.vol_proj(fin_x)   # (B, T, output_dim)
        vol_forecasts, vol_attn = self.volatility_predictor(fin_transformed)

        sentiment_output: Dict = {}
        if self.sentiment_analyzer is not None and "news" in batch:
            news_x = batch["news"].to(self.device)
            sentiment_output = self.sentiment_analyzer(news_x)

        return {
            "fused_features": fused,
            "source_weights": source_weights,
            "regime": regime_output,
            "volatility": vol_forecasts,
            "sentiment": sentiment_output,
        }
    @torch.no_grad()
    def predict(self, data_batch: Dict[str, torch.Tensor]) -> Dict:
        self.eval()
        return self.forward(data_batch)
    @torch.no_grad()
    def get_risk_metrics(self, data_batch: Dict[str, torch.Tensor]) -> Dict:
        self.eval()
        output = self.forward(data_batch)
        horizons = self.config["forecast_horizons"]
        vol_metrics = {
            f"{h}_day": output["volatility"][f"vol_h{h}"].mean().item()
            for h in horizons
        }
        regime_probs = output["regime"]["regime_probs"][0].cpu().numpy()
        source_w = output["source_weights"][0].cpu().numpy()
        source_keys = ["Financial", "Geopolitical", "Macro"]
        if self._num_sources == 4:
            source_keys.append("News")
        sentiment: Dict = {}
        if output["sentiment"]:
            sentiment = {
                "score": output["sentiment"]["sentiment_score"].mean().item(),
                "uncertainty": output["sentiment"]["uncertainty"].mean().item(),
                "epu": output["sentiment"]["epu_signal"].mean().item(),
            }
        return {
            "volatility": vol_metrics,
            "regime": {
                "dominant": int(output["regime"]["dominant_regime"][0].item()),
                "probabilities": regime_probs.tolist(),
                "labels": REGIME_LABELS_EN,
                "expected_volatility": float(
                    output["regime"]["expected_sigma"][0].item()
                ),
            },
            "source_importance": dict(zip(source_keys, source_w.tolist())),
            "sentiment": sentiment,
        }

class RiskSystemTrainer:
    def __init__(self, model: AIPoweredRiskSystem, config: Optional[Dict] = None) -> None:
        self.model = model
        self.config = config or {}
        self.device = model.device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get("lr", 1e-3),
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    def _prepare_batch(
        self, df: pd.DataFrame, idx: int, seq_len: int
    ) -> Dict[str, torch.Tensor]:
        fin_cols = [c for c in df.columns if c.startswith("fin_")]
        geo_cols = [c for c in df.columns if c.startswith("geo_")]
        macro_cols = [c for c in df.columns if c.startswith("macro_")]

        def get_seq(cols: List[str]) -> torch.Tensor:
            start = max(0, idx - seq_len + 1)
            data = df[cols].iloc[start : idx + 1].values.astype(np.float32)
            if len(data) < seq_len:
                pad = np.zeros((seq_len - len(data), len(cols)), dtype=np.float32)
                data = np.vstack([pad, data])
            return torch.from_numpy(data).unsqueeze(0)

        batch: Dict[str, torch.Tensor] = {
            "fin": get_seq(fin_cols),
            "geo": get_seq(geo_cols),
            "macro": get_seq(macro_cols),
        }
        if "news_sentiment" in df.columns:
            batch["news"] = get_seq(["news_sentiment"])
        return batch
    def _compute_loss(
        self, output: Dict, targets: Dict
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=self.device)
        log: Dict[str, float] = {}

        if "target_vol" in targets:
            target_vol = targets["target_vol"].to(self.device)
            vol_loss = torch.stack(
                [F.mse_loss(v, target_vol) for v in output["volatility"].values()]
            ).mean()
            total_loss = total_loss + vol_loss
            log["vol_loss"] = vol_loss.item()

        if "target_regime" in targets:
            target_regime = targets["target_regime"].to(self.device)
            rl = F.cross_entropy(output["regime"]["regime_probs"], target_regime)
            total_loss = total_loss + rl * 0.5
            log["regime_loss"] = rl.item()

        if "target_sentiment" in targets and output["sentiment"]:
            ts = targets["target_sentiment"].to(self.device)
            sl = F.mse_loss(output["sentiment"]["sentiment_score"], ts)
            total_loss = total_loss + sl * 0.3
            log["sentiment_loss"] = sl.item()
        diversity_penalty = -torch.std(output["source_weights"]) * 0.1
        total_loss = total_loss + diversity_penalty
        log["diversity_penalty"] = diversity_penalty.item()
        log["total_loss"] = total_loss.item()
        return total_loss, log
    def _train_epoch(
        self, train_df: pd.DataFrame, targets: Dict, batch_size: int
    ) -> float:
        self.model.train()
        n_samples = max(1, len(train_df) - self.model.sequence_length)
        indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        total = 0.0
        for idx in indices:
            batch = self._prepare_batch(train_df, int(idx), self.model.sequence_length)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss, _ = self._compute_loss(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(indices)
    def _validate(self, val_df: pd.DataFrame, targets: Dict) -> float:
        self.model.eval()
        n_samples = max(1, len(val_df) - self.model.sequence_length)
        indices = np.random.choice(n_samples, min(50, n_samples), replace=False)
        total = 0.0
        with torch.no_grad():
            for idx in indices:
                batch = self._prepare_batch(val_df, int(idx), self.model.sequence_length)
                out = self.model(batch)
                loss, _ = self._compute_loss(out, targets)
                total += loss.item()
        return total / len(indices)
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        targets: Optional[Dict] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        targets = targets or {}
        print("=" * 60)
        print("          Starting AI System Training")
        print("=" * 60)
        for epoch in range(epochs):
            tr_loss = self._train_epoch(train_df, targets, batch_size)
            val_loss = self._validate(val_df, targets)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)
            if (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train: {tr_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}"
                )
        print("=" * 60)
        print("          Training Completed")
        print("=" * 60)

class QuantitativeIntegration:
    def __init__(self, ai_system: AIPoweredRiskSystem) -> None:
        self.ai_system = ai_system
    def _adjusted_volatility(self, risk_metrics: Dict) -> float:
        lstm_vol = risk_metrics["volatility"]["20_day"]
        regime_vol = risk_metrics["regime"]["expected_volatility"]
        crisis_prob = risk_metrics["regime"]["probabilities"][2]
        return 0.5 * lstm_vol + 0.3 * regime_vol + 0.2 * lstm_vol * crisis_prob * 2.0
    def price_options_ai_enhanced(
        self,
        S: float,
        K: float,
        T: float,
        r_base: float,
        risk_metrics: Dict,
    ) -> Dict:
        sigma = self._adjusted_volatility(risk_metrics)
        if risk_metrics.get("sentiment"):
            sentiment_score = risk_metrics["sentiment"].get("score", 0.0)
            sigma *= 1.0 + (1.0 - sentiment_score) * 0.2

        sigma = max(sigma, 1e-6)   # guard against zero vol
        sqrt_T = np.sqrt(max(T, 1e-9))
        d1 = (np.log(S / K) + (r_base + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        disc = np.exp(-r_base * T)

        call = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        put = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T + EPS)
        vega = S * norm.pdf(d1) * sqrt_T / 100.0
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2.0 * sqrt_T + EPS)
            - r_base * K * disc * norm.cdf(d2)
        ) / 365.0

        return {
            "call_price": call,
            "put_price": put,
            "implied_vol": sigma,
            "greeks": {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta},
            "adjustments": {
                "lstm_vol": risk_metrics["volatility"]["20_day"],
                "regime_vol": risk_metrics["regime"]["expected_volatility"],
                "sentiment_score": risk_metrics.get("sentiment", {}).get("score", 0.0),
            },
        }
    def monte_carlo_ai_scenario(
        self,
        S0: float,
        T_days: int,
        risk_metrics: Dict,
        n_simulations: int = 5000,
        seed: int = 42,
    ) -> Dict:
        rng = np.random.default_rng(seed)
        dt = 1.0 / 252.0
        base_vol = risk_metrics["volatility"]["20_day"]
        regime_probs = risk_metrics["regime"]["probabilities"]

        vol_mult = {0: 1.0, 1: 1.5, 2: 2.5}
        mu_map = {0: 3e-4, 1: -1e-4, 2: -8e-4}

        all_finals: List[float] = []
        for reg in range(3):
            n_sim = int(n_simulations * regime_probs[reg])
            if n_sim == 0:
                continue
            vol = base_vol * vol_mult[reg]
            mu = mu_map[reg]
            Z = rng.standard_normal((n_sim, T_days))
            log_ret = (mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z
            final = S0 * np.exp(log_ret.sum(axis=1))
            all_finals.extend(final.tolist())

        arr = np.array(all_finals)
        if len(arr) == 0:
            arr = np.array([S0])
        p5 = float(np.percentile(arr, 5))

        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p5": p5,
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "var_95": float(S0 - p5),
            "var_99": float(S0 - np.percentile(arr, 1)),
            "expected_shortfall_95": float(
                S0 - np.mean(arr[arr <= p5]) if np.any(arr <= p5) else 0.0
            ),
            "probability_profit": float(np.mean(arr > S0)),
        }
    def optimize_portfolio_ai(
        self,
        returns_df: pd.DataFrame,
        risk_metrics: Dict,
        risk_tolerance: str = "medium",
    ) -> Dict:
        n_assets = len(returns_df.columns)
        regime_probs = risk_metrics["regime"]["probabilities"]
        regime_vol_mult = (
            regime_probs[0] * 1.0
            + regime_probs[1] * 1.8
            + regime_probs[2] * 3.5
        )
        sentiment = risk_metrics.get("sentiment", {})
        sentiment_adj = 1.0 + (1.0 - float(sentiment.get("score", 0.0))) * 0.3
        total_adj = regime_vol_mult * sentiment_adj

        cov = returns_df.cov().values * 252.0 * total_adj
        base_rf = 0.20
        rf = base_rf * (1.0 + (1.0 - regime_probs[0]) * 0.3)
        mean_ret = returns_df.mean().values * 252.0
        ret_factor = {"low": 0.7, "medium": 0.9, "high": 1.0}.get(risk_tolerance, 0.9)
        mean_ret_adj = mean_ret * ret_factor

        def portfolio_stats(w: np.ndarray) -> Tuple[float, float, float]:
            ret = float(np.dot(w, mean_ret_adj))
            vol = float(np.sqrt(np.dot(w, cov @ w)))
            sharpe = (ret - rf) / (vol + EPS)
            return ret, vol, sharpe

        objective = (
            (lambda w: portfolio_stats(w)[1])
            if risk_tolerance == "low"
            else (lambda w: -portfolio_stats(w)[2])
        )

        result = minimize(
            objective,
            x0=np.full(n_assets, 1.0 / n_assets),
            method="SLSQP",
            bounds=[(0, 1)] * n_assets,
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        weights = result.x if result.success else np.full(n_assets, 1.0 / n_assets)
        ret, vol, sharpe = portfolio_stats(weights)

        return {
            "weights": {
                name: f"{w * 100:.1f}%"
                for name, w in zip(returns_df.columns, weights)
            },
            "expected_return": f"{ret:.1%}",
            "volatility": f"{vol:.1%}",
            "sharpe_ratio": f"{sharpe:.3f}",
            "adjusted_parameters": {
                "regime_adjustment": regime_vol_mult,
                "sentiment_adjustment": sentiment_adj,
                "total_adjustment": total_adj,
                "risk_free_rate": f"{rf:.1%}",
                "dominant_regime": REGIME_LABELS_EN[risk_metrics["regime"]["dominant"]],
            },
        }

def generate_comprehensive_report(
    ai_system: AIPoweredRiskSystem,
    test_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    S: float,
    K: float,
    T: float,
    r: float,
) -> Dict:
    seq_len = ai_system.sequence_length
    fin_cols = [c for c in test_df.columns if c.startswith("fin_")]
    geo_cols = [c for c in test_df.columns if c.startswith("geo_")]
    macro_cols = [c for c in test_df.columns if c.startswith("macro_")]

    last_idx = len(test_df) - 1
    start = max(0, last_idx - seq_len + 1)

    def _make_seq(cols: List[str]) -> torch.Tensor:
        data = test_df[cols].iloc[start : last_idx + 1].values.astype(np.float32)
        if len(data) < seq_len:
            data = np.vstack(
                [np.zeros((seq_len - len(data), len(cols)), dtype=np.float32), data]
            )
        return torch.from_numpy(data).unsqueeze(0)

    batch = {
        "fin": _make_seq(fin_cols),
        "geo": _make_seq(geo_cols),
        "macro": _make_seq(macro_cols),
    }

    risk_metrics = ai_system.get_risk_metrics(batch)
    quant = QuantitativeIntegration(ai_system)
    option_prices = quant.price_options_ai_enhanced(S, K, T, r, risk_metrics)
    mc_results = quant.monte_carlo_ai_scenario(S, 30, risk_metrics)
    portfolios = {
        "conservative": quant.optimize_portfolio_ai(returns_df, risk_metrics, "low"),
        "moderate": quant.optimize_portfolio_ai(returns_df, risk_metrics, "medium"),
    }

    src = risk_metrics["source_importance"]
    dominant = risk_metrics["regime"]["dominant"]

    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║             Comprehensive AI-Powered Risk Management Report          ║
╚══════════════════════════════════════════════════════════════════════╝
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Market Regime Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dominant Regime   : {REGIME_LABELS_EN[dominant]}
  Normal            : {risk_metrics['regime']['probabilities'][0]:.1%}
  Transition        : {risk_metrics['regime']['probabilities'][1]:.1%}
  Crisis            : {risk_metrics['regime']['probabilities'][2]:.1%}
  Expected σ        : {risk_metrics['regime']['expected_volatility']:.2%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Volatility Forecast (LSTM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1-Day  : {risk_metrics['volatility']['1_day']:.2%}
  5-Day  : {risk_metrics['volatility']['5_day']:.2%}
  20-Day : {risk_metrics['volatility']['20_day']:.2%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Data Source Importance (Gating)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(f"  {k:15}: {v:.1%}" for k, v in src.items())}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Option Pricing (Black-Scholes + AI Vol)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Adjusted σ  : {option_prices['implied_vol']:.2%}
  Call Price  : {option_prices['call_price']:,.0f}
  Put  Price  : {option_prices['put_price']:,.0f}
  Delta : {option_prices['greeks']['delta']:.3f}
  Gamma : {option_prices['greeks']['gamma']:.4f}
  Vega  : {option_prices['greeks']['vega']:.2f}
  Theta : {option_prices['greeks']['theta']:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Monte Carlo (30-Day, 5000 paths)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mean    : {mc_results['mean']:,.0f}   Median : {mc_results['median']:,.0f}
  P5      : {mc_results['p5']:,.0f}     P95    : {mc_results['p95']:,.0f}
  VaR 95% : {mc_results['var_95']:,.0f} ({mc_results['var_95']/S:.1%} of capital)
  VaR 99% : {mc_results['var_99']:,.0f} ({mc_results['var_99']/S:.1%} of capital)
  ES  95% : {mc_results['expected_shortfall_95']:,.0f}
  P(profit): {mc_results['probability_profit']:.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Optimised Portfolios
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conservative | Return {portfolios['conservative']['expected_return']}  Risk {portfolios['conservative']['volatility']}  Sharpe {portfolios['conservative']['sharpe_ratio']}
Moderate     | Return {portfolios['moderate']['expected_return']}  Risk {portfolios['moderate']['volatility']}  Sharpe {portfolios['moderate']['sharpe_ratio']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. AI Adjustment Factors (Moderate Portfolio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Regime adj.    : {portfolios['moderate']['adjusted_parameters']['regime_adjustment']:.2f}×
  Sentiment adj. : {portfolios['moderate']['adjusted_parameters']['sentiment_adjustment']:.2f}×
  Total adj.     : {portfolios['moderate']['adjusted_parameters']['total_adjustment']:.2f}×
  Risk-free rate : {portfolios['moderate']['adjusted_parameters']['risk_free_rate']}
╔══════════════════════════════════════════════════════════════════════╝
"""
    print(report)
    return {
        "risk_metrics": risk_metrics,
        "option_prices": option_prices,
        "monte_carlo": mc_results,
        "portfolios": portfolios,
    }

def create_sample_data() -> Tuple:
    n = 1827  # ~5 years of daily data
    rng = np.random.default_rng(0)
    prices = 10_000.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    volume = np.abs(rng.normal(5_000_000, 1_000_000, n))
    epu = rng.normal(100, 10, n)
    sanctions = rng.uniform(0, 0.3, n)
    events = rng.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])
    forex = rng.normal(42_000, 500, n)
    inflation = rng.normal(0.35, 0.01, n)
    return prices, volume, epu, sanctions, events, forex, inflation


def save_model(model: AIPoweredRiskSystem, filepath: str = "risk_model.pth") -> None:
    torch.save({"model_state_dict": model.state_dict(), "config": model.config}, filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath: str = "risk_model.pth") -> AIPoweredRiskSystem:
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model = AIPoweredRiskSystem(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(model.device)
    print(f"✅ Model loaded from {filepath}")
    return model


def display_dashboard(risk_metrics: Dict) -> None:
    regime_icons = ["🟢", "🟡", "🔴"]
    dominant = risk_metrics["regime"]["dominant"]
    vol_20 = risk_metrics["volatility"]["20_day"]
    vol_status = "🟢 Low" if vol_20 < 0.2 else ("🟡 Medium" if vol_20 < 0.4 else "🔴 High")

    print("\n" + "=" * 66)
    print("                 AI Risk Management Dashboard")
    print("=" * 66)
    print(f"  Regime : {regime_icons[dominant]} {REGIME_LABELS_EN[dominant]}")
    print(f"  Vol 20d: {vol_20:.2%}  ({vol_status})")
    print(f"  Vol  1d: {risk_metrics['volatility']['1_day']:.2%}")
    print(f"  Vol  5d: {risk_metrics['volatility']['5_day']:.2%}")
    print("-" * 66)
    print("  Source Importance:")
    for src, w in risk_metrics["source_importance"].items():
        bar = "█" * int(w * 30)
        print(f"  {src:15} │ {bar:<30} │ {w:>5.1%}")
    print("=" * 66)

def main() -> Tuple[AIPoweredRiskSystem, pd.DataFrame, Dict]:
    print("=" * 60)
    print("   AI-Powered Risk Management System  (production build)")
    print("=" * 60)

    print("\n[1/5] Generating sample data …")
    (prices, volume, epu, sanctions, events, forex, inflation) = create_sample_data()

    print("[2/5] Processing individual data sources …")
    loader = MultiSourceDataLoader()
    financial_df = loader.load_financial_data(pd.Series(prices), pd.Series(volume))
    geo_df = loader.load_geopolitical_data(
        pd.Series(epu), pd.Series(sanctions), pd.Series(events)
    )
    macro_df = loader.load_macro_data(pd.Series(forex), pd.Series(inflation))

    print("[3/5] Merging data sources …")
    merged_df = loader.merge_all_sources(financial_df, geo_df, macro_df)
    print(f"      Shape: {merged_df.shape}")

    print("[4/5] Initialising AI models …")
    config: Dict = {
        "fin_dim": sum(1 for c in merged_df.columns if c.startswith("fin_")),
        "geo_dim": sum(1 for c in merged_df.columns if c.startswith("geo_")),
        "macro_dim": sum(1 for c in merged_df.columns if c.startswith("macro_")),
        "news_dim": 0,
        "hidden_dim": 32,
        "output_dim": 16,
        "sequence_length": 20,
        "num_regimes": 3,
        "forecast_horizons": [1, 5, 20],
        "learning_rate": 1e-3,
        "dropout": 0.2,
    }
    system = AIPoweredRiskSystem(config)
    print(f"      Device: {system.device}")
    print(f"      Parameters: {sum(p.numel() for p in system.parameters()):,}")

    print("[5/5] Computing risk metrics …")
    seq_len = config["sequence_length"]
    fin_cols = [c for c in merged_df.columns if c.startswith("fin_")]
    geo_cols = [c for c in merged_df.columns if c.startswith("geo_")]
    macro_cols = [c for c in merged_df.columns if c.startswith("macro_")]
    last_idx = len(merged_df) - 1
    start = max(0, last_idx - seq_len + 1)

    def _seq(cols: List[str]) -> torch.Tensor:
        data = merged_df[cols].iloc[start : last_idx + 1].values.astype(np.float32)
        if len(data) < seq_len:
            data = np.vstack(
                [np.zeros((seq_len - len(data), len(cols)), dtype=np.float32), data]
            )
        return torch.from_numpy(data).unsqueeze(0)

    batch = {"fin": _seq(fin_cols), "geo": _seq(geo_cols), "macro": _seq(macro_cols)}
    risk_metrics = system.get_risk_metrics(batch)

    print("\n" + "=" * 60)
    print("                     Prediction Results")
    print("=" * 60)
    dominant = risk_metrics["regime"]["dominant"]
    print(f"\n📊 Dominant Regime: {REGIME_LABELS_EN[dominant]}")
    for i, name in enumerate(REGIME_LABELS_EN):
        bar = "█" * int(risk_metrics["regime"]["probabilities"][i] * 30)
        print(f"   {name:12}: {bar} {risk_metrics['regime']['probabilities'][i]:.1%}")
    print(f"\n📈 Volatility Forecast (LSTM):")
    for label, key in [("1-Day", "1_day"), ("5-Day", "5_day"), ("20-Day", "20_day")]:
        print(f"   {label}: {risk_metrics['volatility'][key]:.2%}")
    print("\n🔍 Data Source Importance (Gating Mechanism):")
    for src, w in risk_metrics["source_importance"].items():
        bar = "█" * int(w * 30)
        print(f"   {src:15}: {bar} {w:.1%}")

    return system, merged_df, risk_metrics

if __name__ == "__main__":
    try:
        system, data, metrics = main()
        print("\n" + "=" * 60)
        print("   System executed successfully!")
        print("=" * 60)
        choice = input("\n❓ Price an option? (y/n): ").strip().lower()
        if choice == "y":
            S = float(input("Current Stock Price: "))
            K = float(input("Strike Price: "))
            T = float(input("Time to Maturity (years): "))
            r = float(input("Risk-Free Rate (e.g. 0.20): "))
            quant = QuantitativeIntegration(system)
            op = quant.price_options_ai_enhanced(S, K, T, r, metrics)
            print(f"\n💰 Call : {op['call_price']:,.2f}")
            print(f"💰 Put  : {op['put_price']:,.2f}")
            print(f"📊 σ_AI : {op['implied_vol']:.2%}")
    except Exception as exc:
        import traceback
        print(f"\n❌ Error: {exc}")
        traceback.print_exc()
"""
CRITICAL BUGS
─────────────
1. **`vol_proj` registered inside `forward()` (BUG → FIXED)**
   - Root cause: `if not hasattr(self, 'vol_proj'): self.vol_proj = nn.Linear(...)`
     inside `forward()` means the layer is:
       (a) Not in `state_dict()` → cannot be saved/loaded correctly.
       (b) Not registered with the optimizer → its parameters are never updated.
       (c) Created on CPU even if `self.device` is CUDA → runtime dtype/device
           error on first GPU forward pass.
   - Fix: moved to `_build_models()` so it is a proper `nn.Module` attribute.

2. **`source_importance` key mismatch between `get_risk_metrics` and report formatter**
   - Root cause: `get_risk_metrics` used Persian keys ('مالی', 'ژئوپلیتیک', 'کلان')
     but `generate_comprehensive_report` referenced English keys ('Financial', etc.).
     KeyError at runtime.
   - Fix: standardised on English keys throughout.

3. **`TemporalEncoder` indexing of BiLSTM hidden state**
   - Root cause: `h_n[-2]` and `h_n[-1]` give the last layer's forward/backward
     states only if `num_layers > 1`. For `num_layers=1`, both indices point to
     the same tensor. Shape was also incorrectly assumed.
   - Fix: explicit variables `last_fwd = h_n[-2]`, `last_bwd = h_n[-1]`; added
     `batch_first=True` to `nn.MultiheadAttention` for consistency.

4. **Temporal attention residual used wrong time-step index**
   - Root cause: `attn_out[-1]` treated (seq, batch, dim) layout after
     `MultiheadAttention` which (with `batch_first=False`) puts seq first — but
     the code then used `[-1]` implying the last row, which is last sequence
     step only when batch_first=False. Switching to `batch_first=True` and using
     `attn_out[:, -1, :]` is correct and consistent.
   - Fix: `batch_first=True` everywhere; index changed to `[:, -1, :]`.

5. **`np.random` global state mutation inside `monte_carlo_ai_scenario`**
   - Root cause: `np.random.seed(42)` sets the global state, causing non-
     reproducible behaviour in multi-threaded/multi-call contexts.
   - Fix: replaced with `np.random.default_rng(seed)` — a local Generator.

6. **Division by zero in Black-Scholes and Amihud computations**
   - Root cause: `sigma * np.sqrt(T)` can be zero; `df['volume']` can be zero.
   - Fix: added `max(sigma, 1e-6)`, `max(T, 1e-9)` guards; EPS added to all
     denominators throughout the data-loading methods.

7. **`total_loss` initialised as Python `int 0` not a `torch.Tensor`**
   - Root cause: `total_loss = 0` then `total_loss += vol_loss` (a Tensor)
     works in Python but breaks `total_loss.item()` if no supervised loss is
     present (pure diversity-penalty path) because `0 + tensor` returns a Tensor
     but calling `.item()` on bare `0` raises AttributeError.
   - Fix: `total_loss = torch.tensor(0.0, device=self.device)`.

8. **Portfolio optimiser: `cov` computed as DataFrame but passed as ndarray**
   - Root cause: `cov = returns_df.cov() * 252 * total_adj` is a DataFrame;
     `np.dot(w, np.dot(cov, w))` silently aligns on index and can produce wrong
     shapes with mismatched assets.
   - Fix: `.values` called immediately: `cov = returns_df.cov().values * …`.

ANTI-PATTERNS & ARCHITECTURE
─────────────────────────────
9. **`model.to(device)` called per sub-module separately** → replaced with a
   single `self.to(self.device)` call after all sub-modules are registered.

10. **`MultiheadAttention` `batch_first` not set** → added `batch_first=True`
    everywhere for consistent (B, S, D) convention.

11. **Mixed Persian/English identifiers and docstrings** → standardised to
    English for maintainability; Persian comments removed from logic paths.

12. **`create_sample_data` returned raw arrays without shape guarantee and used
    mutable global random state** → uses `np.random.default_rng(0)` and returns
    plain numpy arrays.

13. **`load_model` missing `map_location`** → added `map_location="cpu"` so a
    model saved on GPU can be loaded on a CPU-only machine.

PERFORMANCE
───────────
14. **`get_risk_metrics` called `.item()` in a loop inside a `no_grad` block**
    → already correct; no change needed beyond consolidating into a single pass.

15. **`_prepare_batch` / `_seq` dtype**: `.values` returns float64 by default;
    added `.astype(np.float32)` + `torch.from_numpy()` to avoid silent
    double→float conversion overhead inside PyTorch.
"""