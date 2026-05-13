
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# بخش ۱: لایه داده و استخراج ویژگی
# ============================================================

class MultiSourceDataLoader:
    """
    بارگذاری و یکپارچه‌سازی داده‌های چندمنبعی
    """
    
    def __init__(self):
        self.scalers = {}
        
    def load_financial_data(self, prices_df, volume_df):
        """
        بارگذاری داده‌های مالی سنتی
        """
        df = pd.DataFrame()
        df['price'] = prices_df
        df['volume'] = volume_df
        
        # بازدهی لگاریتمی
        df['return'] = np.log(df['price'] / df['price'].shift(1))
        df['log_return'] = df['return']
        
        # نوسان محقق‌شده (realized volatility)
        df['realized_vol'] = df['return'].rolling(window=20).std() * np.sqrt(252)
        
        # شاخص‌های فنی
        df['price_momentum'] = df['price'] / df['price'].rolling(10).mean() - 1
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # نرخ تغییر قیمت
        df['price_velocity'] = df['price'].pct_change()
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # شاخص‌های نقدشوندگی
        df['amihud_illiquidity'] = np.abs(df['return']) / df['volume']
        df['liquidity_ratio'] = 1 / (df['amihud_illiquidity'] + 1e-8)
        
        return df.dropna()
    
    def load_geopolitical_data(self, epu_df, sanctions_df, events_df):
        """
        بارگذاری داده‌های ژئوپلیتیک
        """
        df = pd.DataFrame()
        df['epu_index'] = epu_df  # شاخص عدم قطعیت سیاست
        
        # شاخص تحریم (0 = بدون تحریم، 1 = تحریم کامل)
        df['sanction_intensity'] = sanctions_df
        
        # رویدادهای سیاسی (رمزگذاری شده)
        df['political_event'] = events_df
        
        # شاخص‌های مشتق
        df['epu_ma5'] = df['epu_index'].rolling(5).mean()
        df['epu_ma20'] = df['epu_index'].rolling(20).mean()
        df['epu_acceleration'] = df['epu_index'].diff()
        
        # تعامل تحریم و EPU
        df['sanction_epu_interaction'] = df['sanction_intensity'] * df['epu_index']
        
        return df.dropna()
    
    def load_macro_data(self, forex_df, inflation_df, cds_df=None, embi_df=None):
        """
        بارگذاری داده‌های کلان اقتصادی
        """
        df = pd.DataFrame()
        df['forex_rate'] = forex_df
        df['inflation'] = inflation_df
        
        # تلاطم ارزی
        df['forex_volatility'] = df['forex_rate'].rolling(20).std()
        
        # تغییرات ارز
        df['forex_change'] = df['forex_rate'].pct_change()
        
        # شاخص‌های ریسک کشور
        if cds_df is not None:
            df['cds_spread'] = cds_df
        if embi_df is not None:
            df['embi_spread'] = embi_df
            
        # تأثیر تورم بر بازار سهام
        df['real_forex_rate'] = df['forex_rate'] / (1 + df['inflation'])
        
        return df.dropna()
    
    def merge_all_sources(self, financial_df, geo_df, macro_df, news_sentiment=None):
        """
        یکپارچه‌سازی تمام منابع داده
        """
        # اطمینان از هم‌اندازه بودن
        min_len = min(len(financial_df), len(geo_df), len(macro_df))
        
        df = pd.DataFrame()
        for col in financial_df.columns:
            df[f'fin_{col}'] = financial_df[col].values[:min_len]
        for col in geo_df.columns:
            df[f'geo_{col}'] = geo_df[col].values[:min_len]
        for col in macro_df.columns:
            df[f'macro_{col}'] = macro_df[col].values[:min_len]
            
        if news_sentiment is not None:
            df['news_sentiment'] = news_sentiment.values[:min_len]
            
        return df


# ============================================================
# بخش ۲: مدل‌های AI - لایه استخراج ویژگی
# ============================================================

class FeatureExtractor(nn.Module):
    """
    استخراج ویژگی از هر منبع داده با شبکه‌های جداگانه
    """
    
    def __init__(self, input_dim, hidden_dim=32, output_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class MultiSourceFeatureExtractor(nn.Module):
    """
    استخراج ویژگی از چند منبع با شبکه‌های مجزا
    """
    
    def __init__(self, fin_dim, geo_dim, macro_dim, news_dim,
                 hidden_dim=32, output_dim=16):
        super().__init__()
        
        # شبکه‌های جداگانه برای هر منبع
        self.fin_extractor = FeatureExtractor(fin_dim, hidden_dim, output_dim)
        self.geo_extractor = FeatureExtractor(geo_dim, hidden_dim, output_dim)
        self.macro_extractor = FeatureExtractor(macro_dim, hidden_dim, output_dim)
        
        # اگر داده اخبار داشته باشیم
        self.news_extractor = None
        if news_dim > 0:
            self.news_extractor = FeatureExtractor(news_dim, hidden_dim, output_dim)
            
        self.output_dim = output_dim
        self.num_sources = 3 if news_dim == 0 else 4
        
    def forward(self, fin_x, geo_x, macro_x, news_x=None):
        # استخراج ویژگی از هر منبع
        fin_feat = self.fin_extractor(fin_x)
        geo_feat = self.geo_extractor(geo_x)
        macro_feat = self.macro_extractor(macro_x)
        
        if news_x is not None and self.news_extractor is not None:
            news_feat = self.news_extractor(news_x)
            features = torch.stack([fin_feat, geo_feat, macro_feat, news_feat], dim=1)
        else:
            features = torch.stack([fin_feat, geo_feat, macro_feat], dim=1)
            
        return features  # shape: (batch, num_sources, output_dim)


# ============================================================
# بخش ۳: Attention-based Fusion Layer (ستون اصلی AI)
# ============================================================

class CrossModalAttention(nn.Module):
    """
    مکانیزم Attention بین منابع مختلف

    """
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value برای هر منبع
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.o_proj = nn.Linear(feature_dim, feature_dim)
        
        # لایه‌های نرمال‌سازی
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, features):
        """
        features: (batch, num_sources, feature_dim)
        """
        batch_size, num_sources, feature_dim = features.shape
        
        #Multi-Head Self-Attention
        q = self.q_proj(features)
        k = self.k_proj(features)
        v = self.v_proj(features)
        
        # تغییر شکل برای Multi-Head
        q = q.view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_sources, self.num_heads, self.head_dim).transpose(1, 2)
        
        # محاسبه Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # اعمال Attention
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, num_sources, feature_dim)
        out = self.o_proj(out)
        
        # Residual + Norm
        features = self.norm1(features + out)
        features = self.norm2(features + self.ffn(features))
        
        return features


class GatingMechanism(nn.Module):
    """
    مکانیزم Gating برای تلفیق هوشمند منابع
    
    """
    
    def __init__(self, feature_dim, num_sources):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * num_sources, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_sources),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features):
        """
        features: (batch, num_sources, feature_dim)
        """
        batch_size = features.shape[0]
        
        # Flatten برای محاسبه گیت
        flat_features = features.view(batch_size, -1)
        
        # محاسبه وزن‌ها
        weights = self.gate_network(flat_features)  # (batch, num_sources)
        
        # اعمال وزن‌ها
        fused = (features * weights.unsqueeze(-1)).sum(dim=1)  # (batch, feature_dim)
        
        return fused, weights


class FusionLayer(nn.Module):
    """
    لایه تلفیق نهایی
    ================
    ترکیب Cross-Attention + Gating
    """
    
    def __init__(self, feature_dim, num_heads=4, num_sources=3):
        super().__init__()
        self.attention = CrossModalAttention(feature_dim, num_heads)
        self.gating = GatingMechanism(feature_dim, num_sources)
        
    def forward(self, features):
        # ۱. Attention بین منابع
        attended = self.attention(features)
        
        # ۲. Gating برای تلفیق
        fused, source_weights = self.gating(attended)
        
        return fused, source_weights


# ============================================================
# بخش ۴: مدل LSTM برای پیش‌بینی سری زمانی
# ============================================================

class TemporalEncoder(nn.Module):
    """
    کدگذاری زمانی با LSTM + Attention
    =====================================
    برای پیش‌بینی نوسان و تشخیص رژیم
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # اطلاعات از هر دو جهت
        )
        
        # لایه Attention برای تمرکز بر زمان‌های مهم
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # چون bidirectional
            num_heads=4,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x):
        """
        x: (batch, sequence_length, input_dim)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # پشته کردن خروجی‌های دو جهت
        # h_n: (2, batch, hidden_dim)
        bidirectional_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        
        # Temporal Attention
        lstm_out_t = lstm_out.transpose(0, 1)  # (seq, batch, hidden*2)
        attn_out, attn_weights = self.temporal_attention(lstm_out_t, lstm_out_t, lstm_out_t)
        
        # آخرین خروجی توجه
        final_out = self.layer_norm(attn_out[-1] + bidirectional_hidden)
        
        return final_out, attn_weights


# ============================================================
# بخش ۵: مدل تشخیص رژیم (بدون قواعد دستی)
# ============================================================

class RegimeClassifier(nn.Module):
    """
    تشخیص خودکار رژیم بازار با یادگیری عمیق
    =========================================
    تفاوت با Markov Switching: هیچ فرضی درباره تعداد رژیم‌ها نداریم
    و مدل خودش الگوها را کشف می‌کند
    """
    
    def __init__(self, input_dim, hidden_dim=32, num_regimes=3):
        super().__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # خروجی برای هر رژیم
        self.regime_logits = nn.Linear(hidden_dim, num_regimes)
        
        # پارامترهای هر رژیم (نوسان، بازده)
        self.regime_params = nn.Linear(hidden_dim, num_regimes * 2)  # mu و sigma
        
        self.num_regimes = num_regimes
        
    def forward(self, x):
        """
        x: ویژگی‌های فیوژن‌شده
        """
        feat = self.feature_net(x)
        
        # احتمال هر رژیم
        regime_probs = F.softmax(self.regime_logits(feat), dim=-1)
        
        # پارامترهای هر رژیم
        params = self.regime_params(feat)  # (batch, num_regimes * 2)
        params = params.view(-1, self.num_regimes, 2)
        
        # میانگین وزنی پارامترها بر اساس احتمال
        # به جای انتخاب یک رژیم، از انتظار ریاضی استفاده می‌کنیم
        regime_mu = params[:, :, 0]  # بازده
        regime_sigma = F.softplus(params[:, :, 1])  # نوسان (همیشه مثبت)
        
        # میانگین وزنی
        expected_mu = (regime_probs * regime_mu).sum(dim=-1)
        expected_sigma = (regime_probs * regime_sigma).sum(dim=-1)
        
        return {
            'regime_probs': regime_probs,
            'regime_params': params,
            'expected_mu': expected_mu,
            'expected_sigma': expected_sigma,
            'dominant_regime': regime_probs.argmax(dim=-1)
        }


# ============================================================
# بخش ۶: مدل پیش‌بینی نوسان
# ============================================================

class VolatilityPredictor(nn.Module):
    """
    پیش‌بینی نوسان با LSTM + Attention
    ورودی: توالی ویژگی‌های چندمنبعی فیوژن‌شده
    """
    
    def __init__(self, input_dim, hidden_dim=64, forecast_horizons=[1, 5, 20]):
        super().__init__()
        
        self.forecast_horizons = forecast_horizons
        
        # کدگذار زمانی
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # پیش‌بینی برای هر افق زمانی
        self.forecast_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # نوسان همیشه مثبت
            )
            for _ in forecast_horizons
        ])
        
    def forward(self, x_sequence):
        """
        x_sequence: (batch, sequence_length, input_dim)
        """
        # کدگذاری زمانی
        temporal_feat, attn_weights = self.temporal_encoder(x_sequence)
        
        # پیش‌بینی برای هر افق
        forecasts = {}
        for i, horizon in enumerate(self.forecast_horizons):
            forecasts[f'vol_h{horizon}'] = self.forecast_heads[i](temporal_feat).squeeze(-1)
            
        return forecasts, attn_weights


# ============================================================
# بخش ۷: تحلیل احساسات با Transformer
# ============================================================

class SentimentAnalyzer(nn.Module):
    """
    تحلیل احساسات اخبار با Attention
    ===================================
    ورودی: امبدینگ اخبار
    خروجی: امتیاز احساسات + عدم قطعیت + EPU
    """
    
    def __init__(self, embed_dim=32, hidden_dim=32):
        super().__init__()
        
        # Attention برای تحلیل متن
        self.text_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # تحلیل احساسات
        self.sentiment_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # منفی، خنثی، مثبت
        )
        
        # تخمین عدم قطعیت
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 0 تا 1
        )
        
    def forward(self, text_embeddings):
        """
        text_embeddings: (batch, seq_len, embed_dim)
        """
        # Attention روی متن
        attn_out, attn_weights = self.text_attention(
            text_embeddings, text_embeddings, text_embeddings
        )
        
        # میانگین وزنی
        text_repr = attn_out.mean(dim=1)  # (batch, embed_dim)
        
        # احساسات
        sentiment_logits = self.sentiment_head(text_repr)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        
        # امتیاز: -1 (منفی) تا +1 (مثبت)
        sentiment_score = sentiment_probs[:, 2] - sentiment_probs[:, 0]
        
        # عدم قطعیت
        uncertainty = self.uncertainty_head(text_repr).squeeze(-1)
        
        # EPU: ترکیب احساسات منفی و عدم قطعیت
        epu = (1 - sentiment_score) / 2 * (1 + uncertainty) / 2
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_probs': sentiment_probs,
            'uncertainty': uncertainty,
            'epu_signal': epu,
            'attention_weights': attn_weights
        }


# ============================================================
# بخش ۸: سیستم یکپارچه (کلاس اصلی)
# ============================================================

class AIPoweredRiskSystem(nn.Module):
  
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or self._default_config()
        
        # ابعاد هر منبع
        self.fin_dim = self.config['fin_dim']
        self.geo_dim = self.config['geo_dim']
        self.macro_dim = self.config['macro_dim']
        self.news_dim = self.config.get('news_dim', 0)
        
        # دستگاه
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ساخت مدل‌ها
        self._build_models()
        
        # بهینه‌ساز
        self.optimizer = None
        self.scheduler = None
        
    def _default_config(self):
        return {
            'fin_dim': 10,
            'geo_dim': 7,
            'macro_dim': 6,
            'news_dim': 0,
            'hidden_dim': 64,
            'output_dim': 16,
            'sequence_length': 20,
            'num_regimes': 3,
            'forecast_horizons': [1, 5, 20],
            'learning_rate': 0.001,
            'dropout': 0.2
        }
        
    def _build_models(self):
        """ساخت تمام مدل‌ها"""
        
        # استخراج‌کننده ویژگی چندمنبعی
        self.feature_extractor = MultiSourceFeatureExtractor(
            fin_dim=self.fin_dim,
            geo_dim=self.geo_dim,
            macro_dim=self.macro_dim,
            news_dim=self.news_dim,
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim']
        ).to(self.device)
        
        # لایه تلفیق
        self.fusion = FusionLayer(
            feature_dim=self.config['output_dim'],
            num_heads=4,
            num_sources=3 if self.news_dim == 0 else 4
        ).to(self.device)
        
        # تشخیص رژیم
        self.regime_classifier = RegimeClassifier(
            input_dim=self.config['output_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_regimes=self.config['num_regimes']
        ).to(self.device)
        
        # پیش‌بینی نوسان
        self.volatility_predictor = VolatilityPredictor(
            input_dim=self.config['output_dim'],
            hidden_dim=self.config['hidden_dim'],
            forecast_horizons=self.config['forecast_horizons']
        ).to(self.device)
        
        # تحلیل احساسات
        if self.news_dim > 0:
            self.sentiment_analyzer = SentimentAnalyzer(
                embed_dim=self.config['output_dim'],
                hidden_dim=self.config['hidden_dim']
            ).to(self.device)
        else:
            self.sentiment_analyzer = None
            
        # ذخیره ابعاد
        self.sequence_length = self.config['sequence_length']
        
    def forward(self, batch):
        """
        پاس رو به جلوی کل سیستم
        =========================
        batch: {
            'fin': (batch, seq_len, fin_dim),
            'geo': (batch, seq_len, geo_dim),
            'macro': (batch, seq_len, macro_dim),
            'news': (batch, seq_len, news_dim) - اختیاری
        }
        """
        fin_x = batch['fin'].to(self.device)
        geo_x = batch['geo'].to(self.device)
        macro_x = batch['macro'].to(self.device)
        
        # ====== مرحله ۱: استخراج ویژگی ======
        # آخرین زمان برای تشخیص رژیم
        fin_last = fin_x[:, -1, :]
        geo_last = geo_x[:, -1, :]
        macro_last = macro_x[:, -1, :]
        
        fin_feat = self.feature_extractor.fin_extractor(fin_last)
        geo_feat = self.feature_extractor.geo_extractor(geo_last)
        macro_feat = self.feature_extractor.macro_extractor(macro_last)
        
        if self.news_dim > 0 and 'news' in batch:
            news_last = batch['news'][:, -1, :].to(self.device)
            news_feat = self.feature_extractor.news_extractor(news_last)
            features = torch.stack([fin_feat, geo_feat, macro_feat, news_feat], dim=1)
        else:
            features = torch.stack([fin_feat, geo_feat, macro_feat], dim=1)
            
        # ====== مرحله ۲: تلفیق ======
        fused_features, source_weights = self.fusion(features)
        
        # ====== مرحله ۳: تشخیص رژیم ======
        regime_output = self.regime_classifier(fused_features)
        
        # ====== مرحله ۴: پیش‌بینی نوسان ======
        # تبدیل ابعاد fin_x به output_dim (16)
        if not hasattr(self, 'vol_proj'):
            self.vol_proj = nn.Linear(fin_x.shape[-1], self.config['output_dim']).to(self.device)
        
        fin_transformed = self.vol_proj(fin_x)
        vol_forecasts, vol_attn = self.volatility_predictor(fin_transformed)        
        # ====== مرحله ۵: تحلیل احساسات (اگر موجود باشد) ======
        sentiment_output = {}
        if self.sentiment_analyzer is not None and 'news' in batch:
            sentiment_output = self.sentiment_analyzer(batch['news'])
            
        return {
            'fused_features': fused_features,
            'source_weights': source_weights,  # وزن هر منبع
            'regime': regime_output,
            'volatility': vol_forecasts,
            'sentiment': sentiment_output
        }
    
    def predict(self, data_batch):
        """پیش‌بینی بدون محاسبه گرادیان"""
        
        with torch.no_grad():
            return self.forward(data_batch)
    
    def get_risk_metrics(self, data_batch):
        """
        محاسبه معیارهای ریسک
        """
        output = self.predict(data_batch)
        
        # نوسان پیش‌بینی‌شده
        vol_1d = output['volatility']['vol_h1'].item()
        vol_5d = output['volatility']['vol_h5'].item()
        vol_20d = output['volatility']['vol_h20'].item()
        
        # احتمال رژیم‌ها
        regime_probs = output['regime']['regime_probs'][0].cpu().numpy()
        dominant_regime = output['regime']['dominant_regime'].item()
        
        # وزن منابع
        source_weights = output['source_weights'][0].cpu().numpy()
        
        # نوسان شرطی به رژیم
        expected_sigma = output['regime']['expected_sigma'].item()
        
        # احساسات
        sentiment = {}
        if output['sentiment']:
            sentiment = {
                'score': output['sentiment']['sentiment_score'].item(),
                'uncertainty': output['sentiment']['uncertainty'].item(),
                'epu': output['sentiment']['epu_signal'].item()
            }
            
        return {
            'volatility': {
                '1_day': vol_1d,
                '5_day': vol_5d,
                '20_day': vol_20d
            },
            'regime': {
                'dominant': dominant_regime,
                'probabilities': regime_probs.tolist(),
                'labels': ['عادی', 'گذار', 'بحران'],
                'expected_volatility': expected_sigma
            },
            'source_importance': {
                'مالی': source_weights[0],
                'ژئوپلیتیک': source_weights[1],
                'کلان': source_weights[2]
            },
            'sentiment': sentiment
        }


# ============================================================
# بخش ۹: کلاس آموزش
# ============================================================

class RiskSystemTrainer:
    """
    آموزش سیستم یکپارچه
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        self.device = model.device
        
        # بهینه‌ساز با تمام پارامترها
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 0.001),
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.history = {'train_loss': [], 'val_loss': []}
        
    def prepare_batch(self, df, idx, seq_len):
        """
        آماده‌سازی یک batch
        """
        batch = {}
        
        # استخراج ویژگی‌های هر منبع
        fin_cols = [c for c in df.columns if c.startswith('fin_')]
        geo_cols = [c for c in df.columns if c.startswith('geo_')]
        macro_cols = [c for c in df.columns if c.startswith('macro_')]
        
        # ساخت توالی
        def get_sequence(cols, end_idx):
            start = max(0, end_idx - seq_len + 1)
            data = df[cols].iloc[start:end_idx+1].values
            # padding اگر کم باشد
            if len(data) < seq_len:
                pad = np.zeros((seq_len - len(data), len(cols)))
                data = np.vstack([pad, data])
            return torch.FloatTensor(data).unsqueeze(0)
        
        batch['fin'] = get_sequence(fin_cols, idx)
        batch['geo'] = get_sequence(geo_cols, idx)
        batch['macro'] = get_sequence(macro_cols, idx)
        
        if 'news_sentiment' in df.columns:
            batch['news'] = get_sequence(['news_sentiment'], idx)
            
        return batch
        
    def compute_loss(self, output, targets):
        """
        محاسبه تابع هزینه چندهدفی
        """
        total_loss = 0
        loss_dict = {}
        
        # ۱. Loss نوسان (اگر هدف داشته باشیم)
        if 'target_vol' in targets:
            target_vol = targets['target_vol'].to(self.device)
            # میانگین MSE برای همه افق‌ها
            vol_loss = 0
            for key, pred in output['volatility'].items():
                vol_loss += F.mse_loss(pred, target_vol)
            vol_loss /= len(output['volatility'])
            total_loss += vol_loss
            loss_dict['vol_loss'] = vol_loss.item()
            
        # ۲. Loss تشخیص رژیم (اگر برچسب داشته باشیم)
        if 'target_regime' in targets:
            target_regime = targets['target_regime'].to(self.device)
            regime_loss = F.cross_entropy(
                output['regime']['regime_probs'],
                target_regime
            )
            total_loss += regime_loss * 0.5  # وزن کمتر
            loss_dict['regime_loss'] = regime_loss.item()
            
        # ۳. Loss احساسات (اگر هدف داشته باشیم)
        if 'target_sentiment' in targets and output['sentiment']:
            target_sent = targets['target_sentiment'].to(self.device)
            sent_loss = F.mse_loss(
                output['sentiment']['sentiment_score'],
                target_sent
            )
            total_loss += sent_loss * 0.3
            loss_dict['sentiment_loss'] = sent_loss.item()
            
        # ۴. Loss تنظیمی: تشویق به تنوع در وزن منابع
        source_weights = output['source_weights']
        diversity_loss = -torch.std(source_weights) * 0.1
        total_loss += diversity_loss
        loss_dict['diversity_loss'] = diversity_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
        
    def train_epoch(self, train_df, targets, batch_size=32):
        """آموزش یک epoch"""
        self.model.train()
        total_loss = 0
        
        n_samples = len(train_df) - self.model.sequence_length
        indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        
        for idx in indices:
            batch = self.prepare_batch(train_df, idx, self.model.sequence_length)
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            loss, _ = self.compute_loss(output, targets)
            loss.backward()
            
            # گرادیان کلیپینگ
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(indices)
    
    def validate(self, val_df, targets):
        """اعتبارسنجی"""
        self.model.eval()
        total_loss = 0
        
        n_samples = len(val_df) - self.model.sequence_length
        val_indices = np.random.choice(n_samples, min(50, n_samples), replace=False)
        
        with torch.no_grad():
            for idx in val_indices:
                batch = self.prepare_batch(val_df, idx, self.model.sequence_length)
                output = self.model(batch)
                loss, _ = self.compute_loss(output, targets)
                total_loss += loss.item()
                
        return total_loss / len(val_indices)
    
    def train(self, train_df, val_df, targets=None, epochs=100, batch_size=32):
        """
        آموزش کامل سیستم AI
        """
        print("=" * 60)
        print("          Starting AI System Training")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_df, targets or {}, batch_size)
            val_loss = self.validate(val_df, targets or {})
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")
                
        print("=" * 60)
        print("          Training Completed")
        print("=" * 60)


# ============================================================
# بخش ۱۰: یکپارچه‌سازی با مدل‌های کمی
# ============================================================

class QuantitativeIntegration:
  
    
    def __init__(self, ai_system):
        self.ai_system = ai_system
        
    def calculate_ai_adjusted_volatility(self, risk_metrics):
       
        # نوسان پیش‌بینی‌شده توسط LSTM
        lstm_vol = risk_metrics['volatility']['20_day']
        
        # نوسان شرطی به رژیم
        regime_vol = risk_metrics['regime']['expected_volatility']
        
        # احتمال هر رژیم
        probs = risk_metrics['regime']['probabilities']
        
        # ترکیب وزن‌دار
        combined_vol = (
            0.5 * lstm_vol +           # ۵۰٪ وزن: پیش‌بینی LSTM
            0.3 * regime_vol +         # ۳۰٪ وزن: نوسان شرطی
            0.2 * lstm_vol * probs[2] * 2  # ۲۰٪ وزن: ریسک بحران
        )
        
        return combined_vol
    
    def price_options_ai_enhanced(self, S, K, T, r_base, risk_metrics):
      
        # دریافت نوسان تعدیل‌شده
        sigma = self.calculate_ai_adjusted_volatility(risk_metrics)
        
        # احساسات: اگر منفی باشد، ریسک بالاتر
        if risk_metrics['sentiment']:
            sentiment_adj = 1 + (1 - risk_metrics['sentiment']['score']) * 0.2
            sigma *= sentiment_adj
            
        # محاسبه d1 و d2
        d1 = (np.log(S / K) + (r_base + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # قیمت‌ها
        call = S * norm.cdf(d1) - K * np.exp(-r_base * T) * norm.cdf(d2)
        put = K * np.exp(-r_base * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Greeks با نوسان AI
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r_base * K * np.exp(-r_base * T) * norm.cdf(d2)) / 365
        
        return {
            'call_price': call,
            'put_price': put,
            'implied_vol': sigma,
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta
            },
            'adjustments': {
                'lstm_vol': risk_metrics['volatility']['20_day'],
                'regime_vol': risk_metrics['regime']['expected_volatility'],
                'sentiment_adj': risk_metrics['sentiment'].get('score', 0) if risk_metrics['sentiment'] else 0
            }
        }
    
    def monte_carlo_ai_scenario(self, S0, T_days, risk_metrics,
                                n_simulations=5000):
        
        np.random.seed(42)
        n_steps = T_days
        dt = 1 / 252
        
        # دریافت پارامترها از AI
        regime_probs = risk_metrics['regime']['probabilities']
        
        # نوسان‌های هر سناریو
        vol_scenario = {
            0: risk_metrics['volatility']['20_day'],    # عادی
            1: risk_metrics['volatility']['20_day'] * 1.5,  # گذار
            2: risk_metrics['volatility']['20_day'] * 2.5   # بحران
        }
        
        # نرخ رشد هر سناریو
        mu_scenario = {
            0: 0.0003,   # عادی: مثبت
            1: -0.0001,  # گذار: نزدیک صفر
            2: -0.0008   # بحران: منفی
        }
        
        # شبیه‌سازی
        final_prices = []
        
        for regime_id in range(3):
            vol = vol_scenario[regime_id]
            mu = mu_scenario[regime_id]
            weight = regime_probs[regime_id]
            
            # GBM
            n_sim_per_regime = int(n_simulations * weight)
            if n_sim_per_regime == 0:
                continue
                
            Z = np.random.standard_normal((n_sim_per_regime, n_steps))
            drift = (mu - 0.5 * vol**2) * dt
            diffusion = vol * np.sqrt(dt) * Z
            returns = drift + diffusion
            
            prices = np.zeros((n_sim_per_regime, n_steps + 1))
            prices[:, 0] = S0
            for t in range(1, n_steps + 1):
                prices[:, t] = prices[:, t-1] * np.exp(returns[:, t-1])
                
            final_prices.extend(prices[:, -1].tolist())
            
        final_prices = np.array(final_prices)
        
        # آمار
        return {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'p5': np.percentile(final_prices, 5),
            'p25': np.percentile(final_prices, 25),
            'p50': np.percentile(final_prices, 50),
            'p75': np.percentile(final_prices, 75),
            'p95': np.percentile(final_prices, 95),
            'var_95': S0 - np.percentile(final_prices, 5),
            'var_99': S0 - np.percentile(final_prices, 1),
            'expected_shortfall_95': S0 - np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]),
            'probability_profit': np.mean(final_prices > S0)
        }
    
    def optimize_portfolio_ai(self, returns_df, risk_metrics, risk_tolerance='medium'):
        
        n_assets = len(returns_df.columns)
        
        # نوسان تعدیل‌شده
        regime_probs = risk_metrics['regime']['probabilities']
        regime_vol_mult = regime_probs[0] * 1.0 + regime_probs[1] * 1.8 + regime_probs[2] * 3.5
        
        # اگر احساسات منفی باشد، نوسان بیشتر
        sentiment = risk_metrics.get('sentiment', {})
        sentiment_adj = 1 + (1 - sentiment.get('score', 0)) * 0.3 if sentiment else 1
        
        total_adj = regime_vol_mult * sentiment_adj
        
        # ماتریس کوواریانس تعدیل‌شده
        cov = returns_df.cov() * 252 * total_adj
        
        # نرخ بدون ریسک تعدیل‌شده
        # در دوران گذار، نرخ بهره بالاتر می‌رود
        base_rf = 0.20
        rf = base_rf * (1 + (1 - regime_probs[0]) * 0.3)
        
        # میانگین بازدهی تعدیل‌شده
        mean_ret = returns_df.mean() * 252
        
        # بازده تعدیل‌شده بر اساس رژیم
        ret_adj = {
            'low': 0.7,    # بحران: بازده کمتر
            'medium': 0.9,  # گذار
            'high': 1.0     # عادی
        }
        adj_factor = ret_adj[risk_tolerance]
        mean_ret_adj = mean_ret * adj_factor
        
        # ====== بهینه‌سازی ======
        def portfolio_stats(w):
            ret = np.dot(w, mean_ret_adj)
            vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            sharpe = (ret - rf) / vol if vol > 0 else 0
            return ret, vol, sharpe
        
        def neg_sharpe(w):
            _, _, sr = portfolio_stats(w)
            return -sr
        
        def min_variance(w):
            return portfolio_stats(w)[1]
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # اگر ریسک‌پذیری پایین → حداقل واریانس
        if risk_tolerance == 'low':
            objective = min_variance
        else:
            objective = neg_sharpe
            
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        ret, vol, sharpe = portfolio_stats(result.x)
        
        return {
            'weights': {name: f"{w:.1f}%" for name, w in zip(returns_df.columns, result.x * 100)},
            'expected_return': f"{ret:.1%}",
            'volatility': f"{vol:.1%}",
            'sharpe_ratio': f"{sharpe:.3f}",
            'adjusted_parameters': {
                'regime_adjustment': regime_vol_mult,
                'sentiment_adjustment': sentiment_adj,
                'total_adjustment': total_adj,
                'risk_free_rate': f"{rf:.1%}",
                'dominant_regime': ['عادی', 'گذار', 'بحران'][risk_metrics['regime']['dominant']]
            }
        }


# ============================================================
# بخش ۱۱: گزارش‌گیری نهایی
# ============================================================

def generate_comprehensive_report(ai_system, test_df, returns_df, S, K, T, r):
    
    # آماده‌سازی داده برای آخرین مشاهده
    last_idx = len(test_df) - 1
    batch = {}
    
    fin_cols = [c for c in test_df.columns if c.startswith('fin_')]
    geo_cols = [c for c in test_df.columns if c.startswith('geo_')]
    macro_cols = [c for c in test_df.columns if c.startswith('macro_')]
    
    seq_len = ai_system.sequence_length
    start = max(0, last_idx - seq_len + 1)
    
    for name, cols in [('fin', fin_cols), ('geo', geo_cols), ('macro', macro_cols)]:
        data = test_df[cols].iloc[start:last_idx+1].values
        if len(data) < seq_len:
            data = np.vstack([np.zeros((seq_len - len(data), len(cols))), data])
        batch[name] = torch.FloatTensor(data).unsqueeze(0)
    
    # دریافت معیارهای ریسک
    risk_metrics = ai_system.get_risk_metrics(batch)
    
    # سیستم کمی
    quant = QuantitativeIntegration(ai_system)
    
    # قیمت اختیارات
    option_prices = quant.price_options_ai_enhanced(S, K, T, r, risk_metrics)
    
    # شبیه‌سازی
    mc_results = quant.monte_carlo_ai_scenario(S, 30, risk_metrics)
    
    # سبدها
    portfolios = {
        'conservative': quant.optimize_portfolio_ai(returns_df, risk_metrics, 'low'),
        'moderate': quant.optimize_portfolio_ai(returns_df, risk_metrics, 'medium')
    }
    
# ====== گزارش ======
    # ====== Report Generation ======
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                 Comprehensive Risk Management Report                 ║
║                 AI-Powered Risk Management Report                    ║
╚══════════════════════════════════════════════════════════════════════╝
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Market Regime Detection (AI Output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dominant Regime: {['Normal', 'Transition', 'Crisis'][risk_metrics['regime']['dominant']]}
Probabilities:
  ▪ Normal:   {risk_metrics['regime']['probabilities'][0]:.1%}
  ▪ Transition: {risk_metrics['regime']['probabilities'][1]:.1%}
  ▪ Crisis:   {risk_metrics['regime']['probabilities'][2]:.1%}
Expected Volatility (Conditional): {risk_metrics['regime']['expected_volatility']:.2%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Volatility Prediction (LSTM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ▪ Next 1 Day:   {risk_metrics['volatility']['1_day']:.2%}
  ▪ Next 5 Days:  {risk_metrics['volatility']['5_day']:.2%}
  ▪ Next 20 Days: {risk_metrics['volatility']['20_day']:.2%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Data Source Importance (Gating Mechanism)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ▪ Financial Data:   {risk_metrics['source_importance']['Financial']:.1%}
  ▪ Geopolitical Data: {risk_metrics['source_importance']['Geopolitical']:.1%}
  ▪ Macroeconomic Data: {risk_metrics['source_importance']['Macro']:.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Option Pricing (Black-Scholes + AI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adjusted Volatility: {option_prices['implied_vol']:.2%} (vs {risk_metrics['volatility']['20_day']:.2%} baseline)
Call Option Price: {option_prices['call_price']:,.0f} Toman
Put Option Price: {option_prices['put_price']:,.0f} Toman
Greeks:
  ▪ Delta:  {option_prices['greeks']['delta']:.3f}
  ▪ Gamma:  {option_prices['greeks']['gamma']:.4f}
  ▪ Vega:   {option_prices['greeks']['vega']:.2f} (Sensitivity to 1% vol change)
  ▪ Theta:  {option_prices['greeks']['theta']:.2f} (Daily decay)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Monte Carlo Simulation (1,000 Paths)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Geopolitical Scenarios: Normal, Transition, Crisis (AI-based probabilities)
Results:
  ▪ Mean:   {mc_results['mean']:,.0f} Toman
  ▪ Median: {mc_results['median']:,.0f} Toman
  ▪ 5th Percentile: {mc_results['p5']:,.0f} Toman
  ▪ 95th Percentile: {mc_results['p95']:,.0f} Toman
Risk Metrics:
  ▪ VaR 95%:   {mc_results['var_95']:,.0f} Toman ({(mc_results['var_95']/S):.1%} of capital)
  ▪ VaR 99%:   {mc_results['var_99']:,.0f} Toman ({(mc_results['var_99']/S):.1%} of capital)
  ▪ Expected Shortfall 95%: {mc_results['expected_shortfall_95']:,.0f} Toman
Probability of Profit: {mc_results['probability_profit']:.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Optimized Portfolio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conservative Portfolio:
  ▪ Expected Return: {portfolios['conservative']['expected_return']}
  ▪ Risk: {portfolios['conservative']['volatility']}
  ▪ Sharpe Ratio: {portfolios['conservative']['sharpe_ratio']}
Allocation:
{chr(10).join([f'  ▪ {k}: {v}' for k, v in portfolios['conservative']['weights'].items()])}
Moderate Portfolio:
  ▪ Expected Return: {portfolios['moderate']['expected_return']}
  ▪ Risk: {portfolios['moderate']['volatility']}
  ▪ Sharpe Ratio: {portfolios['moderate']['sharpe_ratio']}
Allocation:
{chr(10).join([f'  ▪ {k}: {v}' for k, v in portfolios['moderate']['weights'].items()])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. AI-Adjusted Parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ▪ Regime Adjustment: {portfolios['moderate']['adjusted_parameters']['regime_adjustment']:.2f}x
  ▪ Sentiment Adjustment: {portfolios['moderate']['adjusted_parameters']['sentiment_adjustment']:.2f}x
  ▪ Total Adjustment: {portfolios['moderate']['adjusted_parameters']['total_adjustment']:.2f}x
  ▪ Risk-Free Rate: {portfolios['moderate']['adjusted_parameters']['risk_free_rate']}
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(report)
    return {
        'risk_metrics': risk_metrics,
        'option_prices': option_prices,
        'monte_carlo': mc_results,
        'portfolios': portfolios
    }

def create_sample_data():
    """Create sample data for testing"""
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    n = len(dates)
    prices = 10000 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
    volume = np.random.randn(n) * 1000000 + 5000000
    epu = np.random.randn(n) * 10 + 100
    sanctions = np.random.rand(n) * 0.3
    events = np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])
    forex = np.random.randn(n) * 500 + 42000
    inflation = np.random.randn(n) * 0.01 + 0.35
    return dates, prices, volume, epu, sanctions, events, forex, inflation

def save_model(model, filepath="risk_model.pth"):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config
    }, filepath)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath="risk_model.pth"):
    """Load saved model"""
    checkpoint = torch.load(filepath)
    model = AIPoweredRiskSystem(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(model.device)
    print(f"✅ Model loaded from {filepath}")
    return model

def display_dashboard(risk_metrics):
    """Display risk dashboard in text format"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    regime_names = ['🟢 Normal', '🟡 Transition', '🔴 Crisis']
    dominant = risk_metrics['regime']['dominant']
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    AI Risk Management Dashboard                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    print(f"📌 Current Regime: {regime_names[dominant]}")
    print("-" * 66)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  Risk Metrics                                                       │
├─────────────────────────────────────────────────────────────────────┤""")
    vol = risk_metrics['volatility']['20_day']
    if vol < 0.2:
        status = "🟢 Low"
    elif vol < 0.4:
        status = "🟡 Medium"
    else:
        status = "🔴 High"
    print(f"│  20-Day Volatility: {vol:>6.2%}                              Status: {status}")
    print(f"│  1-Day Volatility:  {risk_metrics['volatility']['1_day']:>6.2%}")
    print(f"│  5-Day Volatility:  {risk_metrics['volatility']['5_day']:>6.2%}")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  Data Source Importance (AI-Activated)                              │
├─────────────────────────────────────────────────────────────────────┤""")
    for source, weight in risk_metrics['source_importance'].items():
        bar = "█" * int(weight * 30)
        print(f"│  {source:10} │ {bar:<30} │ {weight:>5.1%} │")
    print("└─────────────────────────────────────────────────────────────────────┘")

def main():
    """Main execution function for the system"""
    print("=" * 60)
    print("   AI-Powered Risk Management System")
    print("=" * 60)
    print("\n[1/5] Loading data...")
    dates, prices, volume, epu, sanctions, events, forex, inflation = create_sample_data()
    print("[2/5] Processing data...")
    loader = MultiSourceDataLoader()
    financial_df = loader.load_financial_data(prices, volume)
    geo_df = loader.load_geopolitical_data(epu, sanctions, events)
    macro_df = loader.load_macro_data(forex, inflation)
    print("[3/5] Integrating data sources...")
    merged_df = loader.merge_all_sources(financial_df, geo_df, macro_df)
    print(f"     Data Shape: {merged_df.shape}")
    print(f"     Columns: {list(merged_df.columns[:5])}...")
    print("[4/5] Initializing AI models...")
    config = {
        'fin_dim': len([c for c in merged_df.columns if c.startswith('fin_')]),
        'geo_dim': len([c for c in merged_df.columns if c.startswith('geo_')]),
        'macro_dim': len([c for c in merged_df.columns if c.startswith('macro_')]),
        'news_dim': 0,
        'hidden_dim': 32,
        'output_dim': 16,
        'sequence_length': 20,
        'num_regimes': 3,
        'forecast_horizons': [1, 5, 20],
        'learning_rate': 0.001,
        'dropout': 0.2
    }
    system = AIPoweredRiskSystem(config)
    print(f"     Model running on device: {system.device}")
    print("[5/5] Calculating risk metrics...")
    seq_len = config['sequence_length']
    last_idx = len(merged_df) - 1
    start_idx = max(0, last_idx - seq_len + 1)
    batch = {}
    fin_cols = [c for c in merged_df.columns if c.startswith('fin_')]
    geo_cols = [c for c in merged_df.columns if c.startswith('geo_')]
    macro_cols = [c for c in merged_df.columns if c.startswith('macro_')]
    for name, cols in [('fin', fin_cols), ('geo', geo_cols), ('macro', macro_cols)]:
        data = merged_df[cols].iloc[start_idx:last_idx+1].values
        if len(data) < seq_len:
            pad = np.zeros((seq_len - len(data), len(cols)))
            data = np.vstack([pad, data])
        batch[name] = torch.FloatTensor(data).unsqueeze(0)
    risk_metrics = system.get_risk_metrics(batch)
    print("\n" + "=" * 60)
    print("                    Prediction Results")
    print("=" * 60)
    regime_names = ['Normal', 'Transition', 'Crisis']
    dominant = risk_metrics['regime']['dominant']
    print(f"\n📊 Dominant Regime: {regime_names[dominant]}")
    print(f"   Probabilities:")
    for i, name in enumerate(regime_names):
        bar = "█" * int(risk_metrics['regime']['probabilities'][i] * 30)
        print(f"   {name}: {bar} {risk_metrics['regime']['probabilities'][i]:.1%}")
    print(f"\n📈 Volatility Prediction (LSTM):")
    print(f"   Next 1 Day: {risk_metrics['volatility']['1_day']:.2%}")
    print(f"   Next 5 Days: {risk_metrics['volatility']['5_day']:.2%}")
    print(f"   Next 20 Days: {risk_metrics['volatility']['20_day']:.2%}")
    print(f"\n🔍 Data Source Importance (Gating Mechanism):")
    for source, weight in risk_metrics['source_importance'].items():
        bar = "█" * int(weight * 30)
        print(f"   {source}: {bar} {weight:.1%}")
    return system, merged_df, risk_metrics

if __name__ == "__main__":
    try:
        system, data, metrics = main()
        print("\n" + "=" * 60)
        print("   System executed successfully!")
        print("=" * 60)
        print("\n📌 Suggested Next Steps:")
        print("   1. To price options, enter the following values:")
        print("      - Current Stock Price (S)")
        print("      - Strike Price (K)")
        print("      - Time to Maturity (T)")
        print("      - Risk-Free Rate (r)")
        choice = input("\n❓ Do you want to price an option? (y/n): ")
        if choice.lower() == 'y':
            S = float(input("Current Stock Price (Toman): "))
            K = float(input("Strike Price (Toman): "))
            T = float(input("Time to Maturity (Years): "))
            r = float(input("Risk-Free Rate (decimal, e.g., 0.2 for 20%): "))
            quant = QuantitativeIntegration(system)
            option_prices = quant.price_options_ai_enhanced(S, K, T, r, metrics)
            print(f"\n💰 Call Option Price: {option_prices['call_price']:,.0f} Toman")
            print(f"💰 Put Option Price: {option_prices['put_price']:,.0f} Toman")
            print(f"📊 Implied Volatility: {option_prices['implied_vol']:.2%}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()