# app.py - فایل اصلی وب‌اپلیکیشن
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import plotly.graph_objs as go
import plotly.utils

# وارد کردن سیستم اصلی شما
from risk_system import (
    AIPoweredRiskSystem, MultiSourceDataLoader, 
    QuantitativeIntegration, create_sample_data
)

app = Flask(__name__)
CORS(app)

# متغیرهای گلوبال
risk_system = None
risk_metrics = None
quant_integration = None
current_data = None

def initialize_system():
    """راه‌اندازی اولیه سیستم"""
    global risk_system, risk_metrics, quant_integration, current_data
    
    print("🔄 در حال راه‌اندازی سیستم AI...")
    
    # ایجاد داده‌های نمونه
    dates, prices, volume, epu, sanctions, events, forex, inflation = create_sample_data()
    
    # بارگذاری داده‌ها
    loader = MultiSourceDataLoader()
    financial_df = loader.load_financial_data(prices, volume)
    geo_df = loader.load_geopolitical_data(epu, sanctions, events)
    macro_df = loader.load_macro_data(forex, inflation)
    merged_df = loader.merge_all_sources(financial_df, geo_df, macro_df)
    current_data = merged_df
    
    # تنظیمات مدل
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
    
    risk_system = AIPoweredRiskSystem(config)
    
    # آماده‌سازی batch برای پیش‌بینی
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
    
    risk_metrics = risk_system.get_risk_metrics(batch)
    quant_integration = QuantitativeIntegration(risk_system)
    
    print("✅ سیستم با موفقیت راه‌اندازی شد!")
    return True

@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('dashboard.html')

@app.route('/api/risk-metrics')
def get_risk_metrics():
    """API دریافت معیارهای ریسک"""
    if risk_metrics is None:
        return jsonify({'error': 'سیستم راه‌اندازی نشده است'}), 500
    
    regime_names = ['عادی', 'گذار', 'بحران']
    dominant = risk_metrics['regime']['dominant']
    
    return jsonify({
        'regime': {
            'dominant': regime_names[dominant],
            'dominant_index': dominant,
            'probabilities': {
                'عادی': risk_metrics['regime']['probabilities'][0],
                'گذار': risk_metrics['regime']['probabilities'][1],
                'بحران': risk_metrics['regime']['probabilities'][2]
            },
            'expected_volatility': risk_metrics['regime']['expected_volatility']
        },
        'volatility': {
            '1_day': risk_metrics['volatility']['1_day'],
            '5_day': risk_metrics['volatility']['5_day'],
            '20_day': risk_metrics['volatility']['20_day']
        },
        'source_importance': risk_metrics['source_importance'],
        'sentiment': risk_metrics.get('sentiment', {}),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/option-pricing', methods=['POST'])
def price_option():
    """API قیمت‌گذاری اختیار معامله"""
    data = request.json
    S = float(data.get('S', 10000))
    K = float(data.get('K', 11000))
    T = float(data.get('T', 30) / 365)  # تبدیل روز به سال
    r = float(data.get('r', 0.20))
    
    result = quant_integration.price_options_ai_enhanced(S, K, T, r, risk_metrics)
    
    return jsonify({
        'call_price': result['call_price'],
        'put_price': result['put_price'],
        'implied_vol': result['implied_vol'],
        'greeks': result['greeks']
    })

@app.route('/api/monte-carlo')
def monte_carlo():
    """API شبیه‌سازی مونت کارلو"""
    S = float(request.args.get('S', 10000))
    days = int(request.args.get('days', 30))
    
    result = quant_integration.monte_carlo_ai_scenario(S, days, risk_metrics)
    
    return jsonify({
        'mean': result['mean'],
        'median': result['median'],
        'p5': result['p5'],
        'p95': result['p95'],
        'var_95': result['var_95'],
        'var_99': result['var_99'],
        'expected_shortfall': result['expected_shortfall_95'],
        'profit_probability': result['probability_profit']
    })

@app.route('/api/chart-data')
def chart_data():
    """API داده‌های نمودار"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'داده موجود نیست'}), 500
    
    # گرفتن ۱۰۰ روز آخر برای نمودار
    data = current_data.tail(100)
    
    # ستون‌های قیمت و حجم
    price_col = [c for c in data.columns if 'fin_price' in c][0]
    vol_col = [c for c in data.columns if 'fin_realized_vol' in c][0]
    
    return jsonify({
        'dates': data.index.astype(str).tolist(),
        'prices': data[price_col].values.tolist(),
        'volatility': data[vol_col].values.tolist()
    })

if __name__ == '__main__':
    initialize_system()
    print("\n" + "=" * 60)
    print("🚀 وب‌سرویس مدیریت ریسک هوشمند راه‌اندازی شد!")
    print("📍 آدرس: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)