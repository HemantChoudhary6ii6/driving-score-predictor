"""
app.py  —  Deep Learning Driving Score Dashboard
=================================================
Upload any raw session CSV -> driving score from LSTM, GRU, or Transformer.

SCORE DIRECTION:
  100 = perfect / safest driver  (Rating 5)
  0   = most dangerous driver    (Rating 1)

Run:  streamlit run app.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

WINDOW      = 100
STEP        = 50
SENSOR_COLS = ['X_Acc', 'Y_Acc', 'Z_Acc', 'X_Gyro', 'Y_Gyro', 'Z_Gyro']
ACC_LIMIT   = 40.0
GYRO_LIMIT  = 300.0
GAP_MS      = 5000

# 100 = safest (Rating 5), 0 = most dangerous (Rating 1)
RISK_W       = np.array([0, 25, 50, 75, 100], dtype=np.float32)
MODEL_COLORS = {'lstm': '#2563EB', 'gru': '#0E7490', 'transformer': '#854F0B'}
MODEL_LABELS = {'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer'}


def score_to_label(score):
    """100=perfect, 0=dangerous."""
    if score >= 80: return 'Excellent',  '#22c55e', '★★★★★'
    if score >= 60: return 'Good',       '#84cc16', '★★★★☆'
    if score >= 40: return 'Moderate',   '#f59e0b', '★★★☆☆'
    if score >= 20: return 'Poor',       '#ef4444', '★★☆☆☆'
    return                 'Dangerous',  '#991b1b', '★☆☆☆☆'


def score_to_stars(score):
    return round(score / 20, 1)


RECOMMENDATIONS = {
    'Excellent':  '✅ Excellent driving! Smooth, controlled, and consistent throughout.',
    'Good':       '👍 Good driving overall. Minor improvements possible in cornering or braking.',
    'Moderate':   '⚡ Moderate driving. Some aggressive events detected — try smoother acceleration.',
    'Poor':       '⚠️ Poor driving detected. Reduce speed and avoid sudden braking or sharp turns.',
    'Dangerous':  '🚨 Dangerous driving patterns. Immediate improvement needed for safety.',
}


# ── Version-safe TransformerBlock ─────────────────────────────────────────────
def _make_transformer_block(tf):
    keras = tf.keras
    register = None
    for _try in [
        lambda: keras.saving.register_keras_serializable,
        lambda: keras.utils.register_keras_serializable,
        lambda: __import__('keras').saving.register_keras_serializable,
    ]:
        try:
            fn = _try()
            if callable(fn):
                register = fn
                break
        except Exception:
            pass

    class TransformerBlock(keras.layers.Layer):
        def __init__(self, d, heads, ff, drop=0.2, **kw):
            super().__init__(**kw)
            self.d = d; self.heads = heads
            self.ff = ff; self.drop = drop
            self.attn  = keras.layers.MultiHeadAttention(
                num_heads=heads, key_dim=d // heads)
            self.ffn   = keras.Sequential([
                keras.layers.Dense(ff, activation='gelu'),
                keras.layers.Dense(d),
            ])
            self.ln1   = keras.layers.LayerNormalization(epsilon=1e-6)
            self.ln2   = keras.layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = keras.layers.Dropout(drop)
            self.drop2 = keras.layers.Dropout(drop)

        def call(self, x, training=False):
            x = self.ln1(x + self.drop1(self.attn(x, x), training=training))
            x = self.ln2(x + self.drop2(self.ffn(x),     training=training))
            return x

        def get_config(self):
            cfg = super().get_config()
            cfg.update(d=self.d, heads=self.heads, ff=self.ff, drop=self.drop)
            return cfg

    if register is not None:
        try:
            TransformerBlock = register(
                package='DrivingRisk', name='TransformerBlock')(TransformerBlock)
        except Exception:
            pass
    return TransformerBlock


@st.cache_resource
def load_all_models():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        return None, None

    TransformerBlock = _make_transformer_block(tf)
    custom_objects   = {'TransformerBlock': TransformerBlock}

    models = {}
    for name in ['lstm', 'gru', 'transformer']:
        path = f'models/{name}_model.keras'
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(
                    path, custom_objects=custom_objects)
            except Exception as e:
                st.warning(f'Could not load {name}: {e}')

    scaler = joblib.load('data/dl_scaler.pkl') \
             if os.path.exists('data/dl_scaler.pkl') else None
    return models, scaler


def clean_upload(df):
    drop = [c for c in df.columns if c.strip().upper() in ('ID', 'NAME')]
    df   = df.drop(columns=drop, errors='ignore')
    if 'Rating' not in df.columns:
        df['Rating'] = 3
    df[SENSOR_COLS] = df[SENSOR_COLS].apply(pd.to_numeric, errors='coerce')
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().bfill()
    df = df.dropna(subset=SENSOR_COLS)
    df = df[~(df[SENSOR_COLS] == 0).all(axis=1)]
    df = df[df['Z_Acc'] != 0].reset_index(drop=True)
    for col in ['X_Acc', 'Y_Acc', 'Z_Acc']:
        df[col] = df[col].clip(-ACC_LIMIT, ACC_LIMIT)
    for col in ['X_Gyro', 'Y_Gyro', 'Z_Gyro']:
        df[col] = df[col].clip(-GYRO_LIMIT, GYRO_LIMIT)
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    df['session_id'] = (df['Timestamp'].diff().fillna(0) > GAP_MS).cumsum()
    return df


def make_windows(df, scaler):
    mu, sig = scaler['mean'], scaler['std']
    wins = []
    for _, sdf in df.groupby('session_id'):
        sdf  = sdf.reset_index(drop=True)
        vals = sdf[SENSOR_COLS].values.astype(np.float32)
        for i in range(0, len(sdf) - WINDOW + 1, STEP):
            wins.append(vals[i: i + WINDOW])
    if not wins:
        return np.empty((0, WINDOW, 6), dtype=np.float32)
    X = np.array(wins, dtype=np.float32)
    return (X - mu) / sig


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Driving Score', page_icon='🏍️', layout='wide')
st.title('🏍️ Driving Score Dashboard')
st.caption('LSTM  ·  GRU  ·  Transformer  |  **100 = Perfect Driver**  ·  **0 = Dangerous Driver**')

tab1, tab2, tab3 = st.tabs(['📂 Score a Session', '📊 Model Evaluation', 'ℹ️ About'])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Score a session
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Upload a raw session CSV')
    st.caption('Columns needed: `Timestamp, X_Acc, Y_Acc, Z_Acc, X_Gyro, Y_Gyro, Z_Gyro`  '
               '(ID/Name column is ignored automatically)')

    models, scaler = load_all_models()

    if not models:
        st.error('No trained models found. Run `python step1_train.py` first.')
    else:
        avail_names  = list(models.keys())
        model_choice = st.radio(
            'Choose model',
            avail_names,
            format_func=lambda n: MODEL_LABELS[n],
            horizontal=True
        )

        uploaded = st.file_uploader('Choose CSV', type=['csv'])

        if uploaded:
            raw_df = pd.read_csv(uploaded, low_memory=False)
            st.success(f'Loaded {len(raw_df):,} rows')

            with st.expander('Preview raw data (first 20 rows)'):
                st.dataframe(raw_df.head(20), use_container_width=True)

            missing = [c for c in SENSOR_COLS if c not in raw_df.columns]
            if missing:
                st.error(f'Missing required columns: {missing}')
                st.stop()

            with st.spinner('Cleaning and windowing …'):
                clean_df = clean_upload(raw_df)
                if len(clean_df) < WINDOW:
                    st.error(f'Not enough rows ({len(clean_df)}) — need >= {WINDOW}')
                    st.stop()
                X = make_windows(clean_df, scaler)

            if len(X) == 0:
                st.error('No complete windows could be extracted.')
                st.stop()

            with st.spinner(f'Running {MODEL_LABELS[model_choice]} …'):
                model  = models[model_choice]
                proba  = model.predict(X, verbose=0)
                scores = (proba * RISK_W).sum(axis=1)
                score  = float(scores.mean())

            label, color, stars = score_to_label(score)
            star_num = score_to_stars(score)
            c = MODEL_COLORS[model_choice]

            st.markdown('---')

            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric('Driving Score', f'{score:.1f} / 100')
            col2.metric('Star Rating',   f'{star_num} / 5.0')
            col3.metric('Band',          label)
            col4.metric('Windows',       str(len(scores)))
            col5.metric('Model',         MODEL_LABELS[model_choice])

            # Banner
            st.markdown(
                f'<div style="background:{color}22;border-left:6px solid {color};'
                f'padding:16px 20px;border-radius:8px;margin:12px 0">'
                f'<span style="font-size:28px">{stars}</span>'
                f'&nbsp;&nbsp;<strong style="color:{color};font-size:20px">{label}</strong>'
                f'&nbsp;&nbsp;—&nbsp;&nbsp;'
                f'<span style="font-size:18px">Driving score: <strong>{score:.1f} / 100</strong>'
                f'&nbsp;|&nbsp;<strong>{star_num} / 5.0 stars</strong></span></div>',
                unsafe_allow_html=True
            )
            st.progress(min(score / 100, 1.0))
            st.info(RECOMMENDATIONS[label])

            # Timeline chart
            st.subheader('Driving score across the session')
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.fill_between(range(len(scores)), scores, alpha=0.15, color=c)
            ax.plot(scores, lw=1.5, color=c)
            ax.axhline(80, color='#22c55e', ls='--', lw=0.8, alpha=0.7, label='Excellent (80)')
            ax.axhline(60, color='#84cc16', ls='--', lw=0.8, alpha=0.7, label='Good (60)')
            ax.axhline(40, color='#f59e0b', ls='--', lw=0.8, alpha=0.7, label='Moderate (40)')
            ax.axhline(20, color='#ef4444', ls='--', lw=0.8, alpha=0.7, label='Poor (20)')
            ax.set_xlabel('Window (each approx 1-2 seconds)')
            ax.set_ylabel('Driving score (0=dangerous, 100=perfect)')
            ax.set_ylim(-2, 105)
            ax.legend(fontsize=8, loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Window breakdown
            st.subheader('Window breakdown')
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric('🟢 Excellent (80-100)', int((scores >= 80).sum()))
            b2.metric('🟡 Good (60-79)',        int(((scores >= 60) & (scores < 80)).sum()))
            b3.metric('🟠 Moderate (40-59)',    int(((scores >= 40) & (scores < 60)).sum()))
            b4.metric('🔴 Poor (20-39)',         int(((scores >= 20) & (scores < 40)).sum()))
            b5.metric('⛔ Dangerous (0-19)',     int((scores < 20).sum()))

            # All models comparison
            if len(models) > 1:
                st.subheader('All models on this session')
                rows = []
                for name, mdl in models.items():
                    p = mdl.predict(X, verbose=0)
                    s = (p * RISK_W).sum(axis=1)
                    lbl, _, sts = score_to_label(float(s.mean()))
                    rows.append({
                        'Model':       MODEL_LABELS[name],
                        'Score / 100': f'{s.mean():.1f}',
                        'Stars / 5':   f'{score_to_stars(s.mean())}',
                        'Stars':       sts,
                        'Band':        lbl,
                        'Windows':     len(s),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            st.info('👆 Upload a CSV from the Android sensor app.\n\n'
                    'Required columns: `Timestamp, X_Acc, Y_Acc, Z_Acc, X_Gyro, Y_Gyro, Z_Gyro`')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation charts
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Deep Learning Model Evaluation')

    if os.path.exists('outputs/results.csv'):
        df_res = pd.read_csv('outputs/results.csv')
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    charts = [
        ('outputs/1_training_curves.png',   'Training curves',
         'Loss and accuracy per epoch for each model.'),
        ('outputs/2_model_comparison.png',  'Model comparison',
         'F1, AUC-ROC, and training time side by side.'),
        ('outputs/3_confusion_matrices.png','Confusion matrices',
         'Which ratings get confused with which.'),
        ('outputs/4_risk_distributions.png','Driving score distributions',
         'Higher scores = safer driving. Should cluster right for good drivers.'),
        ('outputs/5_risk_by_rating.png',    'Mean score per rating',
         'Sanity check: score should increase from Rating 1 to Rating 5.'),
    ]

    found = False
    for path, title, caption in charts:
        if os.path.exists(path):
            st.subheader(title)
            st.caption(caption)
            st.image(path, use_column_width=True)
            found = True

    if not found:
        st.info('Run the pipeline first:\n```\npython run_pipeline.py\n```')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader('How the driving score works')
    st.markdown("""
    ### Score direction — higher is better
    | Score | Stars | Band | Meaning |
    |-------|-------|------|---------|
    | 80–100 | ★★★★★ | Excellent | Near-perfect, smooth driving |
    | 60–79  | ★★★★☆ | Good      | Good driving, minor events |
    | 40–59  | ★★★☆☆ | Moderate  | Some aggressive events |
    | 20–39  | ★★☆☆☆ | Poor      | Frequent risky behaviour |
    | 0–19   | ★☆☆☆☆ | Dangerous | Consistently dangerous |

    ### Score formula
    ```
    score = P(Rating=1)×0  + P(Rating=2)×25 + P(Rating=3)×50
          + P(Rating=4)×75 + P(Rating=5)×100
    ```
    - **Rating 1** = most dangerous → contributes **0** to score
    - **Rating 5** = safest → contributes **100** to score
    - Session score = mean across all windows
    - Star rating = score ÷ 20  (e.g. 80 → 4.0 stars)

    ### Three deep learning models
    | Model | Architecture | Strength |
    |-------|-------------|----------|
    | **LSTM** | 2x LSTM layers (128→64) | Step-by-step temporal patterns |
    | **GRU** | 2x GRU layers (128→64) | Faster, similar accuracy to LSTM |
    | **Transformer** | 2x self-attention (64 dim, 4 heads) | Global attention over window |

    ### Pipeline
    ```
    step0_prepare.py   → build sequences
    step1_train.py     → train LSTM, GRU, Transformer
    step2_evaluate.py  → compare models, generate charts
    step3_score.py     → score any new session CSV
    streamlit run app.py → this dashboard
    ```
    """)