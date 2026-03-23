"""
step1_train.py  (fixed)
========================
Trains LSTM, GRU, and Transformer on prepared sensor sequences.

FIXES IN THIS VERSION:
  1. Early stopping REMOVED — all models always train for exactly EPOCHS epochs
  2. ReduceLROnPlateau REMOVED — constant learning rate for fair comparison
  3. verbose=1 → shows clean per-epoch progress for all three models
  4. Saves .keras format (compatible with all TF 2.x versions)
  5. Added per-model training progress header so you know which model is running
  6. Saves train_times.json and train_history.pkl correctly

Input:
  data/sequences_X.npy
  data/sequences_y.npy
  data/split_indices.pkl

Output:
  models/lstm_model.keras
  models/gru_model.keras
  models/transformer_model.keras
  models/train_history.pkl
  models/train_times.json
"""

import os
import time
import json
import warnings
import numpy as np
import joblib

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.makedirs('models', exist_ok=True)

EPOCHS    = 30 
BATCH     = 64
N_CLASSES = 5


def check_tf():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        print(f'  TensorFlow {tf.__version__}')
        return tf
    except ImportError:
        print('\n  ERROR: TensorFlow not installed.')
        print('  Run:  pip install tensorflow')
        raise SystemExit(1)


# ── Version-safe TransformerBlock ─────────────────────────────────────────────
def _make_transformer_block(tf):
    keras    = tf.keras
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


# ── Model definitions ─────────────────────────────────────────────────────────
def build_lstm(tf):
    keras = tf.keras
    return keras.Sequential([
        keras.layers.Input(shape=(100, 6)),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(N_CLASSES, activation='softmax'),
    ], name='lstm')


def build_gru(tf):
    keras = tf.keras
    return keras.Sequential([
        keras.layers.Input(shape=(100, 6)),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.GRU(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(N_CLASSES, activation='softmax'),
    ], name='gru')


def build_transformer(tf):
    TransformerBlock = _make_transformer_block(tf)
    keras = tf.keras
    inp = keras.Input(shape=(100, 6))
    x   = keras.layers.Dense(64)(inp)
    x   = TransformerBlock(64, heads=4, ff=128, name='attn_block_1')(x)
    x   = TransformerBlock(64, heads=4, ff=128, name='attn_block_2')(x)
    x   = keras.layers.GlobalAveragePooling1D()(x)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Dense(64, activation='gelu')(x)
    x   = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    return keras.Model(inp, out, name='transformer')


# ── Training — NO early stopping, always runs all EPOCHS ──────────────────────
def train_one(tf, name, build_fn, X_tr, y_tr):
    print(f'\n  {"="*50}')
    print(f'  Training {name.upper()}  —  {EPOCHS} epochs')
    print(f'  {"="*50}')

    # Labels must be 0-indexed for sparse_categorical_crossentropy
    y_tr0 = y_tr - 1

    model = build_fn(tf)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # NO callbacks — no EarlyStopping, no ReduceLROnPlateau
    # Every model trains for exactly EPOCHS epochs
    t0   = time.time()
    hist = model.fit(
        X_tr, y_tr0,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[],        # empty — no early stopping
        verbose=1,           # show progress every epoch
    )
    elapsed = round(time.time() - t0, 1)

    # Confirm all epochs ran
    actual_epochs = len(hist.history['loss'])
    print(f'\n  {name.upper()} completed: {actual_epochs}/{EPOCHS} epochs  '
          f'(should always be {EPOCHS})')
    print(f'  Final train acc : {hist.history["accuracy"][-1]:.4f}')
    print(f'  Final val acc   : {hist.history["val_accuracy"][-1]:.4f}')
    print(f'  Best val acc    : {max(hist.history["val_accuracy"]):.4f}')
    print(f'  Training time   : {elapsed}s')

    save_path = f'models/{name}_model.keras'
    model.save(save_path)
    print(f'  Saved → {save_path}')

    return hist, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '='*55)
    print(f'  Step 1 — Training deep learning models ({EPOCHS} epochs each)')
    print('='*55)

    tf = check_tf()

    # Check all required files exist
    for f in ['data/sequences_X.npy', 'data/sequences_y.npy',
              'data/split_indices.pkl']:
        if not os.path.exists(f):
            print(f'  ERROR: {f} not found. Run step0_prepare.py first.')
            raise SystemExit(1)

    X      = np.load('data/sequences_X.npy')
    y      = np.load('data/sequences_y.npy')
    splits = joblib.load('data/split_indices.pkl')
    tr_idx = splits['train']

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]

    print(f'\n  Training windows : {len(X_tr):,}')
    print(f'  Shape            : {X_tr.shape}')
    print(f'  Epochs per model : {EPOCHS}')
    print(f'  Batch size       : {BATCH}')
    print(f'\n  Label distribution (train):')
    for r, cnt in zip(*np.unique(y_tr, return_counts=True)):
        pct = 100 * cnt / len(y_tr)
        print(f'    Rating {r}: {cnt:>6,}  ({pct:.1f}%)')

    builders  = {
        'lstm':        build_lstm,
        'gru':         build_gru,
        'transformer': build_transformer,
    }
    histories = {}
    times     = {}

    for name, builder in builders.items():
        hist, elapsed = train_one(tf, name, builder, X_tr, y_tr)
        histories[name] = {
            'loss':         hist.history['loss'],
            'val_loss':     hist.history['val_loss'],
            'accuracy':     hist.history['accuracy'],
            'val_accuracy': hist.history['val_accuracy'],
        }
        times[name] = elapsed

    # Save histories and times
    joblib.dump(histories, 'models/train_history.pkl')
    with open('models/train_times.json', 'w') as f:
        json.dump(times, f, indent=2)

    # Summary table
    print(f'\n\n  {"="*55}')
    print(f'  TRAINING SUMMARY')
    print(f'  {"="*55}')
    print(f'  {"Model":<14} {"Epochs":>8} {"Final val acc":>15} {"Time (s)":>10}')
    print(f'  {"-"*50}')
    for name in ['lstm', 'gru', 'transformer']:
        h = histories[name]
        print(f'  {name.upper():<14} {len(h["loss"]):>8} '
              f'{h["val_accuracy"][-1]:>15.4f} '
              f'{times[name]:>10.1f}')
    print(f'  {"="*55}')
    print(f'\n  Saved → models/train_history.pkl')
    print(f'  Saved → models/train_times.json')
    print(f'\n  All 3 models trained for exactly {EPOCHS} epochs each.')
    print(f'  Next: python step2_evaluate.py')


if __name__ == '__main__':
    main()