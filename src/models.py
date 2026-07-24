import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import keras.backend as K

class CategoricalFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, alpha=None, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Clip untuk numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross-entropy per kelas: -y_true * log(y_pred)
        cross_entropy = -y_true * tf.math.log(y_pred)

        # p_t: probabilitas kelas yang benar
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Focal loss per sampel
        focal_loss = focal_weight * cross_entropy

        # Optional: alpha weighting per kelas
        if self.alpha is not None:
            alpha_t = tf.cast(self.alpha, y_pred.dtype)
            focal_loss = alpha_t * focal_loss

        # Sum across classes, mean across batch
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config


# =============================================================================
# CBAM — Convolutional Block Attention Module
# =============================================================================

class CBAM:

    @staticmethod
    def channel_attention(input_feature, ratio=8, name=''):
        channel = input_feature.shape[-1]

        shared_dense_one = layers.Dense(
            channel // ratio, activation='relu',
            kernel_initializer='he_normal',
            use_bias=True, bias_initializer='zeros',
            name=f'{name}_channel_shared_dense_one'
        )
        shared_dense_two = layers.Dense(
            channel,
            kernel_initializer='he_normal',
            use_bias=True, bias_initializer='zeros',
            name=f'{name}_channel_shared_dense_two'
        )

        avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_channel_avg_pool')(input_feature)
        avg_pool = layers.Reshape((1, 1, channel), name=f'{name}_channel_avg_reshape')(avg_pool)
        avg_pool = shared_dense_one(avg_pool)
        avg_pool = shared_dense_two(avg_pool)

        max_pool = layers.GlobalMaxPooling2D(name=f'{name}_channel_max_pool')(input_feature)
        max_pool = layers.Reshape((1, 1, channel), name=f'{name}_channel_max_reshape')(max_pool)
        max_pool = shared_dense_one(max_pool)
        max_pool = shared_dense_two(max_pool)

        cbam_feature = layers.Add(name=f'{name}_channel_add')([avg_pool, max_pool])
        cbam_feature = layers.Activation('sigmoid', name=f'{name}_channel_sigmoid')(cbam_feature)

        return layers.Multiply(name=f'{name}_channel_multiply')([input_feature, cbam_feature])

    @staticmethod
    def spatial_attention(input_feature, kernel_size=7, name=''):
        avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
        max_pool = tf.reduce_max(input_feature,  axis=3, keepdims=True)

        concat = layers.Concatenate(axis=3, name=f'{name}_spatial_concat')([avg_pool, max_pool])

        cbam_feature = layers.Conv2D(
            filters=1, kernel_size=kernel_size, strides=1,
            padding='same', activation='sigmoid',
            kernel_initializer='he_normal', use_bias=False,
            name=f'{name}_spatial_conv'
        )(concat)

        return layers.Multiply(name=f'{name}_spatial_multiply')([input_feature, cbam_feature])

    @classmethod
    def cbam_block(cls, cbam_feature, ratio=8, kernel_size=7, name=''):
        cbam_feature = cls.channel_attention(cbam_feature, ratio, name=f'{name}_ca')
        cbam_feature = cls.spatial_attention(cbam_feature, kernel_size, name=f'{name}_sa')
        return cbam_feature


# =============================================================================
# Model 1 — EfficientNetB0 Baseline
# =============================================================================

def create_efficientnetb0_baseline(input_shape=(224, 224, 3),
                                    num_classes=4,
                                    dropout_rate=0.3):

    inputs = tf.keras.Input(shape=input_shape, name='input_image')

    # ── Preprocessing ─────────────────────────────────────────────────────────
    x = layers.Rescaling(scale=255.0, name='rescale_to_255')(inputs)

    # ── Backbone ──────────────────────────────────────────────────────────────
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )

    base_model.trainable = False

    # ── Classification Head ───────────────────────────────────────────────────
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization(name='bn_final')(x)
    x = layers.Dense(
        256, activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='dense_256'
    )(x)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.Dense(
        128, activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='dense_128'
    )(x)
    x = layers.BatchNormalization(name='bn_dense_2')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)

    predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=inputs, outputs=predictions, name='EfficientNetB0_Baseline')
    return model


# =============================================================================
# Model 2 — EfficientNetB0 + CBAM
# =============================================================================

def create_efficientnetb0_cbam(input_shape=(224, 224, 3),
                                num_classes=4,
                                dropout_rate=0.3,
                                reduction_ratio=8,
                                kernel_size=7):

    inputs   = tf.keras.Input(shape=input_shape, name='input_image')
    x_scaled = layers.Rescaling(scale=255.0, name='rescale_to_255')(inputs)

    # ── Backbone dengan input_shape (bukan input_tensor) ──────────────────────
    _base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Cari residual 'add' layers untuk multi-stage CBAM
    add_layers = [l for l in _base.layers if l.name.endswith('_add')]
    n          = len(add_layers)

    backbone = tf.keras.Model(
        inputs=_base.input,
        outputs=[
            add_layers[n // 3].output,      # early: ~40ch
            add_layers[2 * n // 3].output,  # mid:   ~112-192ch
            _base.output                     # final: 1280ch
        ],
        name='efficientnetb0'
    )
    backbone.trainable = False 
    
    # ── Jalankan backbone, dapatkan 3 output sekaligus ────────────────────────
    early_out, mid_out, final_out = backbone(x_scaled)

    # ── Multi-Stage CBAM ──────────────────────────────────────────────────────
    early_cbam = CBAM.cbam_block(early_out, ratio=4,  kernel_size=5, name='cbam_early')
    mid_cbam   = CBAM.cbam_block(mid_out,   ratio=8,  kernel_size=7, name='cbam_mid')
    final_cbam = CBAM.cbam_block(final_out, ratio=16, kernel_size=7, name='cbam_final')

    # GAP per branch
    early_gap = layers.GlobalAveragePooling2D(name='gap_early')(early_cbam)
    mid_gap   = layers.GlobalAveragePooling2D(name='gap_mid')(mid_cbam)
    final_gap = layers.GlobalAveragePooling2D(name='gap_final')(final_cbam)

    # ── Equal Projection ──────────────────────────────────────────────────────
    early_proj = layers.Dense(
        256, activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='proj_early'
    )(early_gap)
    mid_proj = layers.Dense(
        256, activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='proj_mid'
    )(mid_gap)
    final_proj = layers.Dense(
        256, activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='proj_final'
    )(final_gap)

    x = layers.Concatenate(name='concat_branches')([early_proj, mid_proj, final_proj])  # 768d

    # ── Classification Head ───────────────────────────────────────────────────
    x = layers.BatchNormalization(name='bn_concat')(x)
    x = layers.Dense(
        256, activation='relu',
        kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='dense_256'
    )(x)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    x = layers.Dense(
        128, activation='relu',
        kernel_initializer='he_uniform',
        name='dense_128'
    )(x)
    x = layers.BatchNormalization(name='bn_dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)

    predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=predictions, name='EfficientNetB0_CBAM')
    return model


# =============================================================================
# Compile helper (Adam fixed LR, no ExponentialDecay)
# =============================================================================

def compile_model(model, initial_learning_rate=0.0001,
                  decay_steps=40, decay_rate=0.96,
                  gamma=2.0):
    model.compile(
        loss=CategoricalFocalLoss(gamma=gamma),
        optimizer=Adam(learning_rate=initial_learning_rate),
        metrics=['accuracy']
    )
    return model


def create_lr_schedule(initial_learning_rate=1e-4, decay_steps=40, decay_rate=0.96):
    """
    Learning rate schedule function (untuk LearningRateScheduler callback jika diperlukan).
    """
    def schedule(epoch):
        return initial_learning_rate * (decay_rate ** (epoch // decay_steps))
    return schedule