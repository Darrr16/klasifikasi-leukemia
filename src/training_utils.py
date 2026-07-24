import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, mean_absolute_error
)
import itertools


# =============================================================================
# 1. Train model
# =============================================================================

def train_model(model, train_gen, valid_gen, epochs=50, model_name='model'):
    """
    Train model dan kembalikan history (workers=2).
    """
    print(f'Training {model_name} for {epochs} epochs...')
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        steps_per_epoch=None,
        workers=2,
        verbose=1
    )
    print(f'Training completed for {model_name}')
    return history


# =============================================================================
# 2. Training history plot
# =============================================================================

def plot_training_history(history, model_name='Model', save_path=None):
    """
    Plot kurva loss dan accuracy training/validation.
    Marker epoch terbaik.
    """
    train_acc  = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc    = history.history['val_accuracy']
    val_loss   = history.history['val_loss']
    epochs     = range(1, len(train_acc) + 1)

    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_acc_epoch  = np.argmax(val_acc) + 1

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(epochs, train_loss, 'r-', label='Training Loss',   linewidth=2)
    ax1.plot(epochs, val_loss,   'b-', label='Validation Loss', linewidth=2)
    ax1.scatter(best_val_loss_epoch, val_loss[best_val_loss_epoch - 1],
                s=100, c='orange', label=f'Best Epoch: {best_val_loss_epoch}', zorder=5)
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, 'r-', label='Training Accuracy',   linewidth=2)
    ax2.plot(epochs, val_acc,   'b-', label='Validation Accuracy', linewidth=2)
    ax2.scatter(best_val_acc_epoch, val_acc[best_val_acc_epoch - 1],
                s=100, c='orange', label=f'Best Epoch: {best_val_acc_epoch}', zorder=5)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_training_history.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 3. Model evaluation
# =============================================================================

def evaluate_model(model, test_gen, model_name='Model'):
    print(f'Evaluating {model_name}...')

    # Ambil semua test data sekaligus (test_gen batch_size = len(test_df))
    X_test, y_test = next(test_gen)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    predictions       = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes      = np.argmax(y_test, axis=1)

    mae = mean_absolute_error(true_classes, predicted_classes)

    results = {
        'model_name':        model_name,
        'test_accuracy':     test_accuracy,
        'test_loss':         test_loss,
        'mae':               mae,
        'predictions':       predictions,
        'predicted_classes': predicted_classes,
        'true_classes':      true_classes,
        'X_test':            X_test,
        'y_test':            y_test
    }

    print(f'\n{model_name} Test Results:')
    print(f'  Accuracy : {test_accuracy:.4f}')
    print(f'  Loss     : {test_loss:.4f}')
    print(f'  MAE      : {mae:.4f}')

    return results


# =============================================================================
# 4. Confusion matrix
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names,
                           model_name='Model', normalize=True, save_path=None):
    """
    Plot confusion matrix.
    """
    cm       = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(8, 6))

    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh  = cm_plot.max() / 2.0
        plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    else:
        cm_plot = cm
        thresh  = cm.max() / 2.0
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title(f'{model_name} - Confusion Matrix {"(Normalized)" if normalize else ""}')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, f'{cm_plot[i, j]:.3f}', ha='center',
                     color='white' if cm_plot[i, j] > thresh else 'black')
        else:
            plt.text(j, i, f'{cm[i, j]:,}', ha='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label\nAccuracy={accuracy:.4f}; Misclassification={misclass:.4f}')

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        suffix = '_normalized' if normalize else '_raw'
        plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix{suffix}.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 5. Classification report
# =============================================================================

def _calculate_detailed_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return {
        'confusion_matrix':        cm,
        'accuracy_per_class':      (TP + TN) / (TP + FP + FN + TN),
        'precision':               np.where(TP + FP > 0, TP / (TP + FP), 0),
        'recall':                  np.where(TP + FN > 0, TP / (TP + FN), 0),
        'specificity':             np.where(TN + FP > 0, TN / (TN + FP), 0),
        'false_positive_rate':     np.where(FP + TN > 0, FP / (FP + TN), 0),
        'false_negative_rate':     np.where(TP + FN > 0, FN / (TP + FN), 0),
    }


def print_classification_report(y_true, y_pred, class_names, model_name='Model'):
    print(f'\n{model_name} - Detailed Classification Report:')
    print('=' * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    metrics = _calculate_detailed_metrics(y_true, y_pred, class_names)

    print('\nDetailed Metrics per Class:')
    print('-' * 60)
    for i, class_name in enumerate(class_names):
        prec = metrics['precision'][i]
        rec  = metrics['recall'][i]
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f'{class_name}:')
        print(f'  Accuracy    : {metrics["accuracy_per_class"][i]:.4f}')
        print(f'  Precision   : {prec:.4f}')
        print(f'  Recall      : {rec:.4f}')
        print(f'  Specificity : {metrics["specificity"][i]:.4f}')
        print(f'  F1-Score    : {f1:.4f}')
        print()


# =============================================================================
# 6. Sample predictions
# =============================================================================

def plot_sample_predictions(X_test, y_true, y_pred, predictions_proba,
                             class_names, model_name='Model',
                             num_samples=25, save_path=None):
    plt.figure(figsize=(15, 15))
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        pred_class = y_pred[idx]
        true_class = y_true[idx]
        confidence = predictions_proba[idx][pred_class]
        color      = 'green' if pred_class == true_class else 'red'

        plt.xlabel(
            f'Pred: {class_names[pred_class]} ({confidence:.2f})\nTrue: {class_names[true_class]}',
            color=color, fontsize=10
        )

        image = X_test[idx]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        plt.imshow(image)

    plt.suptitle(f'{model_name} - Sample Predictions', fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_sample_predictions.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()


# =============================================================================
# 7. Save model and results
# =============================================================================

def save_model_and_results(model, history, results, model_name, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model_path = os.path.join(save_dir, f'{model_name}.h5')
    model.save(model_path)
    print(f'Model saved to: {model_path}')

    # History
    hist_path = os.path.join(save_dir, f'{model_name}_history.npz')
    np.savez(hist_path, **history.history)
    print(f'History saved to: {hist_path}')

    # Results arrays
    res_path = os.path.join(save_dir, f'{model_name}_results.npz')
    np.savez(res_path, **{k: v for k, v in results.items() if isinstance(v, np.ndarray)})
    print(f'Results saved to: {res_path}')

    # Metrics JSON
    metrics_summary = {
        'model_name':    results['model_name'],
        'test_accuracy': float(results['test_accuracy']),
        'test_loss':     float(results['test_loss']),
        'mae':           float(results['mae']),
        'total_epochs':  len(history.history['loss']),
        'best_val_acc':  float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
    }
    json_path = os.path.join(save_dir, f'{model_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f'Metrics saved to: {json_path}')


# =============================================================================
# 8. Compare models
# =============================================================================

def compare_models(model_results, model_names, class_names, save_path=None):
    metrics_comparison = {
        'Model':    model_names,
        'Accuracy': [r['test_accuracy'] for r in model_results],
        'Loss':     [r['test_loss']     for r in model_results],
        'MAE':      [r['mae']           for r in model_results]
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy
    axes[0].bar(model_names, metrics_comparison['Accuracy'],
                color=['skyblue', 'lightcoral'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(metrics_comparison['Accuracy']):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # Loss
    axes[1].bar(model_names, metrics_comparison['Loss'],
                color=['lightgreen', 'gold'])
    axes[1].set_title('Model Loss Comparison')
    axes[1].set_ylabel('Loss')
    for i, v in enumerate(metrics_comparison['Loss']):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # MAE
    axes[2].bar(model_names, metrics_comparison['MAE'],
                color=['plum', 'orange'])
    axes[2].set_title('Model MAE Comparison')
    axes[2].set_ylabel('Mean Absolute Error')
    for i, v in enumerate(metrics_comparison['MAE']):
        axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'models_comparison.png'),
                    dpi=300, bbox_inches='tight')

    plt.show()

    print('\nModel Performance Comparison:')
    print('=' * 50)
    for i, name in enumerate(model_names):
        print(f'{name}:')
        print(f'  Accuracy : {metrics_comparison["Accuracy"][i]:.4f}')
        print(f'  Loss     : {metrics_comparison["Loss"][i]:.4f}')
        print(f'  MAE      : {metrics_comparison["MAE"][i]:.4f}')
        print()

    return metrics_comparison