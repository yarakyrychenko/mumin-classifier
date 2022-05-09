def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    from sklearn import metrics
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    accuracy = metrics.accuracy_score(y_true=labels, y_pred=preds)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true=labels, y_pred=preds, average=None)

    result = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return result

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    configuration = XLMRobertaConfig()
    model = XLMRobertaForSequenceClassification(configuration).from_pretrained("xlm-roberta-base")

    return model


