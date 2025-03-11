import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from models.initialize_model import initialize_custom_model
from utils.demo_utils import demo_emb_predicting, demo_data_loading, demo_unfreeze_strategy, print_task_title
from utils.data_utils import ner_label_map
from utils.train_utils import evaluate_classification, evaluate_ner, freeze_backbone, train_and_evaluate

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Task 1: Implementing Sentence Transformer
    print_task_title('Task 1: Sentence Transformer Implementation')

    # Load pre-trained BERT model and tokenizer
    pretrained_model = BertModel.from_pretrained("bert-base-uncased")
    custom_model = initialize_custom_model(pretrained_model)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Set models to evaluation mode
    custom_model.eval()
    pretrained_model.eval()

    # Example sentences for embedding demonstration
    sentences = [
        "Fetch Rewards is a great app!",
        "I love eating apples.",
        "I do enjoy oranges!"
    ]

    # Run embedding similarity demonstration
    demo_emb_predicting(custom_model, pretrained_model, tokenizer, sentences)

    # Task 2: Multi-Task Learning Expansion
    print_task_title('Task 2: Multi-Task Learning Expansion')

    # Initialize multi-task model
    model = initialize_custom_model(pretrained_model, model_type="multitask", pooling="mean", num_classes=2,
                                    num_entity_labels=17)
    model.eval()

    # Load classification and NER datasets
    classification_file = "data/raw/classification_sample.xlsx"
    ner_file = "data/raw/ner_sample.xlsx"
    classification_train_loader, classification_test_loader, ner_train_loader, ner_test_loader = demo_data_loading(
        classification_file, ner_file, 0.1)

    # Classification inference demo
    for batch_inputs, batch_labels in classification_test_loader:
        with torch.no_grad():
            classification_logits, _ = model(**batch_inputs)
            predicted_classes = torch.argmax(classification_logits, dim=-1)
            print("Predicted classes:", predicted_classes.tolist())
            print("True labels:", batch_labels.tolist())
        break

    # NER inference demo
    for batch_inputs, batch_labels in ner_test_loader:
        with torch.no_grad():
            _, ner_logits = model(**batch_inputs)
            predicted_labels = torch.argmax(ner_logits, dim=-1)
            print("Predicted NER labels:", predicted_labels.tolist())
            print("True NER labels:", batch_labels.tolist())
        break

    # Evaluate classification performance
    accuracy, report = evaluate_classification(model, classification_test_loader)
    print(f"Accuracy: {accuracy:.4f}")
    print("Report:\n", report)

    # Evaluate NER performance
    id_to_label = {v: k for k, v in ner_label_map.items()}
    ner_report = evaluate_ner(model, ner_test_loader, id_to_label)
    print("Report:\n", ner_report)

# Task 3: Freezing and Unfreezing Layers
print_task_title('Task 3: Training Considerations')

# Demonstrate different unfreeze strategies
demo_unfreeze_strategy(model)

# Task 4: Training Implementation
print_task_title('Task 4: Training Loop Implementation (BONUS)')

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze backbone layers while keeping classification & NER heads trainable
freeze_backbone(model)

# Define loss functions for classification and NER
criterion_classification = nn.CrossEntropyLoss()
criterion_ner = nn.CrossEntropyLoss(ignore_index=-100)

# Define optimizer (only update trainable parameters)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

# ID-to-label mapping for NER evaluation
id_to_label = {v: k for k, v in ner_label_map.items()}

# Train model with separate task training
print("Training with Separate Approach...")
train_and_evaluate(model, classification_train_loader, ner_train_loader,
                   classification_test_loader, ner_test_loader,
                   criterion_classification, criterion_ner,
                   optimizer, device, id_to_label,
                   combine_tasks=False)
