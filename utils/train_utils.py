from sklearn.metrics import accuracy_score, classification_report
from datasets.combined_task_iterator import CombinedTaskIterator
import torch
from tqdm import tqdm
from seqeval.metrics import classification_report as seqeval_report
from seqeval.scheme import IOB2

def evaluate_classification(model, dataloader, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}  # Move inputs to device
            batch_labels = batch_labels.to(device)  # Move labels to device

            classification_logits, _ = model(**batch_inputs)
            preds = torch.argmax(classification_logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

    # Ensure at least two classes are present
    unique_classes = set(all_preds) | set(all_labels)

    if len(unique_classes) < 2:
        print(f"⚠️ Warning: Only one class detected in predictions: {unique_classes}. Adjusting labels.")
        target_names = [str(cls) for cls in unique_classes]  # Dynamically assign class names
    else:
        target_names = ["negative", "positive"]

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names)

    return accuracy, report


# Evaluate Named Entity Recognition (NER) performance
def evaluate_ner(model, dataloader, id_to_label, device=torch.device('cpu')):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}  # Move to device
            batch_labels = batch_labels.to(device)  # Move labels to device

            # Forward pass
            _, ner_logits = model(**batch_inputs)
            preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()
            labels = batch_labels.cpu().numpy()

            for i in range(preds.shape[0]):  # Iterate over batch
                seq_preds = []
                seq_labels = []
                for j in range(preds.shape[1]):  # Iterate over sequence length
                    if labels[i, j] != -100:  # Ignore padding tokens
                        seq_preds.append(id_to_label.get(preds[i, j], "O"))  # Default to "O"
                        seq_labels.append(id_to_label.get(labels[i, j], "O"))  # Default to "O"

                if seq_preds and seq_labels:
                    all_preds.append(seq_preds)
                    all_labels.append(seq_labels)

    # **Debugging Outputs**
    print(f"\nNER Evaluation Debugging:")
    print(f"Total Sentences Evaluated: {len(all_preds)}")
    if all_preds:
        print(f"First Prediction Example: {all_preds[0]}")
    if all_labels:
        print(f"First Label Example: {all_labels[0]}")

    # **Check if predictions exist before calling seqeval_report**
    if not all_preds or not all_labels:
        print("⚠️ Warning: No valid NER predictions found. Skipping classification report.")
        return "No valid NER predictions available."

    report = seqeval_report(all_labels, all_preds, mode='strict', scheme=IOB2)
    return report


# Function to control layer freezing/unfreezing
def set_requires_grad(model, unfreeze_option=None):
    """
    Sets the requires_grad attribute for model parameters based on unfreeze strategy.

    Options:
    - 'heads_only': Unfreezes only classification & NER heads.
    - 'all_adaptive': Unfreezes all layers.
    - 'output_layers': Unfreezes last 3 encoder layers & output heads.
    - 'all': Unfreezes the entire model.
    """
    for param in model.parameters():
        param.requires_grad = False  # Freeze all parameters by default
    print(f"All model parameters frozen.")

    if unfreeze_option == 'heads_only':
        for name, param in model.named_parameters():
            if 'classification_head' in name or 'ner_head' in name:
                param.requires_grad = True
                print(f"Unfrozen: {name}")
        print(f"Unfreezing strategy: Heads Only (Only classification & NER heads are trainable)")

    elif unfreeze_option == 'all_adaptive':
        for param in model.parameters():
            param.requires_grad = True
        print(f"Unfrozen: Entire Model (All Layers)")

    elif unfreeze_option == 'output_layers':
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['encoder_layers.9', 'encoder_layers.10', 'encoder_layers.11',
                                               'classification_head', 'ner_head']):
                param.requires_grad = True
                print(f"Unfrozen: {name}")
        print(f"Unfreezing strategy: Output Layers (Last 3 encoder layers + output heads)")

    elif unfreeze_option == 'all':
        for param in model.parameters():
            param.requires_grad = True
        print(f"Unfrozen: Entire Model (All Layers)")

    else:
        print(f"Invalid unfreeze_option: {unfreeze_option}. Defaulting to all layers frozen.")


# Freeze backbone while keeping output heads trainable
def freeze_backbone(model):
    """Freezes model backbone while allowing classification & NER heads to be trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = ('classification_head' in name or 'ner_head' in name)


# Training and evaluation function
def train_and_evaluate(model, classification_train_loader, ner_train_loader,
                       classification_test_loader, ner_test_loader,
                       criterion_classification, criterion_ner,
                       optimizer, device, id_to_label, combine_tasks=False, epochs=3, loss_weight=0.5):
    """Trains and evaluates the multi-task model in either combined or separate mode."""
    model.to(device)

    # Evaluate before training
    print("\nMetrics Before Training:")
    print("Classification:")
    accuracy_before, report_before = evaluate_classification(model, classification_test_loader, device)
    print(f"Accuracy: {accuracy_before:.4f}")
    print("Report:\n", report_before)

    print("NER:")
    ner_report_before = evaluate_ner(model, ner_test_loader, id_to_label, device)
    print("Report:\n", ner_report_before)

    model.train()

    # Combined training mode
    if combine_tasks:
        print("\nTraining in Combined Mode...")
        combined_iterator = CombinedTaskIterator(classification_train_loader, ner_train_loader)
        for epoch in range(epochs):
            total_loss = 0.0
            for i, batch in enumerate(tqdm(combined_iterator, desc=f"Combined Epoch {epoch + 1}/{epochs}",
                                           total=len(classification_train_loader))):
                optimizer.zero_grad()

                # Classification batch
                class_inputs, class_labels = batch["classification"]
                class_inputs = {k: v.to(device) for k, v in class_inputs.items()}
                class_labels = class_labels.to(device)
                class_logits, _ = model(**class_inputs)
                loss_class = criterion_classification(class_logits, class_labels)

                # NER batch
                ner_inputs, ner_labels = batch["ner"]
                ner_inputs = {k: v.to(device) for k, v in ner_inputs.items()}
                ner_labels = ner_labels.to(device)
                _, ner_logits = model(**ner_inputs)
                loss_ner = criterion_ner(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))

                # Combined loss
                loss = loss_weight * loss_class + (1 - loss_weight) * loss_ner
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"\nCombined Epoch {epoch + 1} Average Loss: {total_loss / len(classification_train_loader):.4f}")

    # Separate training mode
    else:
        print("\nTraining in Separate Mode...")

        # Train Classification
        print("Training Classification Head...")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_inputs, batch_labels in tqdm(classification_train_loader, desc=f"Classification Epoch {epoch + 1}/{epochs}"):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                class_logits, _ = model(**batch_inputs)
                loss = criterion_classification(class_logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"\nClassification Epoch {epoch + 1} Average Loss: {total_loss / len(classification_train_loader):.4f}")

        # Train NER
        print("\nTraining NER Head...")
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_inputs, batch_labels in tqdm(ner_train_loader, desc=f"NER Epoch {epoch + 1}/{epochs}"):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                _, ner_logits = model(**batch_inputs)
                loss = criterion_ner(ner_logits.view(-1, ner_logits.size(-1)), batch_labels.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"\nNER Epoch {epoch + 1} Average Loss: {total_loss / len(ner_train_loader):.4f}")

    # Evaluate after training
    print("\nMetrics After Training:")
    print("Classification:")
    accuracy_after, report_after = evaluate_classification(model, classification_test_loader)
    print(f"Accuracy: {accuracy_after:.4f}")
    print("Report:\n", report_after)

    print("NER:")
    ner_report_after = evaluate_ner(model, ner_test_loader, id_to_label)
    print("Report:\n", ner_report_after)
