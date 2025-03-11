class CombinedTaskIterator:
    """
    Iterator that combines classification and NER data loaders.
    Automatically resets each loader when exhausted.
    """
    def __init__(self, classification_loader, ner_loader):
        self.classification_loader = classification_loader
        self.ner_loader = ner_loader
        self.class_iter = iter(classification_loader)
        self.ner_iter = iter(ner_loader)

    def __iter__(self):
        return self

    def __next__(self):
        """Fetches next batch from each loader, resetting iterators when needed."""
        try:
            class_batch = next(self.class_iter)
        except StopIteration:
            self.class_iter = iter(self.classification_loader)
            class_batch = next(self.class_iter)

        try:
            ner_batch = next(self.ner_iter)
        except StopIteration:
            self.ner_iter = iter(self.ner_loader)
            ner_batch = next(self.ner_iter)

        return {"classification": class_batch, "ner": ner_batch}
