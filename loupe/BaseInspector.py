class BaseInspector:
    def __init__(self, model, rules):
        self.model = model
        self.rules = rules

    def inspect(self):
        pass

    def redact(self):
        pass
