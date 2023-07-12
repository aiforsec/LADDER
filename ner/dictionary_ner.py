from entity_extraction import EntityExtraction


class DictionaryNER(EntityExtraction):
    """
    Entity extraction using Dictionary
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def train(self):
        pass

    def get_entities(self, text):
        pass
