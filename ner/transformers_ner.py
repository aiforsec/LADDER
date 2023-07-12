from entity_extraction import EntityExtraction
import tner


class TransformersNER(EntityExtraction):
    """
    Entity extraction using Transformers
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def train(self):
        config = self.config
        checkpoint_dir = config.get('checkpoint_dir', './ckpt')
        dataset = config.get('dataset', 'cyber')
        transformers_model = config.get('transformers_model', 'xlm-roberta-base')
        random_seed = config.get('random_seed', 1)
        lr = config.get('lr', 1e-5)
        epochs = config.get('epochs', 12)
        warmup_step = config.get('warmup_step', 0)
        weight_decay = config.get('weight_decay', 1e-7)
        batch_size = config.get('batch_size', 32)
        max_seq_length = config.get('max_seq_length', 128)
        fp16 = config.get('fp16', False)
        max_grad_norm = config.get('max_grad_norm', 1)
        lower_case = config.get('lower_case', False)
        num_worker = config.get('num_worker', 0)
        cache_dir = config.get('cache_dir', None)

        trainer = tner.TrainTransformersNER(checkpoint_dir=checkpoint_dir,
                                            dataset=dataset,
                                            transformers_model=transformers_model,
                                            random_seed=random_seed,
                                            lr=lr,
                                            epochs=epochs,
                                            warmup_step=warmup_step,
                                            weight_decay=weight_decay,
                                            batch_size=batch_size,
                                            max_seq_length=max_seq_length,
                                            fp16=fp16,
                                            max_grad_norm=max_grad_norm,
                                            lower_case=lower_case,
                                            num_worker=num_worker,
                                            cache_dir=cache_dir)

        trainer.train(monitor_validation=True)

    def get_entities(self, text):
        pass
