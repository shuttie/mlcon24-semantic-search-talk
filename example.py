from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer("distilbert-base-nli-mean-tokens")


train_examples = [
    InputExample(texts=["tomato", "fresh roma tomatoes", "Heinz mayo"]),
    InputExample(texts=["tomato", "cherry tomato 200g", "Heinz tomato ketchup"]),
]

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
