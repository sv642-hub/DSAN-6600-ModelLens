import torch
from data import number_to_words, TOKEN_TO_ID, ID_TO_TOKEN, END_ID
from model import ArithmeticTransformer

model = ArithmeticTransformer()
model.load_state_dict(torch.load('trained_model_new.pt', map_location='cpu'))
model.eval()

tests = [
    "three plus four equals",
    "one hundred ninety nine plus two hundred ninety three equals",
    "five hundred plus three hundred equals",
    "forty seven plus twenty five equals",
]

for test in tests:
    tokens = test.split()
    ids = [TOKEN_TO_ID[t] for t in tokens]
    input_ids = torch.tensor([ids])
    generated = []
    for _ in range(6):
        logits = model(input_ids)
        next_id = logits[0, -1, :].argmax().item()
        if next_id == END_ID:
            break
        generated.append(ID_TO_TOKEN[next_id])
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)
    print(f"{test.replace(' equals', '')} = {''.join(generated)}")