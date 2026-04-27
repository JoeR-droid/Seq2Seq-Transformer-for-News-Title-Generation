# Seq2Seq-Transformer-for-News-Title-Generation
In this project I built a encoder-decoder transformer architecture from scratch in PyTorch including multi-head attention, positional encoding, and masked self-attention. I trained this on 280k CNN/DailyMail articles across 7 epochs on dual NVIDIA T4 GPUs. At first training I ran into repeition collapse so I decided to add a repetion penalty . I also decided to train for 7 epochs which led me to this final model.
## Pretrained Weights
Download: https://www.kaggle.com/code/joercharles/title-generator/output?select=transformer_title_generator.pt

To load:
model = Transformer(vocab_size=32100, d_model=128, num_heads=4, d_ff=256, num_layers=2)
model.load_state_dict(torch.load("transformer_title_generator.pt", map_location="cpu"))
