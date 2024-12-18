# Abstractive-Text-Summarization
Bidirectional Autoregressive Transformer (BART) is a Transformer-based encoder-decoder model, often used for sequence-to-sequence tasks like summarization and neural machine translation. BART is pre-trained in a self-supervised fashion on a large text corpus. During pre-training, the text is corrupted and BART is trained to reconstruct the original text (hence called a "denoising autoencoder"). Some pre-training tasks include token masking, token deletion, sentence permutation (shuffle sentences and train BART to fix the order), etc.

# Sources and references:
- [Keras: BART Example]([https://keras.io/examples/vision/nerf/](https://keras.io/examples/nlp/abstractive_summarization_with_bart/))  
- [BART article on Arxiv]([https://github.com/bmild/nerf](https://arxiv.org/abs/1910.13461))  
