# [BioCLIP] (https://imageomics.github.io/bioclip/): A Vision Foundation Model for the Tree of Life

Notes by Juliana Gómez

## Intro
- Autogregressive language model (vision transformer ViT-B/16) used to predict next hierarchichal order. i.e: Order can be predicted from kingdom, phyum, class. 
- Training done over 450K class labels
- BioCLIP is the first large-scale multimodal model for general biology questions on images
- TreeOfLife-10M: largest and most diverse dataset of biology images (10.4M images, 454103 unique classes)
- Input: image, Output: Linnaean classification up to desired level. 

## Experiments
**Baselines**: 

- [CLIP](https://github.com/openai/CLIP) 
- [OpenCLIP](https://github.com/mlfoundations/open_clip) 
- [iNat-only model](https://huggingface.co/imageomics/bioclip-vit-b-16-inat-only) (like BioCLIP but only trained on iNat21)

**Experiments**

- Zero-shot classification: BioCLIP outperforms both general-domain baselines and iNat-only ViT model on accuracy by 17% (24% accuracy for plants - 72% accuracy for birds). Especially better at lower levels (i.e species) of classification. 
- One-shot:
- Five-shot: 


## Dataset
**TreeOfLife-10M**

- 10M combined images from: 
1. iNaturalist: 2.7 M images of 10K taxa
2. BIOSCAN-1M: to learn extremely fine-grained visual representations for insects. 1M images of 494 families. 7831 unique classes. 
3. Encyclopedia of Life (2023): 6.6 M images, 448910 unique classes.  

- 450K + species (representing 22% of total described species mentioned by IUCN)
- Weights released for public use. 

## Modeling
- BioCLIP is initialized from OpenAI's public CLIP checkpoint (ViT-B/16 vision transformer image encoder and 77-token causal autoregressive transformer text encoder) and pre-trained on TreeOfLife-10M.   
- CLIP trains 2 uni-modal embedding models: a vision encoder and a text encoder, to (1) maximize feature similarity between positive (image, text) pairs and (2) minimize feature similarity between (image,text) pairs, where positive pairs are from the training data and negative pairsa re all other possible (image, text) pairings in a batch. 
- CLIP accepts free-form text, so training for BioCLIP included pairing each input image with a text randomly samples from all of its available text types. 


### Training
Training and evaluation code is publicly available

*Model* 

- BioCLIP is initialized from OpenAI's public CLIP checkpoint (ViT-B/16 vision transformer image encoder and 77-token causal autoregressive transformer text encoder) and pre-trained on TreeOfLife-10M.  

*Hyperparameters - BioCLIP* 

- 100 epochs
- Cosine learning rate schedule
- 8 NVIDIA A100-80GB GPUs over 2 nodes
- Global batch size of 32,768

*Hyperparameters - iNat21 and multiple ablation models on randomly samples TOL*

- 100 epochs
- Cosine learning rate schedule
- 4 NVIDIA A100-80GB GPUs over 1 node
- Global batch size of 16,384

## Inference

*Classification tasks*

1-8: Biologically-relevant tasks (Plankton, Insects, Insects 2, PlantNet, Fungi, PlantVillage, Medicinal Leaf, and PlantDoc datasets) from Meta-Album (dataset collection for meta-learning) 
9.Birds 525
10. RARE SPECIES task: collect 25K spp on IUCN red list classified as NT, VU, EN, CE or EW, relect 100 spp with at least 30 images from this list from EOL dataset and remove them from != to create a RARE SPECIES test sep with 30 images per spp. 

- Cover all multi-celled kingdoms, and have diverse image distribution (photos, microscope images, drawings, museum specimens)

*Zero-shot learning*

- Follows same procedure as CLIP

*Few-shot learning*

- SimpleShot with nearest-centroid classifier

*k-shot learning*

1. Randomly sample $k$ examples for each class and obtain the image embedding from the visual encoder of pre-trained model
2. Compute average feature vector of $k$ embeddings as centroid. 
3. Remaining sampels = testing
4. Mean subtraction and L2-noramilzation to each centroid and test feature vector
4. Choose class with the nearest centroid to the test vector as the prediction. 

Repeat each few-shot experiment 5 times with different random seeds and report mean accuracy. 

*Class labels*

Whenever the preferred label type is not available, we use labels that come with the dataset. 
**CLIP:** common names
**OpenCLIP:** common names
**BioCLIP:** taxonomic + common names 


## Results 

- BioCLIP outperforms both baseline CLIP models and iNat21-CLIP model,e specially at zero-shot classification, especially on rare species task. 

### Text types --> generalization

- Zero-shot generalization using Rare Species dataset with every text type
- Randomly use 1/5 different captions for each image during training, rather ahan a single fixed caption.
- Mixed text type strategy as well. 

*Caption types:*

1. Common name
2. Scientific name
3. Tax (concatenate and flatten all available taxonomical levels)
4. Scientific + Common name
5. Tax + Common name: strongest performance
6. Mixed text type strategy: best to generalize.  


### CLIP objective

- FSL: Multitask hierarchical training objective to predict the labels from kingdom down to species, using cross entropy for each level of the tazonomy, then summing those losses. **Pseudocode available**
- Massively outperforms both baselines (two ViT-B/16 models using cross-entropy classification loss)
 
 
### BioCLIP to classify more than species

- PlantVillage and PlantDoc datasets to classify both plant and disease: outperforms baselines on diseases in zero-shot. 


--------

# Questions (JGC)

|Page|Question|Answer|
|--|--|--|
|4|Table 1: what do the 7,831 unique classes represent? Complete taxonomical classification?||
|5|Not sure what the final text input to CLIP is, because column 1 says scientific name, column 2 says common name||
|5|What is SimpleShot?||
|5|What is meant by "All the examples left in the dataset are used for testing"?||
|6|I find it hard to believe that the authors obtained common names for all species in the rare dataset||


# Ideas (JGC)

- Use BioCLIP approach for Ribbit, by combining spectrogram with text labels to see how we do.  



 




