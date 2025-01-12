
# CNN-LSTM/RNN for Image Captioning on the Flickr30k Dataset

## 📚 **Project Overview**
This repository contains the implementation of various **CNN-LSTM/RNN architectures** for generating image captions using the **Flickr30k Dataset**. The project explores feature extraction using pretrained convolutional neural networks (CNNs) such as **InceptionV3**, **ResNet50**, and **VGG16**, followed by caption generation using **LSTM** and **Bidirectional LSTM** models. The project also evaluates different decoding techniques such as **greedy search** and **beam search** for generating high-quality captions.

The focus of this project is to balance **computational efficiency** and **caption accuracy** by optimizing conventional CNN-LSTM frameworks rather than using more computationally expensive transformer-based models.

---

## 🧪 **Models Implemented**
The following model architectures were implemented in this project:

1. **VGG16 + Single LSTM**
2. **VGG16 + Bidirectional LSTM**
3. **VGG16 + Multiple LSTM Layers**
4. **ResNet50 + Single LSTM**
5. **ResNet50 + Multiple LSTM Layers**
6. **InceptionV3 + Single Bidirectional LSTM**
7. **InceptionV3 + Two Bidirectional LSTM Layers**

These models were evaluated based on their **BLEU scores**, **qualitative caption analysis**, and **computational efficiency**.

---

## 📂 **Dataset Used**
The project utilized the **Flickr30k Dataset**, which consists of **31,783 images** with **five human-generated captions** per image. This dataset is a widely used benchmark for image captioning tasks and covers a variety of everyday scenes and contexts.

---

## 🛠️ **Methodology**

### **1. Feature Extraction**
Feature extraction was done using pretrained CNN models:
- **InceptionV3**: Input size of 299x299 pixels, output feature vector of size 2048.
- **ResNet50**: Input size of 224x224 pixels, output feature vector of size 2048.
- **VGG16**: Input size of 224x224 pixels, output feature vector of size 4096.

The classification layers of these pretrained models were removed to use them as feature extractors.

### **2. Caption Tokenization and Padding**
- **Tokenization**: Captions were tokenized using the Keras `Tokenizer` to map each word to a unique integer.
- **Padding**: Captions were padded to ensure consistent input length for the LSTM models.

### **3. Model Architectures**
- **Encoder**: Pretrained CNN models (VGG16, ResNet50, InceptionV3) for image feature extraction.
- **Decoder**: LSTM or Bidirectional LSTM layers for sequential caption generation.

---

## 🚀 **Model Evaluation**

The following models were evaluated based on their performance:

| Model                             | BLEU Score | Caption Accuracy           | Training Time | Overall Performance |
|-----------------------------------|------------|----------------------------|---------------|---------------------|
| VGG16 + Single LSTM               | Lower      | Basic and short captions   | Fast          | Basic               |
| ResNet50 + Single LSTM            | Medium     | Balanced                   | Moderate      | Balanced            |
| InceptionV3 + Bidirectional LSTM  | Higher     | Most descriptive and detailed | Long          | Best                |

---

## 📝 **Results and Key Findings**
1. The **InceptionV3 + Bidirectional LSTM** model produced the most **accurate and contextually rich captions**.
2. **Greedy decoding** provided faster but less diverse captions, whereas **beam search decoding** improved the quality of captions at the cost of more computational resources.
3. The models performed well on **basic captioning tasks** but faced challenges with **complex images** and **repetitive captions**.

---

## 🧩 **Challenges and Limitations**
- **Repetitive Captions**: The models occasionally generated repetitive phrases.
- **No Attention Mechanism**: The models lacked an **attention mechanism**, which is known to improve caption quality.
- **Computational Cost**: The more complex models required **longer training times** and **higher computational resources**.

---

## 📈 **Future Work**
1. **Incorporating Attention Mechanisms**: Implement attention mechanisms to improve the focus on specific regions of an image while generating captions.
2. **Exploring Transformer Models**: Experiment with **Vision Transformers (ViT)** and **multi-modal transformers** for better performance on large datasets.
3. **Using Larger Datasets**: Train the models on more diverse datasets like **MS COCO** or **Visual Genome** to improve generalization.
4. **Optimizing for Real-Time Applications**: Apply techniques like **model pruning**, **quantization**, and **knowledge distillation** to optimize the models for deployment on mobile devices and edge systems.

---

## 📊 **Qualitative Examples**

| Image                             | Greedy Decoding                               | Beam Search Decoding                          |
|-----------------------------------|-----------------------------------------------|----------------------------------------------|
| ![Image 1](example1.jpg)          | A brown dog is running in the snow            | A brown dog runs through the snow            |
| ![Image 2](example2.jpg)          | A little girl is sliding down a slide         | A little girl is sliding down a red slide    |
| ![Image 3](example3.jpg)          | A dog is running through the water            | A black dog is running through the water of a body of water |

---

## 📂 **Repository Structure**
```
├── data/
│   └── flickr30k/
├── notebooks/
│   └── CNN-LSTM_RNN_with_Attention.ipynb
└── README.md
```

---

## 🛠️ **Technologies Used**
- **Python**
- **Keras**
- **TensorFlow**
- **NumPy**
- **Pandas**
- **Matplotlib**

---

Feel free to reach out if you have any questions about the repository or the project.
