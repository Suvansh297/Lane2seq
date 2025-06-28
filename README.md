

```markdown
# 🛣️ Lane2Seq – Lane Detection Using Transformers

Lane2Seq is a deep learning project that detects **lane lines in road images** using a special kind of model called a **transformer**. Instead of detecting lanes as pixel maps, this model **predicts a sequence of tokens** (like how language models predict text) to represent lane positions.

This project is based on the [TuSimple lane detection dataset](https://github.com/TuSimple/tusimple-benchmark) and uses a **Vision Transformer (ViT)** to understand the image and a transformer decoder to output the lane information as a sequence.

---

## 🧠 What Has Been Done

- ✅ Used a **Vision Transformer (ViT)** encoder pretrained using masked autoencoding.
- ✅ Designed a **transformer decoder** to generate tokenized lane positions.
- ✅ Converted lane coordinates into a **token sequence** using a custom tokenizer.
- ✅ Trained the model on the **TuSimple lane dataset**.
- ✅ Added **data augmentation**: random flipping, rotation, scaling, and translation.
- ✅ Implemented **evaluation metrics**: precision, recall, and F1 score.
- ✅ Included **visualizations** for model predictions.
- ✅ Wrote a **configurable pipeline** for training, testing, and debugging.

---

## 🛠️ What Has Been Used

- Python & PyTorch
- `transformers` (HuggingFace ViT)
- `safetensors` (to load pretrained ViT safely)
- OpenCV, Pillow (for image handling)
- `tqdm`, `yaml`, `numpy`

---

## 📦 Folder Structure


lane2seq\_project/
├── configs/           # YAML configuration
├── models/            # Encoder (ViT) + Decoder
├── datasets/          # Dataset loading and tokenization
├── utils/             # Augmentations, tokenizer, visualizer
├── train1.py          # Main training script
├── inference.py       # Run predictions and save images
├── evaluation.py      # Evaluate model output
├── test\_tokenizer.py  # Visual test for tokenizer



---

## 🧪 How It Works

1. **Images** are passed through a **Vision Transformer (ViT)** that extracts meaningful features.
2. These features go into a **Transformer Decoder** which outputs a sequence of tokens.
3. Tokens represent **lane points**, decoded using a tokenizer.
4. You can then **draw these points** on the original image to see the predicted lanes!

---

## 🖼️ Example Results

![clips_0530_1492626153155598528_0_20](https://github.com/user-attachments/assets/406aae32-d74f-4d7f-8e6c-af3c7d15139c)



![Lane Prediction Example](path/to/your/image.png)

---

## 🚀 How to Use

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/lane2seq_project.git
cd lane2seq_project


### 2. Install Requirements

```bash
pip install -r requirements.txt


You’ll need `torch`, `transformers`, `opencv-python`, `Pillow`, `safetensors`, etc.

### 3. Prepare the Dataset

Download the [TuSimple dataset](https://github.com/TuSimple/tusimple-benchmark/issues/3) and place it like this:


archive/TUSimple/
├── train_set/
│   ├── clips/
│   └── label_data_0313.json, ...
├── test_set/
│   ├── clips/
│   └── test_label.json


### 4. Train the Model

```bash
python train1.py


* Best model saved to: `checkpoints/best_model.pth`
* Logs are stored in `logs/train.log`

### 5. Run Inference

```bash
python inference.py


* Outputs: `results/inference/*.json` (lane data) and `*.png` (visual images)

### 6. Evaluate the Model

```bash
python evaluation.py


* Outputs precision, recall, F1 score
* Saves results in `results/inference/evaluation_results.json`

### 7. Debug the Tokenizer (Optional)

```bash
python test_tokenizer.py


* Saves side-by-side visualizations of decoded vs original lanes
* Output in `test_outputs_anchor/`

---

## 📌 Configuration

All settings (image size, vocab size, paths, augmentation, etc.) are in:


configs/config.yaml


Change things like number of epochs, checkpoint paths, or augmentation strength from there.

---

## 📈 Metrics Used

* **Precision**: How many predicted lanes are correct?
* **Recall**: How many ground truth lanes are found?
* **F1 Score**: Balance between precision and recall.

---

## 🧩 Format Support

* Supports `anchor`-based format (sequence of points)
* Tokenizer also allows `parameter` and `segmentation` formats (can be extended)

---


