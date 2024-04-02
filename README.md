### Repository Documentation: AI Model for Recognizing Screenshots

#### Overview
This repository contains an AI model trained to distinguish between screenshots and real pictures. The model is based on an improved version of the VGG architecture and has been trained and evaluated on a dataset of screenshots and real images.

- **[Armin Sabourmoghaddam](https://github.com/Arminsbss)** Developed the full architecture of the Neural network.
- **[Pooria Rabie](https://github.com/PooriaRabie)** Improved the accuracy.

#### Demo Video

https://github.com/Arminsbss/binary-classification-screenshots/assets/95970651/35b58fd1-d1d9-4092-b314-ee929f0e68e8

[[Download demo video](https://armin-sabour.epidemic-calculator.info/assets/3979677129529659717%20-210075.mp4)]

#### Model Performance
The model achieved the following classification metrics on a test dataset:

- **Accuracy:** 90%
- **Precision:** 
  - Class 0 (Real Pictures): 96%
  - Class 1 (Screenshots): 85%
- **Recall:**
  - Class 0 (Real Pictures): 81%
  - Class 1 (Screenshots): 97%
- **F1-score:** 
  - Class 0 (Real Pictures): 88%
  - Class 1 (Screenshots): 91%

#### Dataset
The dataset used for training and evaluation consists of screenshots and real pictures. It comprises 1383 images, with 765 screenshots and 638 real pictures.

#### Model Architecture
The model architecture is based on an improved version of the VGG16 architecture, fine-tuned for the task of recognizing screenshots. Key components of the model include:
- Pre-trained VGG16 backbone
- Custom fully connected layers
- Binary cross-entropy loss function
- Adam optimizer with a learning rate of 0.001

#### Training
The model was trained for 2 epochs on the dataset using a batch size of 64. Training performance was monitored using accuracy as the primary metric. Early stopping with a patience of 10 epochs was employed to prevent overfitting.

#### Repository Structure
The repository is structured as follows:
- `README.md`: Overview of the repository, usage instructions, and model performance.
- `requirements.txt`: List of dependencies required to run the code.
- `model_training.ipynb`: Jupyter Notebook containing the code for training the AI model.
- `model_evaluation.ipynb`: Jupyter Notebook for evaluating the trained model on test data.
- `vgg_model.py`: Python script containing the code for the VGG-based model architecture.
- `predict.py`: Python script for making predictions using the trained model.
- `data/`: Directory containing the dataset used for training and evaluation.
- `saved_models/`: Directory containing saved model weights.

#### Usage
To use the AI model for recognizing screenshots, follow these steps:
1. Clone the repository:
   ```bash
   git clone <https://github.com/Arminsbss/binary-classification-screenshots/tree/main>
   ```
2. Navigate to the repository directory:
   ```bash
   cd AI-Model-Screenshot-Recognition
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the prediction script:
   ```bash
   python predict.py <image_path>
   ```
   Replace `<image_path>` with the path to the image you want to classify.

#### GUI Application
Additionally, a simple GUI application (`gui_application.py`) has been provided for easy inference using the trained model. To run the GUI application, execute the following command:
```bash
python gui_application.py
```
Select an image using the "Choose Image" button, and the application will display whether the image is a screenshot or a real picture.

#### Conclusion
This repository serves as a demonstration of using machine learning techniques, specifically deep learning models, for classifying screenshots. The provided model can be further improved and customized for specific applications or domains.

For any inquiries or issues, please contact [[repository owner's name](https://github.com/Arminsbss)] at [contact email].

---

Feel free to customize this documentation according to your specific needs and preferences. Make sure to replace placeholders like `<repository_url>`, `<image_path>`, and `[repository owner's name]` with the actual values.
