```python
import kaggle
import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


kaggle.api.authenticate()

# https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/data?select=data

path = kaggle.api.dataset_download_files('smaranjitghose/corn-or-maize-leaf-disease-dataset', path=".", unzip=True)

data_dir = './data'
train_dir = './train'
test_dir = './test'
val_dir = './val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)

        # Ensure class subdirectories exist in train, val, and test folders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Copy images to train, val, and test directories
        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))

        for image in val_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))

        for image in test_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(test_dir, class_name, image))


train_data_path = "./train"
validation_data_path = "./val"
test_data_path = "./test"

training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

training_data = training_datagen.flow_from_directory(train_data_path, 
                                      target_size=(150, 150), 
                                      batch_size=32,
                                      class_mode='binary')  
 
print("Indices ",training_data.class_indices)

valid_datagen = ImageDataGenerator(rescale=1./255)
 
# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')

# Model Save Path
model_path = "saved_model.h5"
# Save only the best model
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Define Model
model = Sequential([
    Conv2D(32, kernel_size=3, input_shape=[150, 150, 3], activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(4, activation='softmax')  # Assuming 6 output classes
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train Model
history = model.fit(training_data, epochs=10, verbose=1, 
          validation_data=valid_data, callbacks=callbacks_list)
# Save Final Model
model.save(model_path)
print("Training Complete. Model Saved as 'maize_pred.h5'")

# Testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

### Overview of the Project

1. **Objective of the Project**  
    The primary goal of this project is to develop a deep learning model capable of accurately identifying and classifying maize leaf diseases. This can assist farmers in early detection and treatment.

2. **Dataset Utilization**  
    The project uses a publicly available dataset from Kaggle, which contains images of maize leaves with various diseases. The dataset is preprocessed and split into training, validation, and testing sets for model training.

3. **Model Architecture**  
    A Convolutional Neural Network (CNN) is designed with multiple layers, including convolutional, pooling, and dense layers. The architecture is optimized for image classification tasks with a focus on accuracy.

4. **Data Augmentation**  
    To improve model generalization, data augmentation techniques such as rotation, zoom, and horizontal flipping are applied. This ensures the model performs well on unseen data and reduces overfitting.

5. **Evaluation and Deployment**  
    The model's performance is evaluated using accuracy metrics on the test dataset. The trained model is saved for future use, enabling deployment in real-world scenarios for disease detection.

### Problem Statement

1. **Agricultural Challenges**  
    Maize is a staple crop worldwide, and its productivity is often threatened by various leaf diseases. Early detection and classification of these diseases are crucial for minimizing crop losses.

2. **Manual Inspection Limitations**  
    Traditional methods of disease identification rely on manual inspection, which is time-consuming, error-prone, and requires expert knowledge.

3. **Need for Automation**  
    There is a growing need for automated systems that can accurately identify maize leaf diseases to assist farmers and agricultural experts.

4. **Impact on Food Security**  
    Unchecked diseases can lead to significant yield losses, directly impacting food security and the livelihoods of farmers.

5. **Leveraging Deep Learning**  
    Advances in deep learning provide an opportunity to develop robust models capable of identifying diseases from maize leaf images with high accuracy, offering a scalable solution to this problem.

### Objectives of the Project

1. **Accurate Disease Classification**  
    Develop a deep learning model to classify maize leaf diseases with high accuracy, aiding in early detection and treatment.

2. **Scalable Solution**  
    Create a scalable and automated system that can be deployed in real-world agricultural settings to assist farmers.

3. **Improved Crop Management**  
    Provide a tool to help farmers make informed decisions, reducing crop losses and improving overall productivity.

4. **Accessible Technology**  
    Ensure the solution is accessible and user-friendly for farmers and agricultural experts, even in resource-constrained environments.

5. **Contribution to Food Security**  
    Support global food security by minimizing the impact of maize leaf diseases on crop yields.

### Scope of the Project

1. **Disease Coverage**  
    The project focuses on identifying multiple maize leaf diseases, ensuring comprehensive coverage of common issues affecting maize crops.

2. **Dataset Utilization**  
    Leverages a publicly available dataset for training, validation, and testing, ensuring the model is trained on diverse and real-world data.

3. **Model Deployment**  
    The trained model can be deployed on various platforms, including mobile and web applications, for easy accessibility by farmers.

4. **Scalability**  
    The solution is designed to handle large datasets and can be scaled to include additional crops and diseases in the future.

5. **Real-World Application**  
    Provides a practical tool for farmers and agricultural experts, enabling real-time disease detection and decision-making in the field.

### Methodology Adopted

1. **Convolutional Neural Network (CNN)**  
    The project employs a CNN architecture, which is highly effective for image classification tasks. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images, making them ideal for identifying patterns in maize leaf images.

2. **Data Augmentation**  
    Data augmentation techniques such as rotation, zoom, and flipping are applied to artificially expand the dataset. This helps the model generalize better by learning from a diverse set of images, reducing overfitting and improving performance on unseen data.

3. **Adam Optimizer**  
    The Adam optimizer is used for training the model due to its efficiency and adaptability. It combines the advantages of both RMSProp and SGD with momentum, ensuring faster convergence and better handling of sparse gradients in the dataset.

### Background Study

1. **Deep Learning in Agriculture**  
    Deep learning has revolutionized various fields, including agriculture, by enabling automated analysis of complex data such as images. It has been widely adopted for tasks like crop disease detection, yield prediction, and precision farming.

2. **Convolutional Neural Networks (CNNs)**  
    CNNs are a class of deep learning models specifically designed for image processing tasks. They have shown remarkable success in identifying patterns and features in agricultural datasets, making them ideal for disease classification.

3. **Importance of Data Augmentation**  
    Data augmentation is a critical technique in deep learning to enhance model performance. By artificially increasing the diversity of training data, it helps models generalize better and reduces the risk of overfitting.

4. **Challenges in Disease Detection**  
    Identifying crop diseases from images is challenging due to variations in lighting, angles, and disease symptoms. Leveraging robust models and preprocessing techniques is essential to overcome these challenges and achieve high accuracy.

### Existing System

1. **Manual Inspection**  
    Traditional methods rely on manual inspection by agricultural experts or farmers. This approach is time-consuming, prone to human error, and requires significant expertise, making it unsuitable for large-scale or real-time applications.

2. **Rule-Based Image Processing**  
    Some systems use rule-based image processing techniques, such as edge detection and color thresholding, to identify diseases. These methods lack adaptability and struggle with complex patterns or variations in lighting and angles.

3. **Basic Machine Learning Models**  
    Basic machine learning models, such as Support Vector Machines (SVM) or k-Nearest Neighbors (k-NN), are sometimes used for disease classification. However, these models require manual feature extraction and are less effective in handling large datasets or complex image features.

4. **Mobile Applications with Limited Accuracy**  
    Some mobile applications provide disease detection features but often rely on pre-trained models with limited accuracy. These systems may not generalize well to diverse datasets or provide reliable results in real-world scenarios.

### Proposed System
1. **Automated Disease Detection**  
    The proposed system leverages a deep learning-based CNN model to automate the detection and classification of maize leaf diseases, reducing reliance on manual inspection.

2. **High Accuracy and Robustness**  
    By utilizing advanced data augmentation techniques and a well-optimized CNN architecture, the system ensures high accuracy and robustness in identifying diseases across diverse conditions.

3. **Real-Time Application**  
    The system is designed for real-time deployment, enabling farmers and agricultural experts to quickly identify diseases and take timely action to mitigate crop losses.

4. **Scalable and Extensible**  
    The solution is scalable to handle large datasets and can be extended to include additional crops and diseases, making it adaptable for broader agricultural applications.

### Advantages of Proposed System
1. **Improved Accuracy**  
    The deep learning-based approach ensures higher accuracy in disease detection compared to traditional methods, reducing false positives and negatives.

2. **Time Efficiency**  
    Automated detection significantly reduces the time required for disease identification, enabling quicker decision-making for farmers.

3. **Cost-Effective**  
    Eliminates the need for expert consultations or expensive manual inspections, making it a cost-effective solution for farmers.

4. **User-Friendly Interface**  
    The system can be integrated into mobile or web applications, providing an intuitive and accessible interface for users with minimal technical expertise.

5. **Adaptability to Diverse Conditions**  
    The use of data augmentation and robust model training ensures the system performs well under varying environmental conditions and image qualities.

### Steps Involved in Developing the Project

1. **Dataset Acquisition**  
    - Download the maize leaf disease dataset from Kaggle.
    - Unzip and organize the dataset into appropriate directories for training, validation, and testing.

2. **Data Preprocessing**  
    - Split the dataset into training, validation, and test sets.
    - Apply data augmentation techniques such as rotation, zoom, and flipping to enhance the diversity of the training data.

3. **Model Design**  
    - Design a Convolutional Neural Network (CNN) architecture with layers for feature extraction and classification.
    - Include dropout layers to prevent overfitting and improve generalization.

4. **Model Compilation**  
    - Compile the model using the Adam optimizer and sparse categorical cross-entropy loss function.
    - Define accuracy as the evaluation metric.

5. **Model Training**  
    - Train the model using the training dataset and validate it using the validation dataset.
    - Use callbacks such as `ModelCheckpoint` to save the best-performing model during training.

6. **Model Evaluation**  
    - Evaluate the trained model on the test dataset to measure its accuracy and performance.
    - Analyze the results to identify any potential improvements.

7. **Model Saving**  
    - Save the trained model in a format suitable for deployment (e.g., `.h5` file).

8. **Testing and Validation**  
    - Test the model on unseen data to ensure it generalizes well.
    - Validate the model's predictions to confirm its reliability in real-world scenarios.
