import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Directories
DATA_DIR = os.path.expanduser("~/Desktop/Signy/data")
SAVE_DIR = os.path.expanduser("~/Desktop/Signy")

# Step 1: Collect Data
def collect_data(n_classes=10, dataset_size=100):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    cap = cv2.VideoCapture(0)
    for i in range(n_classes):
        class_dir = os.path.join(DATA_DIR, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        print(f'Collecting data for class {i}')
        
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "K" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('k'):
                break
        
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Process and Save Data
def process_and_save_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    
    data, labels = [], []
    
    for dir_ in os.listdir(DATA_DIR):
        if dir_.startswith('.') or not os.path.isdir(os.path.join(DATA_DIR, dir_)):
            continue

        class_dir = os.path.join(DATA_DIR, dir_)
        for img_name in os.listdir(class_dir):
            if img_name.startswith('.') or not os.path.isfile(os.path.join(class_dir, img_name)):
                continue
            
            img = cv2.imread(os.path.join(class_dir, img_name))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    for landmark in hand_landmarks.landmark:
                        data_aux.extend([landmark.x, landmark.y])
                    data.append(data_aux)
                    labels.append(int(dir_))
    
    with open(os.path.join(SAVE_DIR, 'data.pickle'), 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

# Step 3: Train Model
def train_model():
    with open(os.path.join(SAVE_DIR, 'data.pickle'), 'rb') as f:
        data_dict = pickle.load(f)
    
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    print(f"Model Accuracy: {score}")
    
    with open(os.path.join(SAVE_DIR, 'model.p'), 'wb') as f:
        pickle.dump({'model': model}, f)

# Step 4: Real-time Prediction
def real_time_prediction():
    with open(os.path.join(SAVE_DIR, 'model.p'), 'rb') as f:
        model = pickle.load(f)['model']

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'V', 1: 'Nice', 2: 'V', 3: 'I', 4: 'L', 5: 'C', 6: 'U', 7: '0', 8: 'I', 9: '6'}
    
    while True:
        ret, frame = cap.read()
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

                x_, y_, data_aux = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x - min(x_), landmark.y - min(y_)])
                
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
                
                pred = model.predict([np.asarray(data_aux)])
                pred_char = labels_dict[int(pred[0])]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, pred_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('k'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Menu
def menu():
    while True:
        print("Menu:")
        print("1. Collect Data")
        print("2. Process and Save Data")
        print("3. Train Model")
        print("4. Real-time Prediction")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            collect_data()
        elif choice == '2':
            process_and_save_data()
        elif choice == '3':
            train_model()
        elif choice == '4':
            real_time_prediction()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
