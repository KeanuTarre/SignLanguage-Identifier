import numpy as np
import cv2
import os
import sys

import time
import operator
from string import ascii_uppercase
import tkinter as tk
import mediapipe as mp
from PIL import Image, ImageTk
import enchant
from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Define the models directory path
MODELS_DIR = r"C:\Users\keanu\Software-Engineering_ASL_Detection\Sign-Language-To-Text-Conversion-main\Models"

class Application:
    def __init__(self):
        # Setup PyEnchant dictionary
        self.dictionary = enchant.Dict("en_US")
        
        # Setup MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        
        # Model loading
        try:
            # Main model
            model_path = os.path.join(MODELS_DIR, "model_new.json")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Could not find model file at {model_path}")
            
            self.json_file = open(model_path, "r")
            self.model_json = self.json_file.read()
            self.json_file.close()
            self.loaded_model = model_from_json(self.model_json)
            self.loaded_model.load_weights(os.path.join(MODELS_DIR, "model_new.h5"))

            # DRU model
            model_dru_path = os.path.join(MODELS_DIR, "model-bw_dru.json")
            if not os.path.exists(model_dru_path):
                raise FileNotFoundError(f"Could not find DRU model file at {model_dru_path}")
            
            self.json_file_dru = open(model_dru_path, "r")
            self.model_json_dru = self.json_file_dru.read()
            self.json_file_dru.close()
            self.loaded_model_dru = model_from_json(self.model_json_dru)
            self.loaded_model_dru.load_weights(os.path.join(MODELS_DIR, "model-bw_dru.h5"))

            # TKDI model
            model_tkdi_path = os.path.join(MODELS_DIR, "model-bw_tkdi.json")
            if not os.path.exists(model_tkdi_path):
                raise FileNotFoundError(f"Could not find TKDI model file at {model_tkdi_path}")
            
            self.json_file_tkdi = open(model_tkdi_path, "r")
            self.model_json_tkdi = self.json_file_tkdi.read()
            self.json_file_tkdi.close()
            self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
            self.loaded_model_tkdi.load_weights(os.path.join(MODELS_DIR, "model-bw_tkdi.h5"))

            # SMN model
            model_smn_path = os.path.join(MODELS_DIR, "model-bw_smn.json")
            if not os.path.exists(model_smn_path):
                raise FileNotFoundError(f"Could not find SMN model file at {model_smn_path}")
            
            self.json_file_smn = open(model_smn_path, "r")
            self.model_json_smn = self.json_file_smn.read()
            self.json_file_smn.close()
            self.loaded_model_smn = model_from_json(self.model_json_smn)
            self.loaded_model_smn.load_weights(os.path.join(MODELS_DIR, "model-bw_smn.h5"))

            print("All models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            sys.exit(1)

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        # UI Init
        self.root = tk.Tk()
        self.root.title("SIGN LANGUAGE DETECTOR DEMO")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x900")  # Increased width to accommodate 4 buttons
        self.root.resizable(height = None, width = None)
        self.root.configure(bg='#F0F4F8')

        # Fonts and Styling
        title_font = ("San Francisco", 24, "bold")
        label_font = ("San Francisco", 18)
        button_font = ("San Francisco", 16)

        style = {
            'bg_main': '#FFFFFF',
            'border_color': '#E0E4E8',
            'text_color': '#2C3E50',
            'accent_color': '#3498DB',
            'backspace_color': '#E74C3C'
        }

        # Main Video Panel
        self.panel = tk.Label(self.root, bg=style['bg_main'], relief=tk.RAISED, borderwidth=2)
        self.panel.place(x=50, y=50, width=800, height=550)
        self.panel.config(highlightthickness=2, highlightbackground=style['border_color'])

        # Information Panel
        info_panel_x = 900
        
        # Character Display
        #self.panel3 = tk.Label(self.root, 
        #                       font=title_font, 
        #                       fg=style['accent_color'], 
        #                       bg='#F0F4F8')
        #self.panel3.place(x=info_panel_x, y=150)

        # Word Display
        self.T2 = tk.Label(self.root, text="Word:", 
                           font=label_font, 
                           fg=style['text_color'], 
                           bg='#F0F4F8')
        self.T2.place(x=info_panel_x, y=259970)

        self.panel4 = tk.Label(self.root, 
                               font=title_font, 
                               fg=style['accent_color'], 
                               bg='#F0F4F8')
        self.panel4.place(x=info_panel_x, y=300)
        
        # Word Backspace Button
        self.word_backspace_btn = tk.Button(
            self.root, 
            text='⌫', 
            font=button_font,
            bg=style['backspace_color'], 
            fg='white', 
            command=self.backspace_word
        )
        self.word_backspace_btn.place(x=1300, y=300, width=60, height=50)

        # Sentence Display
        self.T3 = tk.Label(self.root, text="Sentence:", 
                           font=label_font, 
                           fg=style['text_color'], 
                           bg='#F0F4F8')
        self.T3.place(x=info_panel_x, y=400)

        self.panel5 = tk.Label(self.root, 
                               font=title_font, 
                               fg=style['accent_color'], 
                               bg='#F0F4F8')
        self.panel5.place(x=info_panel_x, y=450)
        
        # Sentence Backspace Button
        self.sentence_backspace_btn = tk.Button(
            self.root, 
            text='⌫', 
            font=button_font,
            bg=style['backspace_color'], 
            fg='white', 
            command=self.backspace_sentence
        )
        self.sentence_backspace_btn.place(x=1300, y=450, width=65, height=50)

        # Button styling
        button_style = {
            'font': button_font,
            'bg': style['bg_main'],
            'fg': style['accent_color'],
            'relief': tk.RAISED,
            'borderwidth': 1,
            'highlightthickness': 1,
            'highlightbackground': style['border_color']
        }

        # 4 Suggestion Buttons
        self.bt1 = tk.Button(self.root, **button_style, command=lambda: self.action_button(0))
        self.bt1.place(x=50, y=640, width=200, height=50)

        self.bt2 = tk.Button(self.root, **button_style, command=lambda: self.action_button(1))
        self.bt2.place(x=270, y=640, width=200, height=50)

        self.bt3 = tk.Button(self.root, **button_style, command=lambda: self.action_button(2))
        self.bt3.place(x=490, y=640, width=200, height=50)

        self.bt4 = tk.Button(self.root, **button_style, command=lambda: self.action_button(3))
        self.bt4.place(x=710, y=640, width=200, height=50)

        # Initialize variables
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.suggestion_list = []
        self.predicted_symbol = ""
        self.video_loop()

    def backspace_word(self):
        """Remove the last character from the current word"""
        if self.word:
            self.word = self.word[:-1]
        
        # Update word display
        self.panel4.config(text=self.word)

    def backspace_sentence(self):
        """Remove the last word from the sentence"""
        # Split the sentence into words
        words = self.str.split()
        
        # Remove the last word if sentence is not empty
        if words:
            words.pop()
            
            # Reconstruct the sentence
            self.str = " ".join(words)
        
        # Update sentence display
        self.panel5.config(text=self.str)
        
    def get_suggestions(self, word):
        """Get spelling suggestions using PyEnchant"""
        if not word:
            return []
        
        # If the word is not in the dictionary, get suggestions
        if not self.dictionary.check(word):
            suggestions = self.dictionary.suggest(word)
            return suggestions[:3]  # Top 3 suggestions
        return []

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = self.hands.process(rgb_frame)
            
            # Reset hand bbox
            self.hand_bbox = None
            
            # Convert back to BGR for display
            frame_to_display = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # If hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get bounding box of the hand
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    
                    # Add some padding
                    padding = 20
                    x_min = max(0, int(x_min - padding))
                    x_max = min(frame.shape[1], int(x_max + padding))
                    y_min = max(0, int(y_min - padding))
                    y_max = min(frame.shape[0], int(y_max + padding))
                    
                    # Store bounding box for later use
                    self.hand_bbox = (x_min, y_min, x_max, y_max)
                    
                    # Crop the hand region
                    hand_crop = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    
                    # Process the hand region 
                    gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 2)
                    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # Display the grayscale and processed images
                    #cv2.imshow("Grayscale Hand", gray)
                    #cv2.imshow("Processed Hand", res)
                    
                    # Resize to match model input
                    res_resized = cv2.resize(res, (128, 128))
                    
                    # Predict
                    self.predict(res_resized)
                    
                    # Draw a thin green box around the hand
                    cv2.rectangle(frame_to_display, 
                                (int(x_min), int(y_min)), 
                                (int(x_max), int(y_max)), 
                                (255, 0, 0),  # Red color
                                2)  # Thin line width
                    
                    # Display the predicted letter
                    cv2.putText(frame_to_display, 
                                f"Predicted: {self.current_symbol}", 
                                (int(x_min), int(y_min) - 10),  # Position above the box
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9,  # Font scale
                                (255, 0, 0),  # Text color (red)
                                2)  # Thickness
            
            # Convert frame for Tkinter display
            cv2image = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            # Update panels
            #self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            # Update suggestion buttons
            self.suggestion_list = self.get_suggestions(self.word.lower())
            
            # First 3 buttons: spelling suggestions
            for i in range(3):
                button = getattr(self, f'bt{i+1}')
                button.config(
                    text=self.suggestion_list[i] if i < len(self.suggestion_list) else "", 
                    font=("Courier", 20)
                )
            
            # 4th button: predicted symbol
            self.bt4.config(
                text=self.current_symbol, 
                font=("Courier", 20)
            )

        self.root.after(5, self.video_loop)

    def action_button(self, button_index):
        """Handle button click logic"""
        if button_index < 3:
            # Spelling suggestion buttons
            if button_index < len(self.suggestion_list):
                # Add suggested word
                suggestion = self.suggestion_list[button_index]
                self.str += " " + suggestion
                self.word = ""
        else:
            # Predicted symbol button
            if self.current_symbol != "Empty":
                self.word += self.current_symbol

        # Clear previous suggestions
        self.suggestion_list = []

    def predict(self, test_image):
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {}
        prediction['blank'] = result[0][0]
        
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {
                'D': result_dru[0][0],
                'R': result_dru[0][1],
                'U': result_dru[0][2]
            }
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {
                'D': result_tkdi[0][0],
                'I': result_tkdi[0][1],
                'K': result_tkdi[0][2],
                'T': result_tkdi[0][3]
            }
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {
                'M': result_smn[0][0],
                'N': result_smn[0][1],
                'S': result_smn[0][2]
            }
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        
        self.ct[self.current_symbol] += 1
        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def action1(self):
        predicts = self.get_suggestions(self.word.lower())
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]

    def action2(self):
        predicts = self.get_suggestions(self.word.lower())
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):
        predicts = self.get_suggestions(self.word.lower())
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        self.hands.close()
        cv2.destroyAllWindows()

print("Starting Application...")
(Application()).root.mainloop()