import os
from AI import NumberAI
from GUI import GUI
import threading
import numpy as np

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def app_bridge(gui, number_ai):
    while gui.running:
        prediction_array = []
        
        # Only make the AI predict if the array is not empty
        if np.any(gui.canvas.np_arry):
            prediction_array.append(np.reshape(gui.canvas.np_arry,(1, 28*28)))
            prediction_output = number_ai.predict(prediction_array)
            
            gui.prediction_text[1] = prediction_output[0]
            gui.prediction_text[3] = prediction_output[1]
            gui.predicted_prob = prediction_output[2]
       
        else:
            gui.reset_prediction_output()
        cls()

if __name__ == "__main__":
    number_ai = NumberAI()
    number_ai.fit()

    window_scale_factor = 25
    window_scale = ([window_scale_factor, window_scale_factor])

    gui = GUI(window_scale)
    
    ai_thread = threading.Thread(target=app_bridge, args=(gui, number_ai))
    ai_thread.start()
    gui.render()
    

