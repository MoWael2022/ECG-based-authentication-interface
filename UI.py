import tkinter as tk
from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk, ImageSequence
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from testScribt import load_signal , main ,butter_Banpass_filter


root = Tk()
root.title("ECG based authentication interface")
root.configure(background='gray')
root.resizable(False, False)
root.geometry("1200x1000")

root.filename = filedialog.askopenfilename(title="Select a file")

# Read Signal
signal = load_signal(root.filename[0:-4])
filter_signal=butter_Banpass_filter(signal, Low_Cutoff=1.0, High_Cutoff=40.0 , SamplingRate=1000, order=2)
# plot Signal on GUI in a separate window
signal_fig = Figure(figsize=(6, 4))
signal_ax = signal_fig.add_subplot(111)
signal_ax.plot(filter_signal)
signal_ax.set_xlim(0, 4000)

signal_canvas = FigureCanvasTkAgg(signal_fig, master=root)
signal_canvas.get_tk_widget().pack()
signal_canvas.get_tk_widget().configure(width=800, height=400)
signal_canvas.draw()

# Create another container for everything but signal plot
container2 = tk.Frame(root, bg="Black")
container2.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# People on the system
Users = {0: "Salem", 1: "Mohamed", 2: "Ali", 3: "Fathy"}

# Create another container for everything but signal plot
container2 = tk.Frame(root, bg="white")
container2.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# Choose Feature Extraction method
model_name = IntVar()
Label(container2, text='model method', background='white', font=("Helvetica", 16)).grid(row=1, column=0,
                                                                                                     pady=30)

randomForest_radio_button = Radiobutton(container2, text="random forest", variable=model_name, value=1,
                                        font=("Helvetica", 16), background='white', bd=0, relief=tk.RAISED,
                                        highlightthickness=0)
randomForest_radio_button.grid(row=1, column=1, padx=10)

logisticRegression_radio_button = Radiobutton(container2, text="logisticRegression", variable=model_name,
                                        value=2,
                                        font=("Helvetica", 16), background='white', bd=0, relief=tk.RAISED,
                                        highlightthickness=0)
logisticRegression_radio_button.grid(row=1, column=2, padx=10)




# Login button
def execute():
    person_index = main(signal, model_name.get())
    if person_index == -1:  # which is impossible
        Label(container2, text='You are not Authorized', background='white', font=("Helvetica", 16)).grid(row=3,
                                                                                                          column=2,
                                                                                                          pady=30)
    else:
        message = 'hello, {name}'.format(name=Users[person_index])
        Label(container2, text=message, background='white', font=("Helvetica", 16)).grid(row=3,
                                                                                         column=2,
                                                                                         pady=30)
        update_frame(20)



Button(container2, text='Login', width=10, font=("Helvetica", 16), command=execute).grid(row=2, column=2, padx=30)

def update_frame(delay):
    print(f"Frame updated with a delay of {delay}ms")
# unlocked gif


mainloop()
