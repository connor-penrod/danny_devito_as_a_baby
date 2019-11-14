from tkinter import *
from tkinter import font

class gui(Frame):
    def __init__(self, parent=None):
        Frame.__init__(self)
        self.parent = parent
        self.winfo_toplevel().geometry("500x300")
        self.pack()
        self.make_widgets()

    def make_widgets(self):
        win = self.winfo_toplevel()
        win.title("Danny Devito as a Baby")
        helv24 = font.Font(family='Helvetica', size=24,  weight='bold')
        upload = Button(win, text="Upload Image", command=print("upload"), font=helv24)
        nxt = Button(win, text="Swap", font=helv24)
        upload.pack(side=LEFT)
        nxt.pack(side=LEFT)
        



if __name__ == "__main__":
    root = Tk()
    daab = gui(root)
    root.mainloop()
