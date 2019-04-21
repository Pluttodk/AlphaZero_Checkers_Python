import tkinter
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Plots:
    def __init__(self):
        tk = tkinter.Tk()
        tk.title("Performance Plots")
        w, h = 300,200
        pyplot.ion()
        pyplot.hist([0,0,0,1,1,10,0,1,0])
        fig = pyplot.figure(1)
        cv1 = FigureCanvasTkAgg(fig, master=tk)
        self.tk = tk


if __name__ == '__main__':
    plt = Plots()
    plt.tk.mainloop()
